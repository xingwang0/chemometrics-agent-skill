# Copyright 2026 The research-ml-modeling Authors
# SPDX-License-Identifier: Apache-2.0
"""三层交叉验证引擎：外层重复分组 K 折 → 内层分组 K 折调参 → 外层测试折评估。

为什么要三层
------------
* **分组**（GroupKFold by sample_id）：同一样品的多次测量必须同进同出，否则模型在
  "用这个样品的另一次测量预测这个样品"，R² 虚高——这是光谱/材料论文最常见的致命伤。
* **嵌套**：调参必须在内层完成，外层只用最优模型评估。否则报出去的是"调参调出来的
  最好成绩"，审稿人一问"你的 R² 是不是挑出来的"就无法回答。
* **重复**（不同随机种子 R 次）：证明结论不是某一次划分的运气。聚合时先按 repeat 求
  折均值、再对 R 个均值求 sd——直接混算 R×k 个折是伪重复，sd 被低估 2–3 倍。

内层的落地细节：``GridSearchCV`` 的 ``cv`` 参数收**显式 split 列表**。这是必须的——
内层没有 groups 通道（除非启用 metadata routing，而那会改变全局行为）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold, GroupShuffleSplit, KFold, LeaveOneGroupOut, StratifiedGroupKFold,
)

from .config import ConfigError
from .models import ModelSpec, build_pipeline, prep_label, search_space
from .preprocess import prep_name
from .tune import count_candidates, expand_space, inner_search, scoring_for
from .utils import info, warn

__all__ = [
    "make_outer_splits", "make_inner_splits", "run_nested_cv",
    "estimate_fit_calls", "describe_split_plan", "effective_repeats",
]

MAX_INNER_FAILURES = 5


def effective_repeats(cfg: dict) -> int:
    """留一批次 / 固定留出批次没有随机性，重复跑只会得到同一划分。

    旧实现按 repeats 重跑，产生完全相同的折（sd=0、白耗算力，还让报告里的
    n_repeats 看起来像做了多次独立验证）。这里统一降为 1 并明确告警。
    """
    repeats = int(cfg["split"]["repeats"])
    if cfg["split"]["strategy"] in ("loo_batch", "holdout_batch") and repeats > 1:
        warn(f"split.strategy={cfg['split']['strategy']} 的折划分是确定的，"
             f"repeats={repeats} 会重复同一划分；已自动降为 1")
        return 1
    return repeats


# --------------------------------------------------------------------------- #
# 划分器
# --------------------------------------------------------------------------- #
def _check_group_count(n_groups: int, n_splits: int, label: str) -> int:
    if n_groups < 2:
        raise ConfigError("E-CV-001", f"{label}：独立样品只有 {n_groups} 个，无法交叉验证")
    if n_splits > n_groups:
        warn(f"{label}：折数 {n_splits} > 独立样品数 {n_groups}，自动降为 {n_groups}")
        return n_groups
    return n_splits


def make_outer_splits(ds, cfg: dict, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    strategy = cfg["split"]["strategy"]
    groups = np.asarray(ds.groups, dtype=object)
    n_groups = len(pd.unique(groups))

    if strategy == "holdout_random":
        # 旧实现算出的 n_test / n_splits_groups 根本没被使用，且 test_size=None
        # 让 GroupShuffleSplit 回落到默认 0.2，配置里的 test_size 被静默忽略。
        test_size = float(cfg["split"].get("test_size", 0.2))
        if not (0.0 < test_size < 1.0):
            raise ConfigError("E-CV-004",
                              f"split.test_size 必须在 (0,1) 内，收到 {test_size}")
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        return list(gss.split(ds.X, ds.y, groups=groups))

    if strategy in ("loo_batch", "holdout_batch"):
        if ds.batch is None:
            raise ConfigError("E-CV-002", f"strategy={strategy} 需要 data.batch_col")
        batch = np.asarray(ds.batch, dtype=object)
        if strategy == "loo_batch":
            n_batch = len(pd.unique(batch))
            _check_group_count(n_batch, n_batch, "留一批次")
            return list(LeaveOneGroupOut().split(ds.X, ds.y, groups=batch))
        target_batch = str(cfg["split"]["holdout_batch"])
        te = np.flatnonzero(batch == target_batch)
        tr = np.flatnonzero(batch != target_batch)
        if te.size == 0 or tr.size == 0:
            raise ConfigError("E-CV-003", f"holdout_batch={target_batch!r} 在数据中不存在或占满了全部数据")
        return [(tr, te)]

    # nested_group
    n_splits = _check_group_count(n_groups, int(cfg["split"]["outer"]["n_splits"]), "外层分组")
    stratify = bool(cfg["split"].get("stratify")) and ds.task == "classification"
    if stratify:
        try:
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                      random_state=seed)
            return list(cv.split(ds.X, ds.y, groups=groups))
        except (ValueError, TypeError) as exc:
            warn(f"StratifiedGroupKFold 不可用（{exc}），退回 GroupKFold")
    cv = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(ds.X, ds.y, groups=groups))


def make_inner_splits(
    y_tr: np.ndarray,
    g_tr: np.ndarray,
    cfg: dict,
    task: str,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """在训练折内部再按 group 划分。返回显式 split 列表（索引是训练折内的局部下标）。"""
    n_splits = int(cfg["split"]["inner"]["n_splits"])
    n_groups = len(pd.unique(g_tr))
    n_splits = max(2, min(n_splits, n_groups))
    X_dummy = np.zeros((y_tr.size, 1), dtype=np.float32)

    if n_groups < 2 or y_tr.size < 4:
        # 极小的训练折：退化成按行 K 折（并在 fold 记录里注明）
        k = max(2, min(2, y_tr.size))
        return list(KFold(n_splits=k, shuffle=True, random_state=seed).split(X_dummy, y_tr))

    stratify = bool(cfg["split"].get("stratify")) and task == "classification"
    if stratify:
        try:
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            splits = list(cv.split(X_dummy, y_tr, groups=g_tr))
            if len(splits) >= 2:
                return splits
        except (ValueError, TypeError):
            pass
    try:
        cv = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(cv.split(X_dummy, y_tr, groups=g_tr))
        if len(splits) >= 2:
            return splits
    except ValueError:
        pass
    return list(KFold(n_splits=2, shuffle=True, random_state=seed).split(X_dummy, y_tr))


# --------------------------------------------------------------------------- #
# 预估成本（--dry-run 用）
# --------------------------------------------------------------------------- #
def _candidates_per_fold(spec: ModelSpec, cfg: dict, seed: int) -> int:
    space = search_space(spec)
    cands = len(cfg["preprocess"]["candidates"])
    if cands > 1:
        space = {"prep__which": list(range(cands)), **space}
    st = spec.search_type
    if st == "grid":
        return min(count_candidates(space), 240)
    if st == "none":
        return 1
    return int(spec.n_iter)


def _outer_fold_count(ds, cfg: dict, mask: np.ndarray | None = None) -> int:
    """该目标实际会跑几个外层折（多目标各自的有效行/组数可能不同）。"""
    strategy = cfg["split"]["strategy"]
    if strategy == "holdout_random":
        return 1
    groups = np.asarray(ds.groups, dtype=object)
    batch = None if ds.batch is None else np.asarray(ds.batch, dtype=object)
    if mask is not None:
        groups = groups[mask]
        if batch is not None:
            batch = batch[mask]
    if strategy in ("loo_batch", "holdout_batch") and batch is not None:
        return len(pd.unique(batch)) if strategy == "loo_batch" else 1
    n_groups = len(pd.unique(groups))
    return max(2, min(int(cfg["split"]["outer"]["n_splits"]), n_groups))


def estimate_fit_calls(ds, cfg: dict, specs: list[ModelSpec]) -> dict[str, Any]:
    repeats = effective_repeats(cfg)
    n_inner = int(cfg["split"]["inner"]["n_splits"])
    per_target: dict[str, dict[str, int]] = {}
    total_outer = 0
    for tname in ds.target_names:
        y = np.asarray(ds.targets[tname])
        valid = np.isfinite(y) if ds.task == "regression" else ~pd.isna(y)
        n_outer_t = _outer_fold_count(ds, cfg, valid)
        groups_t = np.asarray(ds.groups, dtype=object)[valid]
        per_target[tname] = {
            "n_rows": int(valid.sum()),
            "n_groups": int(len(pd.unique(groups_t))),
            "outer_folds": int(n_outer_t),
        }
        total_outer += n_outer_t
    per_model = {}
    total = 0
    for spec in specs:
        c = _candidates_per_fold(spec, cfg, 0)
        calls = repeats * total_outer * c * n_inner
        per_model[spec.name] = {"candidates_per_fold": c, "inner_folds": n_inner,
                                "outer_folds": total_outer, "repeats": repeats,
                                "fit_calls": calls}
        total += calls
    perm = cfg["permutation"]
    perm_calls = 0
    if perm.get("enabled") and perm.get("mode") == "fast":
        perm_calls = int(perm["n_perm"]) * repeats * total_outer
    return {"per_model": per_model, "per_target": per_target,
            "total_inner_fit_calls": total,
            "permutation_fit_calls_est": perm_calls,
            "grand_total_est": total + perm_calls}


def describe_split_plan(ds, cfg: dict) -> dict[str, Any]:
    strategy = cfg["split"]["strategy"]
    plan = {
        "strategy": strategy,
        "n_rows": ds.n_samples,
        "n_groups": ds.n_groups,
        "repeats": effective_repeats(cfg),
        "outer_n_splits": int(cfg["split"]["outer"]["n_splits"]),
        "inner_n_splits": int(cfg["split"]["inner"]["n_splits"]),
        "group_col": cfg["data"].get("id_col") or "auto_fingerprint",
        "n_rows_by_target": {
            t: int(ds.target_available.get(t, ds.n_samples)) for t in ds.target_names
        },
        "n_groups_by_target": {
            t: int(len(pd.unique(np.asarray(ds.groups, dtype=object)[
                np.isfinite(np.asarray(ds.targets[t]))
                if ds.task == "regression" else ~pd.isna(np.asarray(ds.targets[t]))
            ])))
            for t in ds.target_names
        },
        "note": "",
    }
    if strategy == "nested_group":
        if ds.batch is not None:
            plan["note"] = "外层分组 K 折（按 sample_id）；如需真外推，另跑 strategy=loo_batch"
        else:
            plan["note"] = "外层分组 K 折；无 batch 列，无跨批次外推验证"
    elif strategy == "loo_batch":
        plan["note"] = "留一批次外验证：每折训一批、测另一批，这是审稿人最认的真外推"
    elif strategy == "holdout_batch":
        plan["note"] = f"固定留出批次 {cfg['split'].get('holdout_batch')} 作为测试集"
    return plan


# --------------------------------------------------------------------------- #
# 主执行
# --------------------------------------------------------------------------- #
PROBA_NATIVE = "native"     # 真正的 predict_proba（可算 Brier/ECE/log-loss）
PROBA_DECISION = "decision"  # 只有 decision_function → sigmoid 单调映射，仅可算排序类指标
PROBA_NONE = "none"          # 只有 predict → 硬标签，排序指标也退化


def _predict(est, X: np.ndarray, task: str) -> tuple[np.ndarray, np.ndarray | None, str]:
    """返回 ``(预测标签/值, 概率或得分, 概率能力类型)``。

    概率能力必须显式带出来：拿 sigmoid(decision_function) 当概率去算 Brier/ECE
    不会报错，但结果是错的（那不是概率）。三态让下游能拒绝对非原生概率算概率类指标。
    """
    pred = np.asarray(est.predict(X))
    if task != "classification":
        return pred, None, PROBA_NONE
    if hasattr(est, "predict_proba"):
        try:
            return pred, np.asarray(est.predict_proba(X)), PROBA_NATIVE
        except Exception:  # noqa: BLE001
            pass
    if hasattr(est, "decision_function"):
        try:
            df = np.asarray(est.decision_function(X), dtype=np.float64)
            if df.ndim == 1:
                # 单调变换，AUC/AP 完全不变；但绝不等于概率
                return pred, 1.0 / (1.0 + np.exp(-df)), PROBA_DECISION
            z = df - df.max(axis=1, keepdims=True)
            e = np.exp(z)
            return pred, e / e.sum(axis=1, keepdims=True), PROBA_DECISION
        except Exception:  # noqa: BLE001
            pass
    return pred, None, PROBA_NONE


def _score_fold(task: str, y_true: np.ndarray, pred: np.ndarray,
                proba: np.ndarray | None, n_classes: int,
                proba_kind: str = PROBA_NATIVE) -> dict[str, float]:
    from .metrics import classification_metrics, regression_metrics

    if task == "regression":
        return regression_metrics(y_true, pred)
    score = proba if proba is not None else pred.astype(float)
    return classification_metrics(y_true, score, n_classes=n_classes, proba_kind=proba_kind)


def run_nested_cv(
    ctx,
    specs: list[ModelSpec],
    progress: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """执行完整嵌套 CV，返回 (oof_df, fold_df)。

    oof_df 是全链路唯一真源：指标、置信区间、模型比较、适用域差异、所有图都从它重算。
    """
    cfg = ctx.cfg
    ds = ctx.ds
    task = ds.task
    repeats = effective_repeats(cfg)
    master = int(cfg["runtime"]["seed"])
    n_jobs = int(cfg["runtime"].get("n_jobs", -1))
    fail_fast = bool(cfg["runtime"].get("fail_fast", False))
    n_classes = 1 if task == "regression" else int(len(ds.classes or [0, 1]))
    save_policy = str(cfg["runtime"].get("save_models", "last"))
    model_dir = ctx.run_dir / "cv" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    target_names = ds.target_names
    oof_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    total_fits = 0

    for tname in target_names:
        y_full = np.asarray(ds.targets[tname])
        # 多目标共用一份 X，但每个目标只应在自己 y 有效的行上建模。
        # 旧写法用全表索引，兄弟目标缺一行会把这个目标也少一个样品（且无提示）。
        valid = (np.isfinite(y_full) if task == "regression"
                 else ~pd.isna(y_full))
        if not valid.all():
            tds = ds.subset(valid)
            info(f"[{tname}] 按目标有效行取子集：{int(valid.sum())}/{int(valid.size)} 行"
                 f"（其余目标的缺失行不再连带剔除）")
        else:
            tds = ds
        y_all = np.asarray(tds.targets[tname])
        for r in range(repeats):
            seed_r = master + r * 1000
            outer = make_outer_splits(tds, cfg, seed_r)
            for k, (tr, te) in enumerate(outer):
                y_tr_all, y_te_all = y_all[tr], y_all[te]
                g_tr = np.asarray(tds.groups, dtype=object)[tr]
                inner = make_inner_splits(y_tr_all, g_tr, cfg, task, seed=seed_r * 100 + k)

                for spec in specs:
                    # 种子方案：legacy 保持旧式（模型数 ≥10 时会与下一折 0 号模型撞种子，
                    # 属已记录缺陷 P0-5），per_model 把模型维度乘到 100 上彻底隔开。
                    # 默认 legacy —— 显式开开关才改抽样序列，保证既有结果可复现。
                    scheme = str(cfg["runtime"].get("seed_scheme", "legacy")).lower()
                    if scheme == "per_model":
                        seed_fold = (seed_r * 100 + k) * 100 + specs.index(spec)
                    else:
                        seed_fold = (seed_r * 100 + k) * 10 + specs.index(spec)
                    pipe = build_pipeline(spec, cfg, tds.feature_names, seed=seed_fold)
                    space = search_space(spec)
                    n_cands = len(cfg["preprocess"]["candidates"])
                    if n_cands > 1:
                        space = {"prep__which": list(range(n_cands)), **space}
                    stratify = bool(cfg["preprocess"].get("stratify_sampling", False))
                    param_dicts = expand_space(
                        space, spec.search_type, spec.n_iter,
                        seed=seed_fold, max_grid=240,
                        stratify_key=("prep__which" if (n_cands > 1 and stratify) else None),
                    )
                    msg = (f"[{tname}] repeat {r + 1}/{repeats} fold {k + 1}/{len(outer)} "
                           f"model={spec.name} 候选={len(param_dicts)}")
                    if progress:
                        progress(msg)

                    scoring = scoring_for(pipe, task)   # 能力在估计器上，不在任务上

                    def _fail_row(exc: BaseException, stage: str) -> dict[str, Any]:
                        return {
                            "target": tname, "repeat": r, "fold": k, "model": spec.name,
                            "prep": "FAILED", "n_train": int(tr.size), "n_test": int(te.size),
                            "best_inner_score": float("nan"), "best_params": "{}",
                            "status": f"error:{type(exc).__name__}",
                            "error": f"{stage}｜{type(exc).__name__}: {str(exc)[:300]}",
                            "scoring": scoring,
                        }

                    # 失败边界必须覆盖 fit **和** predict：knn(n_neighbors > n_train)
                    # 这类错误只在 predict 时爆，旧写法会让整轮 90 折的算力全丢。
                    try:
                        est, best_params, best_score, _ = inner_search(
                            pipe, param_dicts, inner, tds.X[tr], y_tr_all,
                            scoring=scoring, n_jobs=n_jobs, fail_fast=fail_fast,
                        )
                        total_fits += len(param_dicts) * len(inner) + 1
                        pred_te, proba_te, kind_te = _predict(est, tds.X[te], task)
                        pred_tr, proba_tr, kind_tr = _predict(est, tds.X[tr], task)
                        m_te = _score_fold(task, y_te_all, pred_te, proba_te, n_classes, kind_te)
                        m_tr = _score_fold(task, y_tr_all, pred_tr, proba_tr, n_classes, kind_tr)
                    except Exception as exc:  # noqa: BLE001 —— 单折失败不应毁掉整轮
                        if fail_fast:
                            raise
                        warn(f"{msg} 失败：{type(exc).__name__}: {exc}")
                        fold_rows.append(_fail_row(exc, "fit/predict"))
                        continue

                    prep_idx = best_params.get("prep__which", 0)
                    prep_lab = prep_label(cfg, int(prep_idx)) if n_cands > 1 else prep_name(
                        cfg["preprocess"]["candidates"][0])
                    status = "ok" if np.isfinite(m_te.get("r2" if task == "regression" else "auc",
                                                          float("nan"))) else "degenerate"

                    fold_row: dict[str, Any] = {
                        "target": tname, "repeat": r, "fold": k, "model": spec.name,
                        "prep": prep_lab, "n_train": int(tr.size), "n_test": int(te.size),
                        "n_train_groups": int(len(pd.unique(g_tr))),
                        "best_inner_score": float(best_score),
                        "best_params": _json_dumps(best_params),
                        "status": status,
                        "scoring": scoring,
                        "proba_kind": kind_te,
                    }
                    for key, val in m_te.items():
                        fold_row[key] = val
                    if task == "regression":
                        fold_row["r2_train"] = m_tr.get("r2", float("nan"))
                        fold_row["rmse_train"] = m_tr.get("rmse", float("nan"))
                    else:
                        fold_row["auc_train"] = m_tr.get("auc", float("nan"))
                    fold_rows.append(fold_row)

                    # OOF 明细（逐行）
                    groups_te = np.asarray(tds.groups, dtype=object)[te]
                    batch_te = None if tds.batch is None else np.asarray(tds.batch, dtype=object)[te]
                    for i, row_idx in enumerate(te):
                        rec = {
                            "target": tname, "repeat": r, "fold": k, "model": spec.name,
                            "prep": prep_lab, "row_id": tds.row_ids[row_idx],
                            "group": groups_te[i],
                            "batch": None if batch_te is None else batch_te[i],
                            "y_true": float(y_all[row_idx]),
                            "y_pred": float(pred_te[i]),
                        }
                        if task == "classification":
                            rec["proba_kind"] = kind_te
                            if proba_te is None:
                                # 没有概率也没有 decision_function：不要把硬标签冒充概率，
                                # 留空让下游按 y_pred 处理，并在指标里屏蔽概率类指标
                                rec["proba"] = None
                            elif proba_te.ndim == 2 and proba_te.shape[1] == 2:
                                rec["proba"] = float(proba_te[i, 1])
                            elif proba_te.ndim == 2:
                                for c in range(proba_te.shape[1]):
                                    rec[f"p{c}"] = float(proba_te[i, c])
                                rec["proba"] = float(proba_te[i].max())
                            else:
                                rec["proba"] = float(np.ravel(proba_te)[i])
                        oof_rows.append(rec)

                    # 模型落盘策略
                    if save_policy == "all" or (save_policy == "last" and k == len(outer) - 1):
                        try:
                            import joblib

                            # compress=1：光谱上的树模型几百 MB，compress=3 会把 CPU
                            # 全花在压缩上（几十秒/个 × 数十个），对复现没任何增益。
                            joblib.dump(est, model_dir / f"{_safe(tname)}_{_safe(spec.name)}_r{r}f{k}.joblib",
                                        compress=1)
                        except Exception as exc:  # noqa: BLE001
                            warn(f"模型保存失败（不影响结果）：{exc}")

    oof = pd.DataFrame(oof_rows)
    folds = pd.DataFrame(fold_rows)
    cfg["runtime"]["_total_fit_calls"] = total_fits
    return oof, folds


def _json_dumps(obj: Any) -> str:
    import json

    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except TypeError:
        return "{}"


def _safe(text: str) -> str:
    keep = [c if (c.isalnum() or c in "-_") else "_" for c in str(text)]
    return "".join(keep)[:40] or "m"


def oof_group_metrics(
    oof: pd.DataFrame,
    task: str,
    n_classes: int,
    group_keys: tuple[str, ...] = ("target", "model", "prep"),
) -> pd.DataFrame:
    """在 OOF 上按 ``group_keys`` 汇总指标（用于与 fold 级结果对照）。"""
    from .metrics import classification_metrics, oof_score_matrix, regression_metrics

    rows = []
    keys = [k for k in group_keys if k in oof.columns]
    for name, sub in oof.groupby(keys, dropna=False):
        name_tuple = name if isinstance(name, tuple) else (name,)
        row = dict(zip(keys, name_tuple))
        if task == "regression":
            row.update(regression_metrics(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy()))
        else:
            score, kind = oof_score_matrix(sub, n_classes)
            row.update(classification_metrics(sub["y_true"].to_numpy(), score,
                                              n_classes=n_classes, proba_kind=kind))
        row["n_oof_rows"] = int(len(sub))
        rows.append(row)
    return pd.DataFrame(rows)
