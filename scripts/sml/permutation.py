# Copyright 2026 The research-ml-modeling Authors
# SPDX-License-Identifier: Apache-2.0
"""Y-randomization（置换检验）与特征/波段级置换零分布。

化学计量学的标准自证
--------------------
"你的模型是不是在拟合噪声？" 唯一有说服力的回答是 Y-randomization：
把 y 在**样品级**打乱，重跑整条流程，得到 R² 的零分布；若真实 R² 落在零分布之外
（通常 p<0.05 或 R²_intercept 为负），才能说明学到的关系不是随机产生的。

三个必须做对的地方
------------------
1. **样品级置换**：同一样品的多次测量共享同一个 y，置换也必须在样品级做，
   否则会造出"同一样品有不同 y"的数据，零分布被系统性偏移。
2. **分组结构不变**：只打乱 y，groups/splits 一律不动，保证零分布模型面对的
   折结构与真实模型完全一致。
3. **加一校正**：p = (1 + #{null ≥ obs}) / (1 + n_perm)，避免出现 p=0。
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from .cv import effective_repeats, make_outer_splits
from .metrics import METRIC_LABELS
from .models import ModelSpec, build_pipeline
from .utils import info, warn

__all__ = ["permute_y_sample_level", "y_randomization", "summarize_permutation"]

PRIMARY = {"regression": "r2", "classification": "auc"}


def permute_y_sample_level(
    y: np.ndarray,
    groups: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """在样品级打乱 y，再映射回每一行（同一样品的所有测量拿到同一个置换后 y）。"""
    y = np.asarray(y)
    groups = np.asarray(groups, dtype=object)
    uniq = pd.unique(groups)
    # 每个样品取第一个 y（同一样品的 y 本应一致；若不一致说明标签有问题）
    first_idx = pd.Series(np.arange(groups.size)).groupby(groups).first().reindex(uniq).to_numpy()
    y_sample = y[first_idx]
    permuted = rng.permutation(y_sample)
    mapping = dict(zip(uniq, permuted))
    return np.array([mapping[g] for g in groups])


def _clean_params(params: dict | None) -> dict:
    return {k: v for k, v in (params or {}).items()
            if k.startswith(("model__", "prep__", "scale__"))}


def _fit_prefix(cfg, spec: ModelSpec, params: dict | None, X_tr: np.ndarray, seed: int):
    """只拟合 Pipeline 的「插补 → 预处理 → 标准化」前缀，返回 transform 后的 X。

    置换检验里 y 变了、X 没变，而这三步都是**无监督**的：同一折的预处理结果
    对 n_perm 次置换完全通用。原实现每次置换都重建整条链，airPLS 这类算子
    在 2000 列 × 上千行上单次要几秒，5000 次置换就是纯浪费。
    """
    from sklearn.pipeline import Pipeline

    pipe = build_pipeline(spec, cfg, None, seed=seed)
    clean = {k: v for k, v in _clean_params(params).items() if k.startswith(("prep__", "scale__"))}
    if clean:
        pipe.set_params(**clean)
    prefix = Pipeline(pipe.steps[:-1])
    prefix.fit(X_tr)
    return prefix


def _model_from_params(cfg, spec: ModelSpec, params: dict | None, seed: int):
    """按固定超参构造裸模型（外层前缀已单独 fit）。"""
    model = build_pipeline(spec, cfg, None, seed=seed).steps[-1][1]
    clean = {k.removeprefix("model__"): v for k, v in _clean_params(params).items()
             if k.startswith("model__")}
    if clean:
        try:
            model.set_params(**clean)
        except ValueError:
            pass
    return model


def _fit_predict_fold(cfg, spec: ModelSpec, X, y, tr, te, params: dict | None, seed: int,
                      task: str = "regression",
                      prepped: tuple | None = None) -> tuple[np.ndarray, str]:
    """拟合一个折并返回 ``(用于算指标的分数, 概率能力类型)``。

    分类必须返回 **概率**而不是硬标签：观测值是用 OOF 概率算的 AUC，零分布若用
    硬标签算，两者不是同一个统计量，p 值会系统性偏小（更容易"显著"）。

    ``prepped=(X_tr_t, X_te_t)`` 时跳过前缀拟合，直接用折级缓存的变换结果
    （数值与整链 fit 完全一致，只是不再重复计算无监督部分）。
    """
    if prepped is not None:
        Xtr_t, Xte_t = prepped
        model = _model_from_params(cfg, spec, params, seed)
        model.fit(Xtr_t, y[tr])
        if task == "regression":
            return np.asarray(model.predict(Xte_t)), "none"
        if hasattr(model, "predict_proba"):
            p = np.asarray(model.predict_proba(Xte_t), dtype=np.float64)
            if p.ndim == 2 and p.shape[1] == 2:
                return p[:, 1], "native"
            return p, "native"
        if hasattr(model, "decision_function"):
            d = np.asarray(model.decision_function(Xte_t), dtype=np.float64)
            if d.ndim == 1:
                return 1.0 / (1.0 + np.exp(-d)), "decision"
            z = d - d.max(axis=1, keepdims=True)
            e = np.exp(z)
            return e / e.sum(axis=1, keepdims=True), "decision"
        return np.asarray(model.predict(Xte_t)), "none"

    pipe = build_pipeline(spec, cfg, None, seed=seed)
    clean = _clean_params(params)
    if clean:
        try:
            pipe.set_params(**clean)
        except ValueError:
            pass
    pipe.fit(X[tr], y[tr])
    if task == "regression":
        return np.asarray(pipe.predict(X[te])), "none"
    if hasattr(pipe, "predict_proba"):
        p = np.asarray(pipe.predict_proba(X[te]), dtype=np.float64)
        # 多分类必须保留 (n, k) 概率矩阵；取最后一列会把它当成二分类概率，
        # AUC/Brier 都会算错。二分类才取正类列。
        if p.ndim == 2 and p.shape[1] == 2:
            return p[:, 1], "native"
        return p, "native"
    if hasattr(pipe, "decision_function"):
        d = np.asarray(pipe.decision_function(X[te]), dtype=np.float64)
        if d.ndim == 1:
            return 1.0 / (1.0 + np.exp(-d)), "decision"
        z = d - d.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True), "decision"
    return np.asarray(pipe.predict(X[te])), "none"


def _effective_jobs(n_jobs: int | None) -> int:
    """把 ``-1`` 解析成实际核数；上限 16，避免线程/BLAS 过度订阅。"""
    cpu = os.cpu_count() or 1
    if n_jobs is None:
        return 1
    n = int(n_jobs)
    if n < 0:
        n = cpu
    return max(1, min(n, cpu, 16))


def y_randomization(
    ctx,
    spec: ModelSpec,
    obs_metric: float,
    fixed_params: dict | None = None,
    n_perm: int = 200,
    seed: int = 0,
    mode: str = "fast",
    metric_name: str | None = None,
    n_jobs: int | None = None,
) -> dict[str, Any]:
    """跑 Y-randomization，返回观测值、零分布与 p 值。

    ``mode="fast"``：固定外层折与最优超参，只重跑外层拟合（成本 = n_perm×R×k）。
    报告里必须注明"置换采用固定超参"，这是保守做法。
    ``mode="full"``：每轮完整重跑（本函数不实现，由调用方用 run_nested_cv 处理）。
    """
    cfg = ctx.cfg
    ds = ctx.ds
    task = ds.task
    metric = metric_name or PRIMARY.get(task, "r2")
    y = np.asarray(ds.targets[ctx.target]) if ctx.target in ds.targets else np.asarray(ds.y)
    groups = np.asarray(ds.groups, dtype=object)
    repeats = effective_repeats(cfg)
    rng = np.random.default_rng(seed)

    outer_by_repeat = []
    for r in range(repeats):
        outer_by_repeat.append(make_outer_splits(ds, cfg, int(cfg["runtime"]["seed"]) + r * 1000))

    # 折级预处理缓存：固定超参模式下，n_perm 次置换共用同一份变换后的 X
    fold_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    if fixed_params:
        budget = float(os.environ.get("SML_PERM_CACHE_MB", "1500") or 1500) * 1024 * 1024
        used = 0
        try:
            for r, outer in enumerate(outer_by_repeat):
                for k, (tr, te) in enumerate(outer):
                    prefix = _fit_prefix(cfg, spec, fixed_params, ds.X[tr],
                                         seed=(int(cfg["runtime"]["seed"]) + r * 1000) * 100 + k)
                    arrs = (prefix.transform(ds.X[tr]), prefix.transform(ds.X[te]))
                    used += int(arrs[0].nbytes) + int(arrs[1].nbytes)
                    if used > budget:
                        raise MemoryError(f"折级预处理缓存超过 {budget / 1024 ** 2:.0f} MB")
                    fold_cache[(r, k)] = arrs
            info(f"  置换检验复用折级预处理：{len(fold_cache)} 个折，"
                 f"省下 {len(fold_cache) * (n_perm if isinstance(n_perm, int) else 0)} 次重复预处理")
        except Exception as exc:  # noqa: BLE001 —— 缓存失败就退回逐个置换重建
            fold_cache.clear()
            warn(f"置换检验的折级预处理缓存不可用（{type(exc).__name__}: {exc}），退回逐次重建")

    def one_run(y_use: np.ndarray) -> float:
        preds, truths = [], []
        kinds: list[str] = []
        for r, outer in enumerate(outer_by_repeat):
            for k, (tr, te) in enumerate(outer):
                pred, kind = _fit_predict_fold(cfg, spec, ds.X, y_use, tr, te, fixed_params,
                                               seed=(int(cfg["runtime"]["seed"]) + r * 1000) * 100 + k,
                                               task=task, prepped=fold_cache.get((r, k)))
                preds.append(pred)
                truths.append(y_use[te])
                kinds.append(kind)
        yt = np.concatenate(truths)
        yp = np.concatenate(preds)
        from .metrics import classification_metrics, regression_metrics

        if task == "regression":
            return float(regression_metrics(yt, yp)[metric])
        proba_kind = "native" if all(k == "native" for k in kinds) else (
            "decision" if all(k in ("native", "decision") for k in kinds) else "none")
        score = np.clip(yp, 0, 1) if yp.ndim == 1 else yp
        return float(classification_metrics(yt, score, n_classes=len(ds.classes or [2]),
                                            proba_kind=proba_kind)[metric])

    # 置换彼此独立，完全可以并行；旧实现串行跑 n_perm×R×k 次拟合，
    # 在 200 次置换时成了整条链的瓶颈，而且没复用 --n-jobs。
    n_perm = int(n_perm)
    perms = [permute_y_sample_level(y, groups, rng) for _ in range(n_perm)]
    null_vals = np.empty(n_perm, dtype=float)
    jobs = _effective_jobs(n_jobs)

    def _one(y_perm: np.ndarray, idx: int) -> float:
        try:
            return float(one_run(y_perm))
        except Exception as exc:  # noqa: BLE001
            warn(f"置换 {idx + 1} 失败：{type(exc).__name__}: {exc}")
            return float("nan")

    if jobs <= 1 or n_perm <= 1:
        for i, y_perm in enumerate(perms):
            null_vals[i] = _one(y_perm, i)
            if (i + 1) % max(1, n_perm // 5) == 0:
                info(f"  Y-randomization {i + 1}/{n_perm}")
    else:
        from joblib import Parallel, delayed

        info(f"  Y-randomization 并行：n_jobs={jobs}，共 {n_perm} 次置换")
        try:
            # 必须用**有序**生成器：结果按提交顺序逐个到达，才能保证 null_vals[i]
            # 就是第 i 次置换的值。旧写法用 generator_unordered 却按到达序号写数组，
            # 零分布的逐置换数值是乱的（统计量不受影响，但 permutation.json 不可复现、
            # F6 散点顺序随机）。
            gen = Parallel(n_jobs=jobs, prefer="threads", return_as="generator")(
                delayed(_one)(y_perm, i) for i, y_perm in enumerate(perms))
            for done, val in enumerate(gen, start=1):
                null_vals[done - 1] = val
                if done % max(1, n_perm // 5) == 0:
                    info(f"  Y-randomization {done}/{n_perm}")
        except TypeError:
            # joblib < 1.3 不支持 return_as，退回一次性收集
            vals = Parallel(n_jobs=jobs, prefer="threads")(
                delayed(_one)(y_perm, i) for i, y_perm in enumerate(perms))
            null_vals[:] = np.asarray(vals, dtype=float)
            info(f"  Y-randomization 完成 {n_perm}/{n_perm}")

    valid = null_vals[np.isfinite(null_vals)]
    if valid.size < 5:
        return {"model": spec.name, "metric": metric, "observed": obs_metric,
                "n_valid": int(valid.size), "error": "有效置换次数过少，无法给出 p 值"}

    higher_better = metric not in ("rmse", "mae", "brier", "log_loss")
    if higher_better:
        n_extreme = int((valid >= obs_metric).sum())
    else:
        n_extreme = int((valid <= obs_metric).sum())
    p_value = (1 + n_extreme) / (1 + valid.size)

    return {
        "model": spec.name,
        "metric": metric,
        "observed": float(obs_metric),
        "null_mean": float(np.mean(valid)),
        "null_sd": float(np.std(valid, ddof=1)) if valid.size > 1 else float("nan"),
        "null_q025": float(np.percentile(valid, 2.5)),
        "null_q975": float(np.percentile(valid, 97.5)),
        "null_min": float(valid.min()),
        "null_max": float(valid.max()),
        "n_perm": int(valid.size),
        "n_extreme": n_extreme,
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
        "mode": mode,
        "fixed_hyperparams": bool(fixed_params),
        "null_values": [float(v) for v in valid],
        # 论文可直引的句子，指标名必须是展示名（R² / AUROC），不能漏出内部名 r2
        "conclusion": (
            f"观测 {METRIC_LABELS.get(metric, metric)}={obs_metric:.4f} 位于零分布之外"
            f"（p={p_value:.4f}），说明模型学到的是真实关系而非噪声"
            if p_value < 0.05 else
            f"观测 {METRIC_LABELS.get(metric, metric)}={obs_metric:.4f} 与零分布无显著差异"
            f"（p={p_value:.4f}），不能排除模型在拟合噪声"
        ),
    }


def summarize_permutation(result: dict[str, Any], null_values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "perm_index": np.arange(null_values.size),
        "null_metric": null_values,
        "observed": result.get("observed", np.nan),
        "metric": result.get("metric"),
    })
