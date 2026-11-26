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
import time
from typing import Any

import numpy as np
import pandas as pd

from .cv import effective_repeats, make_outer_splits
from .metrics import METRIC_LABELS
from .models import ModelSpec, build_pipeline
from .present import fmt_p
from .utils import human_duration, info, warn

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
    """把 ``runtime.n_jobs`` 解析成 worker 数（见 ``utils.resolve_workers``）。

    旧实现把 ``-1`` 解析成核数并硬性封顶 16。实测一台 14 核异构大小核机器上把 worker
    开满反而不如八成：E 核单核吞吐只有 P 核的六成，长任务派到慢核上拖尾。
    """
    from .utils import resolve_workers

    return resolve_workers(n_jobs)


# 多进程并行时 worker 只初始化一次的状态（X / 配置 / 模型规格）。
# 放在模块级 + initializer 里传：每个 worker 只 pickle 一次，而不是每个任务一次。
_PERM_STATE: dict[str, Any] = {}


def _perm_worker_init(X, cfg, spec, params, task) -> None:
    _PERM_STATE.clear()
    _PERM_STATE.update(X=X, cfg=cfg, spec=spec, params=params, task=task)


def _perm_fold_task(y_use: np.ndarray, tr: np.ndarray, te: np.ndarray, seed: int,
                    X_tr_prepped, X_te_prepped) -> tuple[np.ndarray | None, str, str | None]:
    """一次置换的一折：拟合 + 预测，返回 ``(分数, 概率能力, 错误信息)``。

    与主进程串行路径调用**同一个** ``_fit_predict_fold``，参数逐位相同，
    因此并行只改变"谁来算"，不改变算出来的数。
    """
    st = _PERM_STATE
    prepped = None if X_tr_prepped is None else (X_tr_prepped, X_te_prepped)
    try:
        pred, kind = _fit_predict_fold(
            st["cfg"], st["spec"], st["X"], y_use, tr, te, st["params"], seed,
            task=st["task"], prepped=prepped,
        )
        return pred, kind, None
    except Exception as exc:  # noqa: BLE001 —— 与串行路径一致：整轮记 nan
        return None, None, f"{type(exc).__name__}: {exc}"


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
    ``mode="full"``：每轮完整重跑（**本版本未实现**）。传入 full 会直接抛
    ``NotImplementedError``；绝不能"按 full 标注、按 fast 执行"，那会让报告写出
    一句没执行过的协议。
    """
    if str(mode).lower() != "fast":
        raise NotImplementedError(
            f"置换模式 {mode!r} 未实现：当前只有 fixed-hyperparameter（fast）路径。"
            "真要做 full（每轮重搜超参），请先实现每轮 run_nested_cv，再改这里与 "
            "config.validate_config 的 E-CFG-045 校验。"
        )
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

    # 折清单固定一次：串行与并行路径都按这个顺序拼接预测，池化指标才与旧实现逐位一致
    fold_list: list[tuple[int, int, np.ndarray, np.ndarray]] = [
        (r, k, tr, te)
        for r, outer in enumerate(outer_by_repeat)
        for k, (tr, te) in enumerate(outer)
    ]
    base_seed = int(cfg["runtime"]["seed"])

    def _fold_seed(r: int, k: int) -> int:
        return (base_seed + r * 1000) * 100 + k

    def _metric_of(y_use: np.ndarray, preds: list[np.ndarray], kinds: list[str]) -> float:
        """把逐折预测按 fold_list 顺序拼起来算池化指标（串行/并行共用同一段代码）。"""
        from .metrics import classification_metrics, regression_metrics

        truths = [y_use[te] for _r, _k, _tr, te in fold_list]
        yt = np.concatenate(truths)
        yp = np.concatenate(preds)
        if task == "regression":
            return float(regression_metrics(yt, yp)[metric])
        proba_kind = "native" if all(k == "native" for k in kinds) else (
            "decision" if all(k in ("native", "decision") for k in kinds) else "none")
        score = np.clip(yp, 0, 1) if yp.ndim == 1 else yp
        return float(classification_metrics(yt, score, n_classes=len(ds.classes or [2]),
                                            proba_kind=proba_kind)[metric])

    def one_run(y_use: np.ndarray) -> float:
        preds, kinds = [], []
        for r, k, tr, te in fold_list:
            pred, kind = _fit_predict_fold(cfg, spec, ds.X, y_use, tr, te, fixed_params,
                                           seed=_fold_seed(r, k), task=task,
                                           prepped=fold_cache.get((r, k)))
            preds.append(pred)
            kinds.append(kind)
        return _metric_of(y_use, preds, kinds)

    # 置换彼此独立，完全可以并行。
    # 后端选择有实测依据：这里用 **进程** 而不是线程。xgb 在 4 线程下只有
    # 0.76–0.83x（GIL 争用，比串行还慢），rf 也只有 1.23x；换成 loky 进程后
    # 4 worker 就有 3.5x、8 worker 5.25x。
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

    if n_perm <= 0:
        null_vals = np.empty(0, dtype=float)
    elif jobs <= 1 or n_perm <= 1:
        for i, y_perm in enumerate(perms):
            null_vals[i] = _one(y_perm, i)
            if (i + 1) % max(1, n_perm // 5) == 0:
                info(f"  Y-randomization {i + 1}/{n_perm}")
    else:
        # 第 0 次置换先串行跑：既当热身，又用来实测"单折拟合要多久"。
        # 没有这一步，smoke 那种总共 0.2s 的置换会被 10 个 worker 的启动成本
        # 吃掉（实测 0.2s → 22s）。
        from .utils import parallel_pays_off

        t0 = time.perf_counter()
        null_vals[0] = _one(perms[0], 0)
        first_seconds = time.perf_counter() - t0
        per_task = first_seconds / max(1, len(fold_list))
        rest_tasks = (n_perm - 1) * len(fold_list)
        if not parallel_pays_off(rest_tasks, per_task, jobs):
            info(f"  Y-randomization 串行：预估剩余 {human_duration(per_task * rest_tasks)}，"
                 f"并行启动成本（≈{jobs} worker）摊不平")
            for i in range(1, n_perm):
                null_vals[i] = _one(perms[i], i)
                if (i + 1) % max(1, n_perm // 5) == 0:
                    info(f"  Y-randomization {i + 1}/{n_perm}")
        else:
            from joblib import Parallel, delayed

            n_tasks = rest_tasks
            info(f"  Y-randomization 并行：n_jobs={jobs}（进程），{n_perm - 1} 次置换 × "
                 f"{len(fold_list)} 折 = {n_tasks} 个拟合任务"
                 f"（单折实测 {per_task * 1000:.0f} ms）")

            def _tasks():
                """按 (置换, 折) 派任务：粒度比"一次置换"细，尾部空转更少。

                置换本身仍在主进程按原 rng 顺序先生成，worker 只拿到"某一折的输入"，
                所以并行不改变任何一次置换的内容。
                """
                for i in range(1, n_perm):
                    for r, k, tr, te in fold_list:
                        pr = fold_cache.get((r, k))
                        yield delayed(_perm_fold_task)(
                            perms[i], tr, te, _fold_seed(r, k),
                            None if pr is None else pr[0],
                            None if pr is None else pr[1],
                        )

            out = Parallel(n_jobs=jobs, initializer=_perm_worker_init,
                           initargs=(ds.X, cfg, spec, fixed_params, task))(_tasks())

            order = [(i, r, k) for i in range(1, n_perm) for r, k, _tr, _te in fold_list]
            buckets: dict[int, dict[tuple[int, int], tuple]] = {}
            errors: list[str] = []
            for (i, r, k), (pred, kind, err) in zip(order, out):
                buckets.setdefault(i, {})[(r, k)] = (pred, kind, err)
                if err is not None:
                    errors.append(err)

            keys = [(r, k) for r, k, _tr, _te in fold_list]
            for i in range(1, n_perm):
                got = buckets.get(i, {})
                if any(got.get(kk, (None, None, "missing"))[2] is not None for kk in keys):
                    null_vals[i] = float("nan")
                    continue
                try:
                    null_vals[i] = _metric_of(perms[i], [got[kk][0] for kk in keys],
                                              [got[kk][1] for kk in keys])
                except Exception as exc:  # noqa: BLE001
                    warn(f"置换 {i + 1} 的指标计算失败：{type(exc).__name__}: {exc}")
                    null_vals[i] = float("nan")
                if (i + 1) % max(1, n_perm // 5) == 0:
                    info(f"  Y-randomization {i + 1}/{n_perm}")
            if errors:
                warn(f"置换检验有 {len(errors)} 个折拟合失败（首个：{errors[0]}），"
                     f"对应置换已记为无效")

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
        # n_perm 是**有效**置换数（失败的置换被 isfinite 剔除），p 的硬下限是
        # 1/(1+n_perm)。请求值与有效值必须分开记，否则少几次失败就会让 p 低于
        # 「1/(1+n_perm_requested)」这个读者以为的下限。
        "n_perm_requested": int(n_perm),
        "n_extreme": n_extreme,
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
        "mode": "fast",              # 实际执行路径；full 未实现，不能按请求值落盘
        "mode_requested": str(mode),
        "fixed_hyperparams": bool(fixed_params),
        "null_values": [float(v) for v in valid],
        # 论文可直引的句子，指标名必须是展示名（R² / AUROC），不能漏出内部名 r2
        "conclusion": (
            f"观测 {METRIC_LABELS.get(metric, metric)}={obs_metric:.4f} 位于零分布之外"
            f"（p={fmt_p(p_value)}），说明模型学到的是真实关系而非噪声"
            if p_value < 0.05 else
            f"观测 {METRIC_LABELS.get(metric, metric)}={obs_metric:.4f} 与零分布无显著差异"
            f"（p={fmt_p(p_value)}），不能排除模型在拟合噪声"
        ),
    }


def summarize_permutation(result: dict[str, Any], null_values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "perm_index": np.arange(null_values.size),
        "null_metric": null_values,
        "observed": result.get("observed", np.nan),
        "metric": result.get("metric"),
    })
