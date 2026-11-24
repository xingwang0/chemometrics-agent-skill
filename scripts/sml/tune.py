"""内层调参：把搜索空间展开成候选参数字典，用 GridSearchCV 在显式 split 列表上搜。

为什么不用 RandomizedSearchCV
-----------------------------
1. 内层分组 CV 没有 groups 通道（除非启用 metadata routing），必须传显式 split 列表，
   而 GridSearchCV/RandomizedSearchCV 都接受列表——但我们需要自己控制"采样出来的
   每个候选字典"，才能在日志里精确报出搜索了多少个组合、每个组合的最优分数。
2. 我们要把 ``prep__which``（预处理候选）作为搜索的一维，自己展开最直接。

数值型分布（loguniform/int）在这里被采样成有限个点，语义等价于随机搜索。
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
from sklearn.model_selection import GridSearchCV

from .utils import info, warn

__all__ = ["expand_space", "inner_search", "default_scoring", "scoring_for", "count_candidates"]

MAX_GRID_COMBOS = 240
MAX_LOGUNIFORM_POINTS = 40


def _sample_one(spec: Any, rng: np.random.Generator) -> Any:
    if isinstance(spec, dict) and "dist" in spec:
        dist = str(spec["dist"]).lower()
        if dist == "int":
            low, high = int(spec["low"]), int(spec["high"])
            return int(rng.integers(low, high + 1))
        if dist in ("loguniform", "log_uniform"):
            low, high = float(spec["low"]), float(spec["high"])
            return float(np.exp(rng.uniform(np.log(low), np.log(high))))
        if dist == "uniform":
            return float(rng.uniform(float(spec["low"]), float(spec["high"])))
        if dist == "choice":
            vals = list(spec.get("values") or [])
            if not vals:
                raise ValueError("choice 分布缺少 values")
            return vals[int(rng.integers(len(vals)))]
        raise ValueError(f"未知分布类型 {dist!r}")
    if isinstance(spec, (list, tuple)):
        if len(spec) == 0:
            raise ValueError("候选取值列表为空")
        return spec[int(rng.integers(len(spec)))]
    return spec


def _grid_values(spec: Any, rng: np.random.Generator) -> list[Any]:
    if isinstance(spec, dict) and "dist" in spec:
        dist = str(spec["dist"]).lower()
        if dist == "int":
            low, high = int(spec["low"]), int(spec["high"])
            vals = sorted({low, high, (low + high) // 2})
            return vals
        if dist in ("loguniform", "log_uniform"):
            low, high = float(spec["low"]), float(spec["high"])
            pts = np.geomspace(low, high, num=min(5, MAX_LOGUNIFORM_POINTS))
            return [float(v) for v in pts]
        if dist == "uniform":
            low, high = float(spec["low"]), float(spec["high"])
            return [float(v) for v in np.linspace(low, high, num=3)]
        return [_sample_one(spec, rng)]
    if isinstance(spec, (list, tuple)):
        return list(spec)
    return [spec]


def count_candidates(space: dict[str, Any]) -> int:
    total = 1
    for key, val in space.items():
        if isinstance(val, (list, tuple)):
            total *= max(1, len(val))
        elif isinstance(val, dict) and "dist" in val:
            total *= 5
    return total


def expand_space(
    space: dict[str, Any],
    search_type: str = "random",
    n_iter: int = 30,
    seed: int = 0,
    max_grid: int = MAX_GRID_COMBOS,
    stratify_key: str | None = None,
) -> list[dict[str, Any]]:
    """把搜索空间展开成候选参数字典列表。

    返回的每个字典的值都是**单元素列表**，因为 ``GridSearchCV(param_grid=list_of_dicts)``
    要求每个字典里 ``参数 -> 待取值列表``，单值必须包一层。

    ``stratify_key``（可选）：只在 random 搜索下生效。给了它（通常就是
    ``prep__which``）就把 ``n_iter`` 次抽样**按该维度分层均分**，每层至少
    ⌊n_iter/层数⌋ 个候选、余数随机分配；不填时走原路径（rng 调用顺序与旧版
    完全一致，候选逐位相同）。动机：旧版把预处理当普通维度均匀抽，5 条预处理
    × n_iter=30 时每条平均只有 6 个点，二项波动可能压到 2–3 个，"预处理作为
    超参"在统计上被稀释。
    """
    if not space:
        return [{}]
    rng = np.random.default_rng(seed)
    keys = list(space.keys())
    st = str(search_type).lower()

    if st == "grid":
        value_lists = [_grid_values(space[k], rng) for k in keys]
        combos = [dict(zip(keys, vals)) for vals in itertools.product(*value_lists)]
        if len(combos) > max_grid:
            idx = rng.choice(len(combos), size=max_grid, replace=False)
            combos = [combos[i] for i in sorted(idx)]
            warn(f"网格组合数超过 {max_grid}，已随机抽取 {max_grid} 组")
        if stratify_key:
            warn(f"preprocess.stratify_sampling 只对 random 搜索生效；{stratify_key} 这条网格未分层")
        return _wrap(_dedupe(combos))

    if st == "none":
        return [{k: [v] for k, v in {kk: _sample_one(space[kk], rng) for kk in keys}.items()}]

    n_iter = max(1, int(n_iter))

    if stratify_key and stratify_key in space:
        levels = list(space[stratify_key])
        if len(levels) > 1:
            per, extra = divmod(n_iter, len(levels))
            order = rng.permutation(len(levels))
            alloc = [per + (1 if int(j) < extra else 0) for j in range(len(levels))]
            shuffled = [alloc[int(i)] for i in order]
            combos: list[dict[str, Any]] = []
            seen: set[tuple] = set()
            for lev, cnt in zip(levels, shuffled):
                guard = 0
                got = 0
                while got < cnt and guard < cnt * 20:
                    guard += 1
                    cand = {k: _sample_one(space[k], rng) for k in keys}
                    cand[stratify_key] = lev
                    key = tuple((k, round(v, 10) if isinstance(v, float) else v)
                                for k, v in sorted(cand.items()))
                    if key in seen:
                        continue
                    seen.add(key)
                    combos.append(cand)
                    got += 1
            if len(combos) < n_iter:
                warn(f"分层抽样只凑到 {len(combos)}/{n_iter} 个候选（参数空间重复度高）")
            return _wrap(combos)

    combos = []
    seen = set()
    guard = 0
    while len(combos) < n_iter and guard < n_iter * 20:
        guard += 1
        cand = {k: _sample_one(space[k], rng) for k in keys}
        key = tuple((k, round(v, 10) if isinstance(v, float) else v) for k, v in sorted(cand.items()))
        if key in seen:
            continue
        seen.add(key)
        combos.append(cand)
    return _wrap(combos)


def _wrap(combos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把 ``{"a": 1}`` 变成 ``{"a": [1]}``，满足 GridSearchCV 的参数网格格式。"""
    return [{k: (v if isinstance(v, (list, tuple, np.ndarray)) else [v]) for k, v in cand.items()}
            for cand in combos]


def _dedupe(combos: list[dict]) -> list[dict]:
    out, seen = [], set()
    for cand in combos:
        key = tuple((k, round(v, 10) if isinstance(v, float) else v) for k, v in sorted(cand.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
    return out


def default_scoring(task: str, has_proba: bool = True) -> str:
    """按任务的默认打分器。``has_proba=False`` 时分类退回 balanced_accuracy。

    注意：**不要**在模块级按 task 调一次就复用（``default_scoring(task)``）。
    打分器是否可用是**估计器**的能力，不是任务的属性——同一个分类任务里，
    logreg 有 predict_proba 而 ridge_clf 没有。用 ``scoring_for(pipe, task)``。
    """
    if task == "regression":
        return "neg_root_mean_squared_error"
    return "neg_log_loss" if has_proba else "balanced_accuracy"


def scoring_for(est, task: str) -> str:
    """按**这个估计器的实际能力**选打分器。

    用 ``hasattr(est, "predict_proba")`` 而不是维护 kind 白名单：
    sklearn 的 ``Pipeline`` 通过 ``available_if`` 暴露 ``predict_proba``，
    末步没有该方法时 hasattr 就是 False，因此自定义估计器也自动正确。
    """
    if task == "regression":
        return "neg_root_mean_squared_error"
    return default_scoring(task, has_proba=hasattr(est, "predict_proba"))


def inner_search(
    pipe,
    param_dicts: list[dict[str, Any]],
    splits: list[tuple[np.ndarray, np.ndarray]],
    X: np.ndarray,
    y: np.ndarray,
    scoring: str,
    n_jobs: int = 1,
    fail_fast: bool = False,
):
    """在给定 split 列表上做网格/随机搜索。

    返回 ``(best_estimator, best_params, best_score, search)``。
    ``splits`` 必须是**已经按训练折内部 group 切好**的显式列表，
    这样内层不会看到外层测试折的任何数据。
    """
    if not param_dicts:
        raise ValueError("参数候选为空")
    error_score = "raise" if fail_fast else np.nan
    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_dicts,
        cv=splits,
        scoring=scoring,
        refit=True,
        n_jobs=int(n_jobs) if n_jobs else 1,
        error_score=error_score,
        return_train_score=False,
    )
    # 并行时把每个 worker 的 BLAS/OpenMP 线程压到 1：外层的 n_jobs 个任务已经吃满
    # CPU，worker 内部再开 BLAS 线程只会互相抢核（超订），实测反而更慢。
    if int(n_jobs or 1) != 1:
        try:
            from joblib import parallel_backend

            with parallel_backend("loky", inner_max_num_threads=1):
                search.fit(X, y)
        except TypeError:      # joblib 版本不支持 inner_max_num_threads
            search.fit(X, y)
    else:
        search.fit(X, y)
    return search.best_estimator_, dict(search.best_params_), float(search.best_score_), search
