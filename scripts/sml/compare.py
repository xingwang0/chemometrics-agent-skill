# Copyright 2026 The research-ml-modeling Authors
# SPDX-License-Identifier: Apache-2.0
"""模型比较与多重比较校正。

原则
----
* **配对**：所有模型跑在同一批 OOF 样本上，比较必须按样本配对做，不能拿两组独立的
  均值加减 sd 去比（那会把"同一样品难易不同"这个方差算进差异里）。
* **按组重采样**的配对 bootstrap：同时抽样品、同时算两个模型的指标，差值 CI 是否跨 0。
* **多重比较**：一次比较 m 个模型对 × n 个指标，家族很大，必须给 BH-q 与 Bonferroni，
  只报原始 p 是审稿人最容易抓的统计错误之一。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import classification_metrics, oof_score_matrix, regression_metrics
from .utils import warn

__all__ = [
    "paired_bootstrap_diff", "paired_bootstrap_many", "compare_models", "bh_correct",
    "bonferroni", "wilcoxon_repeats", "PRIMARY_METRIC",
]

#: 「主指标」必须与选模同一把尺子（``selection.PRIMARY_BY_TASK``）：回归 R²、分类 AUC。
#: 旧值是回归 ``rmse``，于是 ``comparison.csv`` 的 ``metric_primary=True`` 标在 RMSE 行上，
#: 而选模与显著性检验用的是 R²——按下标筛「主指标」的人会拿 RMSE 判差异（真实数据上
#: 出现过 pls 与 rf 的 RMSE 与 R² 排序方向不同）。RMSE 仍会被强制纳入比对表，
#: 只是不再冒充主指标。
PRIMARY_METRIC = {"regression": "r2", "classification": "auc"}


def _side_frame(merged: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """把 merge(..., suffixes=('_a','_b')) 的一侧还原成原始列名。

    多分类的 ``p0``/``p1``/… 经 merge 后变成 ``p0_a``、``p0_b``，必须重新拼回
    ``p0..pN`` 才能交给 ``oof_score_matrix``；否则会静默退回硬标签 AUC。
    """
    cols = {c[: -len(suffix)]: merged[c] for c in merged.columns if c.endswith(suffix)}
    return pd.DataFrame(cols)


def _metric_value(task: str, y: np.ndarray, pred: np.ndarray,
                  proba: np.ndarray | None, metric: str, n_classes: int = 2,
                  proba_kind: str = "native") -> float:
    if task == "regression":
        return float(regression_metrics(y, pred).get(metric, np.nan))
    score = proba if proba is not None else pred
    return float(classification_metrics(y, score, n_classes=n_classes,
                                        proba_kind=proba_kind).get(metric, np.nan))


def paired_bootstrap_many(
    oof: pd.DataFrame,
    model_a: str,
    model_b: str,
    metrics: list[str],
    task: str = "regression",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    target: str | None = None,
    n_classes: int = 2,
) -> list[dict[str, Any]]:
    """同一批样品、同一组重采样序列上，一次算出多个指标的 A−B 置信区间。

    为什么可以合并：原实现对每个指标各调一次 ``paired_bootstrap_diff``，而它内部
    ``rng = default_rng(seed)`` 重新播种、循环次数与抽样规模完全相同——也就是说
    5 个指标的抽签序列本来就一模一样，只是把同一件事做了 5 遍（每遍还要重新 merge、
    重新建 group 成员表、每轮 ``np.concatenate`` 200 段索引）。合并后抽签序列不变，
    每个指标得到的 Δ 序列与原实现逐位相同。
    """
    metrics_list = list(dict.fromkeys(metrics))
    df = oof
    if target is not None and "target" in df.columns:
        df = df[df["target"] == target]
    a = df[df["model"] == model_a]
    b = df[df["model"] == model_b]
    if a.empty or b.empty:
        return [{"model_a": model_a, "model_b": model_b, "metric": m, "error": "模型缺少 OOF 记录"}
                for m in metrics_list]

    key = ["repeat", "row_id"] if "repeat" in df.columns else ["row_id"]
    merged = a.merge(b, on=key, suffixes=("_a", "_b"))
    if merged.empty:
        return [{"model_a": model_a, "model_b": model_b, "metric": m, "error": "OOF 无对齐行"}
                for m in metrics_list]

    y = merged["y_true_a"].to_numpy(dtype=float)
    pa = merged["y_pred_a"].to_numpy(dtype=float)
    pb = merged["y_pred_b"].to_numpy(dtype=float)
    if task == "regression":
        qa = qb = None
        kind_a = kind_b = "none"
        obs_a = regression_metrics(y, pa)
        obs_b = regression_metrics(y, pb)
    else:
        qa, kind_a = oof_score_matrix(_side_frame(merged, "_a"), n_classes)
        qb, kind_b = oof_score_matrix(_side_frame(merged, "_b"), n_classes)
        obs_a = classification_metrics(y, qa if qa is not None else pa, n_classes=n_classes,
                                      proba_kind=kind_a)
        obs_b = classification_metrics(y, qb if qb is not None else pb, n_classes=n_classes,
                                      proba_kind=kind_b)
    g = merged["group_a"].to_numpy(dtype=object) if "group_a" in merged else None
    if g is None:
        return [{"model_a": model_a, "model_b": model_b, "metric": m, "error": "缺少 group 列"}
                for m in metrics_list]

    uniq = pd.unique(g)
    if uniq.size < 3:
        return [{"model_a": model_a, "model_b": model_b, "metric": m,
                 "delta": float(obs_a.get(m, np.nan) - obs_b.get(m, np.nan)),
                 "error": "独立样品过少，无法做 bootstrap"} for m in metrics_list]

    member = {gg: np.flatnonzero(g == gg) for gg in uniq}
    rng = np.random.default_rng(seed)
    deltas = {m: np.empty(int(n_boot), dtype=float) for m in metrics_list}
    for i in range(int(n_boot)):
        picked = rng.integers(0, uniq.size, size=uniq.size)
        idx = np.concatenate([member[uniq[p]] for p in picked])
        if task == "regression":
            va = regression_metrics(y[idx], pa[idx])
            vb = regression_metrics(y[idx], pb[idx])
        else:
            va = classification_metrics(y[idx], qa[idx] if qa is not None else pa[idx],
                                        n_classes=n_classes, proba_kind=kind_a)
            vb = classification_metrics(y[idx], qb[idx] if qb is not None else pb[idx],
                                        n_classes=n_classes, proba_kind=kind_b)
        for m in metrics_list:
            deltas[m][i] = float(va.get(m, np.nan)) - float(vb.get(m, np.nan))

    out: list[dict[str, Any]] = []
    for metric in metrics_list:
        obs_a_m = float(obs_a.get(metric, np.nan))
        obs_b_m = float(obs_b.get(metric, np.nan))
        delta = obs_a_m - obs_b_m
        d = deltas[metric]
        d = d[np.isfinite(d)]
        if d.size < 10:
            out.append({"model_a": model_a, "model_b": model_b, "metric": metric,
                        "delta": float(delta), "error": "bootstrap 有效样本不足"})
            continue
        lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        p_two = 2 * min(float((d <= 0).mean()), float((d >= 0).mean()))
        p_two = float(min(1.0, max(p_two, 1.0 / (d.size + 1))))
        lower_better = metric in {"rmse", "mae", "brier", "log_loss", "ece"}
        if lower_better:
            better = model_a if delta < 0 else model_b
        else:
            better = model_a if delta > 0 else model_b
        significant = bool((lo > 0) or (hi < 0))
        out.append({
            "model_a": model_a, "model_b": model_b, "metric": metric,
            "value_a": float(obs_a_m), "value_b": float(obs_b_m), "delta": float(delta),
            "ci_low": float(lo), "ci_high": float(hi),
            "p_raw": p_two, "significant": significant,
            "better": better if significant else "无显著差异",
            "n_pairs": int(len(y)),
        })
    return out


def paired_bootstrap_diff(
    oof: pd.DataFrame,
    model_a: str,
    model_b: str,
    metric: str = "rmse",
    task: str = "regression",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    target: str | None = None,
    n_classes: int = 2,
) -> dict[str, Any]:
    """同一批样品上 A−B 的指标差值的 bootstrap 置信区间与双侧 p 值。

    差值方向：metric 为"越小越好"时（rmse/mae/brier），Δ<0 表示 A 更好；
    为"越大越好"时（r2/auc），Δ>0 表示 A 更好。结果里用 ``better`` 字段直接给结论。
    """
    return paired_bootstrap_many(oof, model_a, model_b, [metric], task=task, n_boot=n_boot,
                                 alpha=alpha, seed=seed, target=target, n_classes=n_classes)[0]


def wilcoxon_repeats(
    fold_df: pd.DataFrame,
    model_a: str,
    model_b: str,
    metric: str = "rmse",
    target: str | None = None,
) -> dict[str, Any]:
    """按 repeat 配对的 Wilcoxon 符号秩检验（n=repeat 数，样本量小时只作参考）。"""
    df = fold_df
    if target is not None and "target" in df.columns:
        df = df[df["target"] == target]
    if metric not in df.columns:
        return {"error": f"fold 表没有指标 {metric}"}
    per_repeat = (df[df["model"].isin([model_a, model_b])]
                  .groupby(["repeat", "model"])[metric].mean().unstack())
    if model_a not in per_repeat or model_b not in per_repeat:
        return {"error": "模型缺少逐折记录"}
    pair = per_repeat[[model_a, model_b]].dropna()
    if len(pair) < 4:
        return {"model_a": model_a, "model_b": model_b, "metric": metric,
                "note": f"repeat 数仅 {len(pair)}，Wilcoxon 不作推断"}
    try:
        res = stats.wilcoxon(pair[model_a], pair[model_b])
        return {"model_a": model_a, "model_b": model_b, "metric": metric,
                "statistic": float(res.statistic), "p_raw": float(res.pvalue),
                "n_pairs": int(len(pair))}
    except ValueError as exc:
        return {"model_a": model_a, "model_b": model_b, "metric": metric, "error": str(exc)}


def bh_correct(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR 校正（scipy 1.18 内置）。"""
    arr = np.asarray(pvals, dtype=float)
    if arr.size == 0:
        return []
    if not np.isfinite(arr).all():
        warn("BH 校正的 p 值含非有限值，已用 1.0 替代")
        arr = np.where(np.isfinite(arr), arr, 1.0)
    try:
        return [float(v) for v in stats.false_discovery_control(arr, method="bh")]
    except (AttributeError, ValueError):
        # 手写 BH 兜底
        order = np.argsort(arr)
        m = arr.size
        q = np.empty(m)
        prev = 1.0
        for rank, idx in enumerate(order[::-1], start=1):
            val = arr[idx] * m / (m - rank + 1)
            prev = min(prev, val)
            q[idx] = prev
        return [float(v) for v in q]


def bonferroni(pvals: list[float]) -> list[float]:
    m = max(len(pvals), 1)
    return [float(min(1.0, p * m)) for p in pvals]


def compare_models(
    oof: pd.DataFrame,
    fold_df: pd.DataFrame,
    task: str,
    metrics: list[str],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    n_classes: int = 2,
) -> pd.DataFrame:
    """对所有模型对 × 指标做配对 bootstrap，并对整族做 BH 与 Bonferroni 校正。"""
    models = sorted(oof["model"].unique()) if not oof.empty else []
    if len(models) < 2:
        return pd.DataFrame()
    targets = sorted(oof["target"].unique()) if "target" in oof.columns else [None]
    primary = PRIMARY_METRIC.get(task, "rmse")
    use_metrics = [m for m in (dict.fromkeys([primary] + list(metrics))) if m]

    rows: list[dict[str, Any]] = []
    pairs = [(models[i], models[j]) for i in range(len(models)) for j in range(i + 1, len(models))]
    for tname in targets:
        # 每个 pair 只做一次 bootstrap，一次算齐所有指标（抽签序列与逐指标调用完全相同）
        by_metric: dict[tuple[str, str, str], dict[str, Any]] = {}
        for a, b in pairs:
            for res in paired_bootstrap_many(oof, a, b, use_metrics, task=task, n_boot=n_boot,
                                             alpha=alpha, seed=seed, target=tname,
                                             n_classes=n_classes):
                by_metric[(tname, res["metric"], (a, b))] = res
        for metric in use_metrics:
            for a, b in pairs:
                res = by_metric.get((tname, metric, (a, b)), {})
                res = dict(res)
                res["target"] = tname
                res["metric_primary"] = metric == primary
                rows.append(res)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if "p_raw" in df:
        p = df["p_raw"].to_numpy(dtype=float)
        df["p_BH"] = bh_correct(list(p))
        df["p_bonf"] = bonferroni(list(p))
        df["m_family"] = len(p)
    # 顺带并入 repeat 级 Wilcoxon 结果
    try:
        w = [wilcoxon_repeats(fold_df, r["model_a"], r["model_b"], primary, r.get("target"))
             for _, r in df.iterrows()]
        df["wilcoxon_p"] = [x.get("p_raw", np.nan) for x in w]
    except Exception:
        df["wilcoxon_p"] = np.nan
    return df.sort_values(["target", "metric", "p_raw"], na_position="last").reset_index(drop=True)


def rank_models(summary: pd.DataFrame, task: str) -> pd.DataFrame:
    """按**选模用的同一个指标**排名（目标内），附与同目标最优模型的差值。

    多目标时 ``metrics_summary`` 是多个 target 纵向拼接的表，若直接
    ``df[col].rank()`` 就是拿不同目标的误差绝对值互相比：量纲不同（含量 vs 水分）、
    难度不同，排名与 ``gap_to_best`` 都会指向"跨目标最强的那个"。
    每个 target 有自己的尺度，只能在 target 内部比。

    指标必须与 ``_best_by_primary`` / ``pick_best_model`` 一致：回归用 R²，
    分类用 AUC。早先这里用 RMSE 排名、选模却用 R²，两者在"逐折 R² 差别大但 RMSE
    接近"的数据上会给出不同的第一名（真实数据上出现过 KNN 的折均值 R² 为负、却因
    RMSE 最小被 rank 成第 1，与报告写的「最优模型 XGB」直接打架）。
    """
    if summary.empty:
        return summary
    base_metric = "r2" if task == "regression" else "auc"
    col = f"{base_metric}_mean"
    if col not in summary.columns:
        # 兜底：连选模指标都没有时，退回一个**确实存在**的误差指标（并如实按它排名）。
        # 不要再退回 PRIMARY_METRIC——它就是刚被判定缺失的那个。
        for _cand in (("mae", "rmse", "bias") if task == "regression"
                      else ("balanced_accuracy", "f1_macro")):
            if f"{_cand}_mean" in summary.columns:
                base_metric = _cand
                break
        col = f"{base_metric}_mean"
        if col not in summary.columns:
            return summary
    lower_better = base_metric in {"rmse", "mae", "brier", "log_loss", "ece"}
    df = summary.copy()

    # 口径：stage_cv 与 rebuild_tables 都会在「有效折 < MIN_OK_FOLDS」时把这些行标成
    # metric_basis=pooled（判定统一在 sml.metrics.mark_metric_basis，不要在这里另写一套）。
    # 此时 *_mean 只是极少数折的平均（退化折被 nanmean 跳过），不能用于排名，
    # 改用无后缀的池化列（由 oof_group_metrics 计算）。缺列时退回 fold 口径。
    basis = df["metric_basis"].astype(str) if "metric_basis" in df.columns else pd.Series("fold", index=df.index)
    use_pooled = (basis.eq("pooled") & df[base_metric].notna()) if base_metric in df.columns else pd.Series(False, index=df.index)
    score = df[col].astype(float).copy()
    if base_metric in df.columns:
        score = score.where(~use_pooled, df[base_metric].astype(float))
    df["_rank_score"] = score

    multi = "target" in df.columns and df["target"].nunique(dropna=False) > 1
    if multi:
        grp = df.groupby("target", dropna=False)["_rank_score"]
        df["rank"] = grp.rank(ascending=lower_better, method="min")
        best = grp.transform("min" if lower_better else "max")
    else:
        df["rank"] = df["_rank_score"].rank(ascending=lower_better, method="min")
        best = df["_rank_score"].min() if lower_better else df["_rank_score"].max()
    df[f"{base_metric}_gap_to_best"] = (df["_rank_score"] - best) * (1 if lower_better else -1)
    df = df.drop(columns="_rank_score")
    sort_keys = ["target", "rank"] if multi else ["rank"]
    return df.sort_values(sort_keys, kind="stable")
