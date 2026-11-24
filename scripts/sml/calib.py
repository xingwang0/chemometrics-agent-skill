# Copyright 2026 The research-ml-modeling Authors
# SPDX-License-Identifier: Apache-2.0
"""校准分析：回归的系统偏差，分类的概率校准。

回归：审稿人常问"你的预测有没有系统性偏差"。回答需要
* pred~obs 的斜率/截距（斜率<1 = 预测范围被压缩，斜率>1 = 外推放大）
* Bland-Altman：平均偏差与 95% 一致性界限（LoA），以及是否存在比例偏差

分类：概率输出必须校准才能用于决策。"AUC 高"不代表"概率可信"——
需要用校准曲线 + Brier + Brier Skill Score 回答，而不是只看 AUC。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "regression_calibration", "bland_altman", "calibration_curve",
    "brier_skill_score", "expected_calibration_error", "wilson_ci",
]

EPS = 1e-12


def _ols_slope(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """p = a + b*obs（b 为斜率，a 为截距）。"""
    xm, ym = float(np.mean(x)), float(np.mean(y))
    sxx = float(np.sum((x - xm) ** 2))
    if sxx < EPS:
        return float("nan"), float("nan")
    b = float(np.sum((x - xm) * (y - ym)) / sxx)
    a = float(ym - b * xm)
    return b, a


def regression_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray | None = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """预测-实测回归参数 + Bland-Altman + 按组 bootstrap 的置信区间。"""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if y.size < 5:
        return {"error": "样本不足，无法做校准分析"}

    slope, intercept = _ols_slope(p, y)
    r = float(np.corrcoef(y, p)[0, 1]) if np.std(p) > EPS else float("nan")
    ba = bland_altman(y, p)

    slope_ci = intercept_ci = (float("nan"), float("nan"))
    if groups is not None:
        g = np.asarray(groups, dtype=object)[mask]
        uniq = pd.unique(g)
        if uniq.size >= 5:
            member = {gg: np.flatnonzero(g == gg) for gg in uniq}
            rng = np.random.default_rng(seed)
            slopes, icpts = [], []
            for _ in range(int(n_boot)):
                picked = rng.integers(0, uniq.size, size=uniq.size)
                idx = np.concatenate([member[uniq[i]] for i in picked])
                b, a = _ols_slope(p[idx], y[idx])
                if np.isfinite(b):
                    slopes.append(b)
                    icpts.append(a)
            if len(slopes) > 10:
                lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
                slope_ci = (float(np.percentile(slopes, lo)), float(np.percentile(slopes, hi)))
                intercept_ci = (float(np.percentile(icpts, lo)), float(np.percentile(icpts, hi)))

    interpretation = _slope_interpretation(slope, slope_ci)
    return {
        "slope": slope, "intercept": intercept, "pearson_r": r,
        "slope_ci": slope_ci, "intercept_ci": intercept_ci,
        "bland_altman": ba,
        "interpretation": interpretation,
        "n": int(y.size),
    }


def _slope_interpretation(slope: float, ci: tuple[float, float]) -> str:
    if not np.isfinite(slope):
        return "斜率无法估计（预测值几乎为常数）"
    lo, hi = ci
    covers_one = np.isfinite(lo) and np.isfinite(hi) and lo <= 1.0 <= hi
    gap = abs(slope - 1.0)
    if slope < 0.9:
        head = f"斜率 {slope:.3f} < 1：预测范围被系统性压缩（高值低估、低值高估）"
    elif slope > 1.1:
        head = f"斜率 {slope:.3f} > 1：预测被放大，存在外推倾向"
    else:
        head = f"斜率 {slope:.3f} 接近 1：无明显范围压缩"
    if np.isfinite(lo):
        if covers_one:
            head += f"；95% CI [{lo:.3f}, {hi:.3f}] 包含 1，无显著比例偏差"
        elif gap < 0.1:
            head += (f"；95% CI [{lo:.3f}, {hi:.3f}] 虽不包含 1，但偏离幅度 {gap:.1%} 很小，"
                     f"实际影响有限（n 较大时统计显著不等于实用上重要）")
        else:
            head += f"；95% CI [{lo:.3f}, {hi:.3f}] 不包含 1，比例偏差有实际意义，需讨论原因"
    return head


def bland_altman(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Bland-Altman：平均偏差、LoA、以及偏差是否随均值变化（比例偏差）。"""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    diff = p - y
    mean_val = (p + y) / 2
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if diff.size > 1 else float("nan")
    loa = (bias - 1.96 * sd, bias + 1.96 * sd) if np.isfinite(sd) else (float("nan"), float("nan"))
    prop_p = float("nan")
    if diff.size > 4 and np.std(mean_val) > EPS:
        rho = stats.pearsonr(mean_val, diff)
        prop_p = float(rho.pvalue)
    return {
        "bias": bias, "sd": sd,
        "loa_low": float(loa[0]), "loa_high": float(loa[1]),
        "n_outside_loa": int(np.sum((diff < loa[0]) | (diff > loa[1]))) if np.isfinite(loa[0]) else 0,
        "proportional_bias_p": prop_p,
        "proportional_bias": bool(np.isfinite(prop_p) and prop_p < 0.05),
        "note": ("偏差随测量均值变化（比例偏差显著），建议先做变换（如对数）再建模"
                 if np.isfinite(prop_p) and prop_p < 0.05 else "未见明显比例偏差"),
    }


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """二项比例的 Wilson 区间（小样本比正态近似可靠，无需 statsmodels）。"""
    if n <= 0:
        return float("nan"), float("nan")
    z = float(stats.norm.ppf(1 - alpha / 2))
    phat = k / n
    denom = 1 + z ** 2 / n
    center = (phat + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2)) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def calibration_curve(
    y_true_bin: np.ndarray,
    proba: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """等频分箱的校准曲线（每箱给观测阳性率与 Wilson 区间）。"""
    y = np.asarray(y_true_bin, dtype=float).ravel()
    p = np.asarray(proba, dtype=float).ravel()
    mask = np.isfinite(p) & np.isfinite(y)
    y, p = y[mask], p[mask]
    if y.size < n_bins:
        n_bins = max(2, y.size // 2)
    order = np.argsort(p)
    y_s, p_s = y[order], p[order]
    rows = []
    for i, (cy, cp) in enumerate(zip(np.array_split(y_s, n_bins), np.array_split(p_s, n_bins))):
        if cy.size == 0:
            continue
        k = int(cy.sum())
        lo, hi = wilson_ci(k, cy.size)
        rows.append({
            "bin": i + 1, "n": int(cy.size),
            "pred_mean": float(cp.mean()), "obs_rate": float(cy.mean()),
            "ci_low": lo, "ci_high": hi,
            "abs_gap": float(abs(cp.mean() - cy.mean())),
        })
    return pd.DataFrame(rows)


def expected_calibration_error(y_true_bin: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    df = calibration_curve(y_true_bin, proba, n_bins=n_bins)
    if df.empty:
        return float("nan")
    w = df["n"] / df["n"].sum()
    return float((w * df["abs_gap"]).sum())


def brier_skill_score(y_true_bin: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """Brier Skill Score = 1 - Brier_model / Brier_climatology。"""
    y = np.asarray(y_true_bin, dtype=float)
    p = np.asarray(proba, dtype=float)
    base = float(np.mean(y))
    brier_model = float(np.mean((p - y) ** 2))
    brier_base = float(np.mean((base - y) ** 2))
    return {
        "brier": brier_model,
        "brier_baseline": brier_base,
        "brier_skill": float(1 - brier_model / brier_base) if brier_base > EPS else float("nan"),
    }
