# Copyright 2026 The research-ml-modeling Authors
# SPDX-License-Identifier: Apache-2.0
"""指标计算与不确定度。

三个必须守住的方法学细节
------------------------
1. **按组重采样**（cluster bootstrap）：抽 ``sample_id`` 而不是抽行。同一样品的多次测量
   高度相关，按行重采样会把组内相关当独立，置信区间系统性偏窄、p 值偏小。
2. **充分统计量向量化**：回归指标只依赖每组的 (n, Σy, Σy², Σŷ, Σŷ², Σyŷ, Σ(y-ŷ)², Σ|y-ŷ|)，
   重采样时按组权重累加即可，比每轮重算预测数组快 1–2 个数量级。
3. **跨 repeat 聚合口径**：先算每个 repeat 的均值，再对 R 个均值求 mean±sd。
   把 R×k 个折混在一起算 sd 是伪重复，会把 sd 低估 2–3 倍。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, log_loss, matthews_corrcoef, r2_score,
    roc_auc_score,
)

__all__ = [
    "regression_metrics", "classification_metrics", "bootstrap_regression",
    "bootstrap_classification", "aggregate_repeats", "metric_direction",
    "METRIC_LABELS", "format_metric", "oof_score_matrix",
    "mark_metric_basis", "describe_pooled_rows", "MIN_OK_FOLDS",
    "prediction_range_check", "PRED_RANGE_FACTOR",
]

# 预测值越界的判定倍数：超出训练集目标极差的这么多倍即视为"量纲爆炸"。
# 定 2.0 而不是更宽的 5.0：正常模型的越界量级远小于 1 倍（实测健康数据最大 0.08），
# 而真正的外推爆炸是几倍到几十倍。曾经用 5.0，结果刚好漏掉"真实 153 mV 预测成 2431 mV"
# 这个案例（比值 4.64）——阈值比它要抓的典型故障还宽，等于没装。
PRED_RANGE_FACTOR = 2.0

METRIC_LABELS = {
    "r2": "R²", "rmse": "RMSE", "mae": "MAE", "bias": "Bias(ME)", "rpd": "RPD",
    "rpiq": "RPIQ", "ccc": "CCC", "slope": "斜率", "intercept": "截距",
    "auc": "AUROC", "auprc": "AUPRC", "balanced_accuracy": "平衡准确率",
    "f1_macro": "F1(macro)", "mcc": "MCC", "brier": "Brier", "log_loss": "LogLoss",
    "sensitivity": "灵敏度", "specificity": "特异度", "accuracy": "准确率",
    "ece": "ECE",
}

# 指标方向：True 表示越大越好
_DOWN_METRICS = {"rmse", "mae", "brier", "log_loss", "ece", "bias_abs"}
_NEUTRAL_METRICS = {"bias", "slope", "intercept"}

EPS = 1e-12


def _normalize_rows(proba: np.ndarray) -> np.ndarray:
    """把概率矩阵逐行归一到 1；行和为 0 的坏行退回均匀分布。

    为什么必须做：概率来自 float32 特征/估计器时，行和会偏离 1 约 1e-7
    （LogisticRegression 在多分类下就是这样）。逐元素 clip 到 [1e-12, 1-1e-12]
    只能防止 log(0)，并不修复归一——sklearn 的 ``log_loss`` 于是每折都丢一条
    「y_prob values do not sum to one」警告（一次 3 分类运行能刷上千条），
    而 Brier/ECE 的分母语义本来就假设每行是概率分布，不归一等于在算错的东西。
    """
    p = np.asarray(proba, dtype=np.float64)
    if p.ndim != 2:
        return p
    s = p.sum(axis=1, keepdims=True)
    ok = np.isfinite(s).ravel() & (s.ravel() > 0)
    out = np.full_like(p, 1.0 / p.shape[1])
    out[ok] = p[ok] / s[ok]
    return np.nan_to_num(out, nan=1.0 / p.shape[1])


def metric_direction(name: str) -> bool | None:
    if name in _NEUTRAL_METRICS:
        return None
    return name not in _DOWN_METRICS


def format_metric(name: str, value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


# --------------------------------------------------------------------------- #
# 回归
# --------------------------------------------------------------------------- #
def prediction_range_check(
    y_train: np.ndarray,
    y_pred: np.ndarray,
    factor: float = PRED_RANGE_FACTOR,
) -> dict[str, float]:
    """检查预测值是否跑出目标量纲（外推爆炸的报警器）。

    为什么必须有：线性模型（PLS/PCR/ridge）在"训练集很小 + 测试样品落在潜在空间之外"
    时会把预测推到荒谬的量级。实测一例：真实 153 / 176 mV 的样品被 PLS 预测成
    2431 / 2457 mV，单折 R² 掉到 −37.8；管线此前照单全收，报告里只是安静地显示
    −37.8，没有任何提示——读者会以为"这个模型很差"，实际是"这个模型在这几个样品上
    数值爆炸了"，两者的处理方式完全不同。

    判据用**相对量**（训练集 y 的极差）而不是绝对值：目标可能是 0–1 的分数，也可能是
    0–500 的过电位，绝对阈值没法通用。

    返回 ``ratio`` = 最大越界距离 / 训练集极差。ratio=0 表示全部落在训练集范围内；
    ratio>``factor`` 视为量纲爆炸（``n_extreme`` 计数）。
    """
    ytr = np.asarray(y_train, dtype=float)
    ypr = np.asarray(y_pred, dtype=float)
    ytr = ytr[np.isfinite(ytr)]
    ypr = ypr[np.isfinite(ypr)]
    out: dict[str, float] = {
        "ratio": float("nan"), "n_extreme": 0.0, "n_pred": float(ypr.size),
        "y_min": float("nan"), "y_max": float("nan"),
        "pred_min": float("nan"), "pred_max": float("nan"),
    }
    if ytr.size < 2 or ypr.size == 0:
        return out
    lo, hi = float(ytr.min()), float(ytr.max())
    span = hi - lo
    if not np.isfinite(span) or span <= 0:
        # 训练集目标是常数：用绝对值兜底，避免除零造出 inf
        span = max(abs(hi), 1.0)
    over = np.maximum(ypr - hi, lo - ypr)
    over = np.clip(over, 0.0, None)
    out.update({
        "ratio": float(over.max() / span),
        "n_extreme": float((over > factor * span).sum()),
        "y_min": lo, "y_max": hi,
        "pred_min": float(ypr.min()), "pred_max": float(ypr.max()),
    })
    return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.float64).ravel()
    p = np.asarray(y_pred, dtype=np.float64).ravel()
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    n = y.size
    if n < 2:
        return {k: float("nan") for k in
                ("r2", "rmse", "mae", "bias", "rpd", "rpiq", "ccc", "slope", "intercept", "n")}

    resid = y - p
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    rmse = float(np.sqrt(ss_res / n))
    mae = float(np.mean(np.abs(resid)))
    sd_y = float(np.std(y, ddof=1))
    iqr = float(np.subtract(*np.percentile(y, [75, 25])))
    rpd = float(sd_y / rmse) if rmse > EPS else float("inf")
    rpiq = float(iqr / rmse) if rmse > EPS else float("inf")

    # CCC（Lin 一致性相关系数）：CCC = 2*rho*sy*sp / (sy² + sp² + (mean_y - mean_p)²)
    sy, sp = float(np.std(y, ddof=0)), float(np.std(p, ddof=0))
    if sy > EPS and sp > EPS:
        rho = float(np.corrcoef(y, p)[0, 1])
        ccc = float(2 * rho * sy * sp / (sy ** 2 + sp ** 2 + (y.mean() - p.mean()) ** 2))
    else:
        ccc = float("nan")

    var_y = float(np.var(y, ddof=0))
    slope = float(np.cov(y, p, ddof=0)[0, 1] / var_y) if var_y > EPS else float("nan")
    intercept = float(p.mean() - slope * y.mean()) if np.isfinite(slope) else float("nan")

    return {
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > EPS else float("nan"),
        "rmse": rmse, "mae": mae, "bias": float(np.mean(resid)),
        "rpd": rpd, "rpiq": rpiq, "ccc": ccc,
        "slope": slope, "intercept": intercept,
        "n": float(n),
    }


@dataclass
class _GroupSuffStats:
    groups: np.ndarray
    n: np.ndarray
    sy: np.ndarray
    sy2: np.ndarray
    sp: np.ndarray
    sp2: np.ndarray
    syp: np.ndarray
    sres2: np.ndarray
    sabsres: np.ndarray

    def take(self, idx: np.ndarray) -> dict[str, np.ndarray]:
        return {k: getattr(self, k)[idx] for k in
                ("n", "sy", "sy2", "sp", "sp2", "syp", "sres2", "sabsres")}


def _suffstats(y: np.ndarray, p: np.ndarray, groups: np.ndarray) -> _GroupSuffStats:
    groups = np.asarray(groups, dtype=object)
    uniq = pd.unique(groups)
    pos = {g: i for i, g in enumerate(uniq)}
    idx = np.array([pos[g] for g in groups], dtype=np.int64)
    k = uniq.size

    def agg(vals: np.ndarray) -> np.ndarray:
        out = np.zeros(k, dtype=np.float64)
        np.add.at(out, idx, vals)
        return out

    resid = y - p
    return _GroupSuffStats(
        groups=uniq,
        n=agg(np.ones_like(y)),
        sy=agg(y), sy2=agg(y ** 2), sp=agg(p), sp2=agg(p ** 2), syp=agg(y * p),
        sres2=agg(resid ** 2), sabsres=agg(np.abs(resid)),
    )


def _metrics_from_sums(sums: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    n = sums["n"]
    ok = n > 0
    n = np.where(ok, n, 1.0)
    sy, sy2, sp, sp2, syp = sums["sy"], sums["sy2"], sums["sp"], sums["sp2"], sums["syp"]
    sres2, sabsres = sums["sres2"], sums["sabsres"]

    mean_y = sy / n
    mean_p = sp / n
    var_y = np.maximum(sy2 / n - mean_y ** 2, 0.0)
    var_p = np.maximum(sp2 / n - mean_p ** 2, 0.0)
    cov = syp / n - mean_y * mean_p
    rmse = np.sqrt(np.maximum(sres2 / n, 0.0))
    mae = sabsres / n
    # 样本标准差（ddof=1）用于 RPD，与文献口径一致
    sd_y = np.sqrt(np.maximum((sy2 - n * mean_y ** 2) / np.maximum(n - 1, 1), 0.0))

    ss_tot = np.maximum(sy2 - n * mean_y ** 2, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - sres2 / np.where(ss_tot > EPS, ss_tot, np.nan)
        rpd = sd_y / np.where(rmse > EPS, rmse, np.nan)
        slope = cov / np.where(var_y > EPS, var_y, np.nan)
        denom = var_y + var_p + (mean_y - mean_p) ** 2
        ccc = 2 * cov / np.where(denom > EPS, denom, np.nan)
    return {
        "r2": r2, "rmse": rmse, "mae": mae, "bias": mean_y - mean_p,
        "rpd": rpd, "rpiq": np.full_like(rmse, np.nan), "ccc": ccc, "slope": slope,
    }


def bootstrap_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, tuple[float, float]]:
    """按组（cluster）重采样的百分位 bootstrap 置信区间。"""
    y = np.asarray(y_true, dtype=np.float64).ravel()
    p = np.asarray(y_pred, dtype=np.float64).ravel()
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    g = np.asarray(groups, dtype=object)[mask]
    if y.size < 4:
        return {}
    ss = _suffstats(y, p, g)
    k = ss.groups.size
    if k < 3:
        return {}

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, k, size=(int(n_boot), k))
    n_all, sy_all = ss.n, ss.sy
    acc: dict[str, list[np.ndarray]] = {key: [] for key in
                                        ("r2", "rmse", "mae", "bias", "rpd", "ccc", "slope")}
    # 分块以控制内存：B×k 的索引矩阵在 k 很大时仍然轻量，但求和可以分批
    chunk = max(1, int(4e6 // max(k, 1)))
    for start in range(0, draws.shape[0], chunk):
        block = draws[start:start + chunk]
        sums = {
            "n": n_all[block].sum(axis=1),
            "sy": sy_all[block].sum(axis=1),
            "sy2": ss.sy2[block].sum(axis=1),
            "sp": ss.sp[block].sum(axis=1),
            "sp2": ss.sp2[block].sum(axis=1),
            "syp": ss.syp[block].sum(axis=1),
            "sres2": ss.sres2[block].sum(axis=1),
            "sabsres": ss.sabsres[block].sum(axis=1),
        }
        vals = _metrics_from_sums(sums)
        # RPIQ 需要重采样后的四分位，充分统计量给不出，单独用分位数近似跳过
        for key in acc:
            acc[key].append(vals[key])

    out: dict[str, tuple[float, float]] = {}
    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    for key, chunks in acc.items():
        arr = np.concatenate(chunks)
        arr = arr[np.isfinite(arr)]
        if arr.size < 10:
            continue
        out[key] = (float(np.percentile(arr, lo_q)), float(np.percentile(arr, hi_q)))
    return out


# --------------------------------------------------------------------------- #
# 分类
# --------------------------------------------------------------------------- #
def _ece(y: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    """期望校准误差（等频分箱）。"""
    order = np.argsort(proba)
    y_sorted, p_sorted = y[order], proba[order]
    total = 0.0
    for chunk_y, chunk_p in zip(np.array_split(y_sorted, n_bins),
                                np.array_split(p_sorted, n_bins)):
        if chunk_y.size == 0:
            continue
        total += chunk_y.size * abs(chunk_y.mean() - chunk_p.mean())
    return float(total / max(y.size, 1))


def _youden_threshold(y: np.ndarray, proba: np.ndarray) -> float:
    """在给定数据上取 Youden 指数最大的阈值。注意：只能在训练/内层用，不能在测试折上调。"""
    order = np.argsort(-proba)
    y_sorted = y[order]
    p_sorted = proba[order]
    n_pos = int(y_sorted.sum())
    n_neg = int(y_sorted.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    tpr = tps / n_pos
    fpr = fps / n_neg
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(p_sorted[idx])


def _first_proba_kind(sub: pd.DataFrame) -> str:
    if "proba_kind" not in sub.columns:
        return "native"
    vals = sub["proba_kind"].dropna()
    return str(vals.iloc[0]) if len(vals) else "native"


def oof_score_matrix(sub: pd.DataFrame, n_classes: int) -> tuple[np.ndarray, str]:
    """从 OOF 明细表重建分类分数。

    多分类必须返回 ``(n, k)`` 概率矩阵，不能只取 ``proba``（那是 max 概率，
    不是类别概率向量，算 Brier/AUROC 都会错位）。旧实现把 1D ``proba`` 交给
    ``classification_metrics``，后者又用 ``vstack([1-score, score])`` 造出 2 列，
    于是多分类 Brier 的 ``np.eye(n_classes)[y]`` 与 ``(n, 2)`` 广播失败。
    """
    if n_classes <= 2:
        use_proba = ("proba" in sub.columns
                     and pd.to_numeric(sub["proba"], errors="coerce").notna().any())
        if use_proba:
            score = pd.to_numeric(sub["proba"], errors="coerce").to_numpy(dtype=float)
            return score, _first_proba_kind(sub)
        return sub["y_pred"].to_numpy(dtype=float), "none"

    pcols = [f"p{c}" for c in range(n_classes)]
    if all(c in sub.columns for c in pcols):
        mat = sub[pcols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        return mat, _first_proba_kind(sub)

    # 没有概率列（模型只有 predict）：用硬标签 one-hot，并显式标记 proba_kind=none，
    # 让 Brier/ECE/log-loss 全部置 NaN，而不是拿硬标签冒充概率。
    yhat = pd.to_numeric(sub["y_pred"], errors="coerce").to_numpy(dtype=float)
    mat = np.zeros((len(sub), n_classes), dtype=float)
    ok = np.isfinite(yhat)
    labels = np.clip(np.rint(yhat[ok]).astype(int), 0, n_classes - 1)
    mat[np.flatnonzero(ok), labels] = 1.0
    return mat, "none"


def classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_classes: int = 2,
    threshold: float | None = None,
    proba_kind: str = "native",
) -> dict[str, float]:
    """y_score 为 (n,) 正类概率（二分类）或 (n, k) 概率矩阵（多分类）。

    ``proba_kind`` 声明 y_score 的真实身份：

    * ``"native"``   —— 真概率，全部指标可用；
    * ``"decision"`` —— sigmoid(decision_function)，**单调但非概率**。
      AUC/AP 不受单调变换影响（照算），但 Brier/ECE/log-loss 会算错——
      它们的分母语义是"概率"，这里一律置 NaN 而不是给一个看似合理的假数；
    * ``"none"``     —— 只有硬标签，AUC 退化为 (敏感度+特异度)/2，同样屏蔽概率类指标。

    这条三态是防"静默算错"的：报一个错的 Brier 比报一个 NaN 危险得多。
    """
    proba_kind = str(proba_kind or "native")
    proba_ok = proba_kind == "native"
    y = np.asarray(y_true).astype(int).ravel()
    score = np.asarray(y_score, dtype=np.float64)
    mask = np.isfinite(score if score.ndim == 1 else score.sum(axis=1))
    y = y[mask]
    score = score[mask] if score.ndim == 1 else score[mask]
    if y.size < 4:
        return {"auc": float("nan"), "balanced_accuracy": float("nan"), "f1_macro": float("nan"),
                "mcc": float("nan"), "brier": float("nan"), "sensitivity": float("nan"),
                "specificity": float("nan"), "accuracy": float("nan"), "log_loss": float("nan"),
                "auprc": float("nan"), "ece": float("nan"), "n": float(y.size)}

    out: dict[str, float] = {"n": float(y.size)}
    if n_classes == 2:
        # 概率/得分统一裁剪到 [0,1]：PLS-DA、decision_function 等可能给出越界值，
        # 不裁剪会让 Brier/log_loss 直接抛 ValueError，整折报废。
        score = np.clip(score, 0.0, 1.0)
        proba = score if score.ndim == 1 else score[:, 1]
        yhat = (proba >= (0.5 if threshold is None else threshold)).astype(int)
        try:
            out["auc"] = float(roc_auc_score(y, proba))
        except ValueError:
            out["auc"] = float("nan")
        try:
            out["auprc"] = float(average_precision_score(y, proba))
        except ValueError:
            out["auprc"] = float("nan")
        tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
        out["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) else float("nan")
        out["specificity"] = float(tn / (tn + fp)) if (tn + fp) else float("nan")
        out["brier"] = (float(brier_score_loss(y, proba)) if (proba_ok and len(np.unique(y)) > 1)
                        else float("nan"))
        out["ece"] = _ece(y.astype(float), proba) if proba_ok else float("nan")
        eps = 1e-12
        out["log_loss"] = (float(log_loss(y, np.clip(proba, eps, 1 - eps), labels=[0, 1]))
                           if (proba_ok and len(np.unique(y)) > 1) else float("nan"))
    else:
        if score.ndim == 2:
            if score.shape[1] != n_classes:
                raise ValueError(
                    f"多分类需要 (n, {n_classes}) 概率矩阵，收到 {score.shape}；"
                    f"请先用 oof_score_matrix 从 p0..p{n_classes - 1} 重建")
            proba = _normalize_rows(np.clip(score, 0.0, 1.0))
        else:
            # 1D 分数在多分类下只能是硬标签；one-hot 化后按 proba_kind=none 处理，
            # 绝不能像旧实现那样补成 2 列概率（会让 Brier 广播失败）。
            labels = np.clip(np.rint(score).astype(int), 0, n_classes - 1)
            proba = np.zeros((y.size, n_classes), dtype=float)
            proba[np.arange(y.size), labels] = 1.0
            proba_kind, proba_ok = "none", False
        yhat = np.argmax(proba, axis=1)
        try:
            out["auc"] = float(roc_auc_score(y, proba, multi_class="ovr", average="macro"))
        except ValueError:
            out["auc"] = float("nan")
        out["brier"] = (float(np.mean(np.sum((proba - np.eye(n_classes)[y]) ** 2, axis=1)))
                        if proba_ok else float("nan"))
        out["ece"] = (_ece((y == yhat).astype(float), proba.max(axis=1))
                      if proba_ok else float("nan"))
        # 多分类的敏感度/特异度取宏平均（一对多）
        cm = confusion_matrix(y, yhat, labels=list(range(n_classes)))
        sens, spec = [], []
        for c in range(n_classes):
            tp_c = cm[c, c]
            fn_c = cm[c, :].sum() - tp_c
            fp_c = cm[:, c].sum() - tp_c
            tn_c = cm.sum() - tp_c - fn_c - fp_c
            sens.append(tp_c / (tp_c + fn_c) if (tp_c + fn_c) else np.nan)
            spec.append(tn_c / (tn_c + fp_c) if (tn_c + fp_c) else np.nan)
        out["sensitivity"] = float(np.nanmean(sens))
        out["specificity"] = float(np.nanmean(spec))
        try:
            out["log_loss"] = (float(log_loss(y, np.clip(proba, 1e-12, 1 - 1e-12),
                                              labels=list(range(n_classes))))
                               if proba_ok else float("nan"))
        except ValueError:
            out["log_loss"] = float("nan")

    out["balanced_accuracy"] = float(balanced_accuracy_score(y, yhat))
    out["f1_macro"] = float(f1_score(y, yhat, average="macro", zero_division=0))
    out["mcc"] = float(matthews_corrcoef(y, yhat))
    out["accuracy"] = float(np.mean(y == yhat))
    return out


def bootstrap_classification(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    n_classes: int = 2,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
    proba_kind: str = "native",
) -> dict[str, tuple[float, float]]:
    """按组重采样（分类指标无法用充分统计量，直接重算；B 建议 <= 1000）。

    ``proba_kind != "native"`` 时概率类指标恒为 NaN，因此在 ``arr.size < 10``
    处被过滤掉——CI 表里只会出现 AUC/敏感度这类不依赖概率语义的指标。
    """
    y = np.asarray(y_true).astype(int).ravel()
    score = np.asarray(y_score, dtype=np.float64)
    g = np.asarray(groups, dtype=object)
    if y.size < 6:
        return {}
    uniq = pd.unique(g)
    if uniq.size < 3:
        return {}
    pos = {gg: i for i, gg in enumerate(uniq)}
    member = {i: np.flatnonzero(np.array([pos[x] for x in g], dtype=np.int64) == i)
              for i in range(uniq.size)}

    rng = np.random.default_rng(seed)
    keys = ("auc", "balanced_accuracy", "f1_macro", "mcc", "brier", "sensitivity", "specificity")
    acc: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(int(n_boot)):
        picked = rng.integers(0, uniq.size, size=uniq.size)
        idx = np.concatenate([member[i] for i in picked])
        if np.unique(y[idx]).size < 2:
            continue
        m = classification_metrics(y[idx], score[idx], n_classes=n_classes,
                                   proba_kind=proba_kind)
        for k in keys:
            acc[k].append(m.get(k, float("nan")))
    out: dict[str, tuple[float, float]] = {}
    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    for k, vals in acc.items():
        arr = np.asarray([v for v in vals if np.isfinite(v)])
        if arr.size < 10:
            continue
        out[k] = (float(np.percentile(arr, lo_q)), float(np.percentile(arr, hi_q)))
    return out


# --------------------------------------------------------------------------- #
# 跨 repeat 聚合
# --------------------------------------------------------------------------- #
def aggregate_repeats(
    fold_df: pd.DataFrame,
    metric_cols: list[str],
    group_keys: tuple[str, ...] = ("model", "prep", "target"),
) -> pd.DataFrame:
    """先按 repeat 求折均值，再对 repeat 求 mean±sd。

    直接把 R×k 个折当独立样本会低估 sd（伪重复），这是审稿常抓的点。
    """
    if fold_df.empty:
        return pd.DataFrame()
    keys = [k for k in group_keys if k in fold_df.columns]
    per_repeat = (fold_df.groupby(keys + ["repeat"], dropna=False)[metric_cols]
                  .mean().reset_index())
    rows = []
    for name, sub in per_repeat.groupby(keys, dropna=False):
        name_tuple = name if isinstance(name, tuple) else (name,)
        row: dict[str, Any] = dict(zip(keys, name_tuple))
        row["n_repeats"] = int(sub["repeat"].nunique())
        for col in metric_cols:
            vals = sub[col].to_numpy(dtype=float)
            row[f"{col}_mean"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
            row[f"{col}_sd"] = float(np.nanstd(vals, ddof=1)) if np.isfinite(vals).sum() > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def fold_metric_table(fold_df: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    return fold_df[[c for c in (["repeat", "fold", "model", "prep"] + metric_cols) if c in fold_df.columns]].copy()


# --------------------------------------------------------------------------- #
# 口径判定：折均值可用，还是必须切到池化 OOF
# --------------------------------------------------------------------------- #
# 有效折少于这个数时，「折均值」不再是一个可靠的性能估计：退化折（测试样品 <2、
# 主指标 NaN）会被 nanmean 跳过，于是 k 折的平均其实只来自少数几个折。留一法下每折
# 只有 1 个测试样品，逐折 R² 在单点上退化（−0.7/−0.9 这种数），折均值会给出与池化
# 口径完全相反的结论。
MIN_OK_FOLDS = 3


def mark_metric_basis(
    summary: pd.DataFrame,
    folds: pd.DataFrame,
    group_keys: tuple[str, ...] = ("target", "model"),
    min_ok_folds: int = MIN_OK_FOLDS,
) -> pd.DataFrame:
    """给汇总表补 ``n_folds_ok`` / ``n_folds_degenerate`` / ``metric_basis``。

    ``metric_basis='fold'`` 表示折均值可用；``'pooled'`` 表示有效折不足、该行必须改用
    池化 OOF 指标（``oof_group_metrics`` 产出的 ``r2`` / ``auc`` 原始列）。

    stage_cv 与 rebuild_tables **都必须**走这里。只在 stage_cv 里判口径，会让「只重算表、
    不重训」这条路径静默退回折均值，而且不报警——那正是口径修复最容易被绕过的地方。
    """
    if summary is None or summary.empty:
        return summary
    df = summary.copy()
    keys = [k for k in group_keys if k in df.columns]
    df = df.drop(columns=[c for c in ("n_folds_ok", "n_folds_degenerate") if c in df.columns])
    if folds is not None and not folds.empty and "status" in folds.columns and keys:
        f = folds.copy()
        status = f["status"].astype(str)
        # status 由 cv.py 按「主指标是否有限」写成 ok / degenerate，
        # 数 status 比逐指标判断更稳（回归看 r2、分类看 auc，列名不同）。
        f["_ok"] = status.eq("ok").astype(int)
        f["_degen"] = status.eq("degenerate").astype(int)
        cnt = (f.groupby(keys, dropna=False)
               .agg(n_folds_ok=("_ok", "sum"), n_folds_degenerate=("_degen", "sum"))
               .reset_index())
        df = df.merge(cnt, on=keys, how="left")
        n_ok = pd.to_numeric(df["n_folds_ok"], errors="coerce").fillna(0)
        df["metric_basis"] = np.where(n_ok >= int(min_ok_folds), "fold", "pooled")
        df["n_folds_ok"] = n_ok.astype(int)
        df["n_folds_degenerate"] = (pd.to_numeric(df["n_folds_degenerate"], errors="coerce")
                                    .fillna(0).astype(int))
    else:
        # 没有 status 列（手工拼的逐折表）时无从判断：按折均值处理，但如实留痕，
        # 不要伪装成「已判定」。
        df["n_folds_ok"] = np.nan
        df["n_folds_degenerate"] = np.nan
        df["metric_basis"] = "fold"
    return df


def describe_pooled_rows(summary: pd.DataFrame, strategy: str | None = None) -> str | None:
    """把「折均值不可用」的行写成一句可直接 warn 的话；没有这类行时返回 None。

    ``strategy="holdout_random"`` 时换成中性说法：这种划分本来就只产 1 个外层折，
    折均值不可用是**设计如此**，池化 OOF 就是正确口径。照搬「有效折不足」的警告
    会让人以为哪里出了错，而实际上什么都没错。
    """
    if summary is None or summary.empty or "metric_basis" not in summary.columns:
        return None
    thin = summary[summary["metric_basis"].astype(str).eq("pooled")]
    if thin.empty:
        return None
    if strategy == "holdout_random":
        return ("单次留出划分（holdout_random）：每个模型只有 1 个外层折，折均值不可用属预期，"
                "指标按池化 OOF（即测试集）引用——这不是错误，也不要用 repeats 去凑多折。")
    parts = []
    for _, r in thin.head(6).iterrows():
        ok, deg = r.get("n_folds_ok"), r.get("n_folds_degenerate")
        if pd.notna(ok) and pd.notna(deg):
            parts.append(f"{r['model']}({int(ok)}/{int(ok) + int(deg)})")
        else:
            parts.append(str(r["model"]))
    more = f" 等 {len(thin)} 个组合" if len(thin) > 6 else ""
    return (f"有效折不足（<{MIN_OK_FOLDS}）：{'、'.join(parts)}{more} 的折均值不可用"
            f"（退化折的测试样品 <2，指标为 NaN，被 nanmean 跳过）。"
            f"这些行已标记 metric_basis=pooled，排名与报告改用池化 OOF 指标。")
