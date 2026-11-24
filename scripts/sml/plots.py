"""论文级图表（8 张）。全部 matplotlib + Agg，同时导出 PNG 与 PDF（矢量）。

为什么不用 plotly
-----------------
本机没装 kaleido，``fig.write_image`` 会直接报错。plotly 只适合导交互 HTML，
论文插图一律走 matplotlib（300 dpi PNG + 矢量 PDF）。

图目录
------
F1 预测-实测散点（含 1:1 线、斜率、R²/RMSE/RPD）   F2 残差图（含 ±1.96SD 带）
F3 交叉验证分布（重复×折的箱线 + 散点）            F4 Williams plot（leverage vs 学生化残差）
F5 校准（回归=BA 图；分类=校准曲线）               F6 Y-randomization 零分布
F7 重要性-波数映射（柱状 + 平均谱底图）            F8 模型比较（点估计 + 95% CI）
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import as_path, configure_matplotlib, ensure_dir, info, warn

__all__ = ["render_all", "PALETTE", "FIGURE_INDEX"]

PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7",
           "#56B4E9", "#D55E00", "#F0E442", "#666666",
           "#44AA99", "#AA4499"]

FIGURE_INDEX = {
    "F1": "OOF 性能图（回归=预测-实测散点；分类=ROC，多分类为逐类 OVR）；统计量为 OOF 池化口径，与表 T2 的 repeat 均值口径不可混引",
    "F2": "残差图（仅回归）；偏差/±1.96SD 亦为 OOF 池化口径",
    "F3": "交叉验证分布（逐折；黑菱形为折均值，与 T2 的重复间均值略有差异）",
    "F4": "Williams plot（适用域）",
    "F5": "校准 / 一致性（回归=Bland-Altman；分类=可靠性曲线，多分类为逐类 OVR；OOF 池化口径）",
    "F6": "Y-randomization 零分布",
    "F7": "重要性-波数映射",
    "F8": "模型比较（95% CI）",
}


def _plt(cfg: dict):
    style = as_path(__file__).resolve().parent.parent.parent / "assets" / "style" / "paper.mplstyle"
    return configure_matplotlib(dpi=int(cfg.get("report", {}).get("dpi", 300)), style_path=style)


def _save(fig, out_dir: Path, code: str, cfg: dict) -> list[Path]:
    formats = cfg.get("report", {}).get("formats", ["png", "pdf"])
    dpi = int(cfg.get("report", {}).get("dpi", 300))
    ensure_dir(out_dir)
    paths = []
    for ext in formats:
        p = out_dir / f"{code}.{ext}"
        fig.savefig(p, dpi=dpi if ext == "png" else None, bbox_inches="tight")
        paths.append(p)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return paths


def _color(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


def _is_proba_col(name) -> bool:
    """``p0`` / ``p1`` … 这类逐类概率列（``prep``、``proba`` 不算）。"""
    s = str(name)
    return len(s) > 1 and s[0] == "p" and s[1:].isdigit()


def _class_name(c: int, label_map: dict | None) -> str:
    if label_map and c in label_map:
        return str(label_map[c])
    return f"类{c}"


# --------------------------------------------------------------------------- #
def fig_pred_obs(oof: pd.DataFrame, model: str, task: str, out_dir: Path, cfg: dict,
                 label_map: dict | None = None) -> list[Path]:
    plt = _plt(cfg)
    sub = oof[oof["model"] == model]
    if sub.empty:
        return []
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    if task == "regression":
        y, p = sub["y_true"].to_numpy(), sub["y_pred"].to_numpy()
        ax.scatter(y, p, s=14, alpha=0.65, edgecolors="none", color=_color(0), label="OOF 预测")
        lo = float(min(y.min(), p.min()))
        hi = float(max(y.max(), p.max()))
        pad = 0.04 * (hi - lo if hi > lo else 1.0)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ls="--", lw=1, color="#444444", label="1:1 线")
        from .metrics import regression_metrics
        m = regression_metrics(y, p)
        ax.text(0.04, 0.96,
                f"$R^2$ = {m['r2']:.3f}\nRMSE = {m['rmse']:.3f}\nRPD = {m['rpd']:.2f}\n斜率 = {m['slope']:.3f}\n"
                f"（OOF 池化，见表 T2 口径说明）",
                transform=ax.transAxes, va="top", fontsize=6.8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbbbbb", alpha=0.9))
        ax.set_xlabel("实测值")
        ax.set_ylabel("预测值")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
    else:
        from sklearn.metrics import roc_auc_score, roc_curve

        from .metrics import _normalize_rows

        y = sub["y_true"].to_numpy().astype(int)
        pcols = [c for c in sub.columns if _is_proba_col(c)]
        if len(pcols) > 2:
            # 多分类必须走 per-class OVR。旧实现直接把多类 y 和 1D 分数交给
            # roc_curve，sklearn 抛 "multiclass format is not supported"，
            # 整张 F1 静默消失（只在控制台留一行 "F1 生成失败"）。
            proba = _normalize_rows(sub[pcols].apply(pd.to_numeric, errors="coerce")
                                    .to_numpy(dtype=float))
            aucs: list[float] = []
            for c in range(proba.shape[1]):
                ybin = (y == c).astype(int)
                if ybin.min() == ybin.max():
                    continue      # OOF 中没有该类的正负样本，这一类的 ROC 无定义
                fpr, tpr, _ = roc_curve(ybin, proba[:, c])
                auc_c = roc_auc_score(ybin, proba[:, c])
                ax.plot(fpr, tpr, lw=1.3, color=_color(c),
                        label=f"{_class_name(c, label_map)}: AUROC={auc_c:.3f}")
                aucs.append(auc_c)
            if not aucs:
                warn(f"{model} 的 OOF 里没有同时含正负样本的类别，跳过 ROC")
                return []
            ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#888888")
            ax.text(0.04, 0.96, f"宏平均 AUROC = {float(np.mean(aucs)):.3f}\n（OOF 池化，OVR）",
                    transform=ax.transAxes, va="top", fontsize=6.8,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bbbbbb", alpha=0.9))
            ax.legend(loc="lower right", fontsize=6.4, frameon=False)
        else:
            score = (pd.to_numeric(sub["proba"], errors="coerce").to_numpy(dtype=float)
                     if "proba" in sub else sub["y_pred"].to_numpy(dtype=float))
            if not np.isfinite(score).all():
                score = sub["y_pred"].to_numpy(dtype=float)
            if len(np.unique(y)) > 1:
                fpr, tpr, _ = roc_curve(y, score)
                ax.plot(fpr, tpr, lw=1.6, color=_color(0),
                        label=f"AUROC = {roc_auc_score(y, score):.3f}（OOF 池化）")
                ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#888888")
                ax.legend(loc="lower right", fontsize=7.2, frameon=False)
        ax.set_xlabel("假阳性率")
        ax.set_ylabel("真阳性率")
    ax.set_title(f"OOF 性能（{model}）", fontsize=9)
    ax.tick_params(labelsize=7.5)
    fig.tight_layout()
    return _save(fig, out_dir, "F1_pred_vs_obs", cfg)


def fig_residual(oof: pd.DataFrame, model: str, out_dir: Path, cfg: dict,
                 task: str = "regression") -> list[Path]:
    plt = _plt(cfg)
    sub = oof[oof["model"] == model]
    if sub.empty:
        return []
    if task != "regression":
        # 分类任务里 y 是类别编号，对它做「预测−实测」残差只是整数抖动，
        # 既看不出系统偏差也看不出异方差，画出来等于给审稿人送把柄。
        info("分类任务不画残差图（F2 仅对回归有意义）")
        return []
    y, p = sub["y_true"].to_numpy(), sub["y_pred"].to_numpy()
    resid = p - y
    sd = float(np.std(resid, ddof=1)) if resid.size > 1 else 0.0
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.scatter(y, resid, s=14, alpha=0.65, edgecolors="none", color=_color(2))
    ax.axhline(0, color="#333333", lw=1)
    for k, ls in ((1.96, "--"), (-1.96, "--")):
        ax.axhline(k * sd, color="#D55E00", lw=1, ls=ls)
    ax.text(0.02, 0.98, f"±1.96SD = ±{1.96 * sd:.3f}\n偏差 = {resid.mean():+.4f}\n（OOF 池化）",
            transform=ax.transAxes, va="top", fontsize=7.0,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
    ax.set_xlabel("实测值")
    ax.set_ylabel("残差（预测−实测）")
    ax.set_title("残差图（检查系统偏差与异方差）", fontsize=9)
    ax.tick_params(labelsize=7.5)
    fig.tight_layout()
    return _save(fig, out_dir, "F2_residual", cfg)


def fig_cv_dist(folds: pd.DataFrame, task: str, out_dir: Path, cfg: dict) -> list[Path]:
    plt = _plt(cfg)
    metric = "r2" if task == "regression" else "auc"
    if metric not in folds:
        return []
    df = (folds[folds["status"].astype(str).isin(["ok", "degenerate"])]
          if "status" in folds else folds)
    models = list(dict.fromkeys(df["model"]))
    if not models:
        return []
    data = [df[df["model"] == m][metric].dropna().to_numpy() for m in models]
    fig, ax = plt.subplots(figsize=(max(3.4, 0.75 * len(models) + 2.2), 3.2))
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(_color(i))
        box.set_alpha(0.45)
        box.set_edgecolor(_color(i))
    for i, vals in enumerate(data):
        x = np.random.default_rng(i).normal(i + 1, 0.045, size=vals.size)
        ax.scatter(x, vals, s=8, color=_color(i), alpha=0.75, edgecolors="none", zorder=3)
        ax.scatter([i + 1], [np.mean(vals)], marker="D", s=18, color="#222222", zorder=4)
    ax.set_xticks(range(1, len(models) + 1))
    ax.set_xticklabels(models, rotation=0, fontsize=8)
    ax.set_ylabel("R²" if metric == "r2" else "AUROC")
    ax.set_title(f"交叉验证稳定性（{len(data[0])} 个折）\n黑菱形=均值", fontsize=9)
    ax.tick_params(labelsize=7.5)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    fig.tight_layout()
    return _save(fig, out_dir, "F3_cv_distribution", cfg)


def fig_williams(ad_scores: pd.DataFrame, oof: pd.DataFrame, model: str,
                 out_dir: Path, cfg: dict) -> list[Path]:
    plt = _plt(cfg)
    if ad_scores.empty or "leverage" not in ad_scores:
        return []
    sc = ad_scores.copy()
    if "y_true" not in sc and not oof.empty:
        sub = oof[oof["model"] == model]
        if "row_id" in sub and "row_id" in sc:
            sc = sc.merge(sub[["row_id", "y_true", "y_pred"]], on="row_id", how="left")
    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    x = sc["leverage"].to_numpy()
    if {"y_true", "y_pred"}.issubset(sc.columns):
        resid = (sc["y_pred"] - sc["y_true"]).to_numpy()
        sd = np.nanstd(resid, ddof=1) or 1.0
        yv = resid / sd
        ylabel = "学生化残差"
    else:
        yv = sc["q"].to_numpy() if "q" in sc else np.zeros_like(x)
        ylabel = "Q 残差（无预测，退化为诊断量）"
    ax.scatter(x, yv, s=14, alpha=0.65, edgecolors="none", color=_color(0))
    h_thr = float(sc["leverage_thr"].iloc[0]) if "leverage_thr" in sc else float(np.max(x))
    ax.axvline(h_thr, ls="--", lw=1, color="#D55E00")
    ax.axhline(3, ls=":", lw=1, color="#666666")
    ax.axhline(-3, ls=":", lw=1, color="#666666")
    ax.text(h_thr, ax.get_ylim()[1], f" h* = {h_thr:.3f}", va="top", fontsize=7, color="#D55E00")
    ax.set_xlabel("Leverage（h）")
    ax.set_ylabel(ylabel)
    ax.set_title("Williams plot\n右上/右下 = 域外样本", fontsize=9)
    ax.tick_params(labelsize=7.5)
    fig.tight_layout()
    return _save(fig, out_dir, "F4_williams", cfg)


def fig_calibration(oof: pd.DataFrame, model: str, task: str, out_dir: Path, cfg: dict,
                    label_map: dict | None = None) -> list[Path]:
    plt = _plt(cfg)
    sub = oof[oof["model"] == model]
    if sub.empty:
        return []
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    if task == "regression":
        from .calib import bland_altman
        y, p = sub["y_true"].to_numpy(), sub["y_pred"].to_numpy()
        ba = bland_altman(y, p)
        diff = p - y
        mean_val = (p + y) / 2
        ax.scatter(mean_val, diff, s=14, alpha=0.65, edgecolors="none", color=_color(3))
        ax.axhline(ba["bias"], color="#0072B2", lw=1, label=f"偏差 {ba['bias']:+.4f}")
        ax.axhline(ba["loa_low"], ls="--", lw=1, color="#D55E00")
        ax.axhline(ba["loa_high"], ls="--", lw=1, color="#D55E00")
        ax.fill_between([mean_val.min(), mean_val.max()], ba["loa_low"], ba["loa_high"],
                        color="#D55E00", alpha=0.08)
        ax.set_xlabel("均值（实测, 预测）")
        ax.set_ylabel("预测 − 实测")
        ax.legend(fontsize=7, frameon=False, loc="upper right")
        ax.set_title("Bland-Altman 一致性分析", fontsize=9)
    else:
        from .calib import calibration_curve, expected_calibration_error
        from .metrics import _normalize_rows

        kind = "native"
        if "proba_kind" in sub and sub["proba_kind"].notna().any():
            kind = str(sub["proba_kind"].dropna().iloc[0])
        if kind != "native":
            warn(f"{model} 没有原生概率接口（proba_kind={kind}），跳过校准曲线："
                 f"在 sigmoid(decision_function) 上画可靠性曲线属于误导性图表")
            return []
        y = sub["y_true"].to_numpy().astype(int)
        pcols = [c for c in sub.columns if _is_proba_col(c)]
        if len(pcols) > 2:
            # 多分类：逐类 one-vs-rest 可靠性曲线。旧实现只拿 ``proba``（=各类最大概率）
            # 当「阳性概率」，且只在二分类时画曲线——三分类及以上会产出一张只有
            # 坐标轴、没有任何数据的空白图，还被写进报告的「可用图」清单。
            proba = _normalize_rows(sub[pcols].apply(pd.to_numeric, errors="coerce")
                                    .to_numpy(dtype=float))
            eces: list[float] = []
            for c in range(proba.shape[1]):
                ybin = (y == c).astype(int)
                if ybin.min() == ybin.max():
                    continue
                curve = calibration_curve(ybin, proba[:, c], n_bins=8)
                if curve.empty:
                    continue
                # Wilson 区间理论上包含观测率，但浮点舍入会让下界偶尔比观测率大 1e-17，
                # matplotlib 不接受负 yerr（会整张图抛异常）。这里按 0 截断。
                lo_err = np.maximum(0.0, (curve["obs_rate"] - curve["ci_low"]).to_numpy())
                hi_err = np.maximum(0.0, (curve["ci_high"] - curve["obs_rate"]).to_numpy())
                ax.errorbar(curve["pred_mean"], curve["obs_rate"],
                            yerr=[lo_err, hi_err],
                            fmt="o-", ms=3, lw=1.1, capsize=2, color=_color(c),
                            label=f"{_class_name(c, label_map)}（OVR）")
                eces.append(expected_calibration_error(ybin, proba[:, c], n_bins=8))
            if not eces:
                warn(f"{model} 的 OOF 里没有同时含正负样本的类别，跳过校准曲线")
                return []
            ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#777777")
            ax.text(0.04, 0.96, f"宏平均 ECE = {float(np.mean(eces)):.3f}\n（OOF 池化，OVR）",
                    transform=ax.transAxes, va="top", fontsize=6.8,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bbbbbb", alpha=0.9))
            ax.legend(fontsize=6.4, frameon=False, loc="lower right")
            ax.set_xlabel("预测概率（该类）")
            ax.set_ylabel("实测该类占比")
            ax.set_title("校准曲线（逐类 OVR）", fontsize=9)
        else:
            proba = (pd.to_numeric(sub["proba"], errors="coerce").to_numpy(dtype=float)
                     if "proba" in sub else np.asarray([], dtype=float))
            if proba.size == 0 or not np.isfinite(proba).any():
                warn(f"{model} 缺 proba 列，跳过校准曲线")
                return []
            if len(np.unique(y)) == 2:
                ybin = (y == y.max()).astype(int)
                curve = calibration_curve(ybin, proba, n_bins=8)
                if not curve.empty:
                    lo_err = np.maximum(0.0, (curve["obs_rate"] - curve["ci_low"]).to_numpy())
                    hi_err = np.maximum(0.0, (curve["ci_high"] - curve["obs_rate"]).to_numpy())
                    ax.errorbar(curve["pred_mean"], curve["obs_rate"],
                                yerr=[lo_err, hi_err],
                                fmt="o-", ms=4, lw=1.2, capsize=2, color=_color(0), label="观测率（Wilson 95%CI）")
                ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#777777", label="完美校准")
                ece = expected_calibration_error(ybin, proba, n_bins=8)
                ax.text(0.04, 0.96, f"ECE = {ece:.3f}\n（OOF 池化）", transform=ax.transAxes,
                        va="top", fontsize=7.0)
                ax.legend(fontsize=7, frameon=False, loc="lower right")
            ax.set_xlabel("预测概率")
            ax.set_ylabel("实测阳性率")
            ax.set_title("校准曲线", fontsize=9)
    ax.tick_params(labelsize=7.5)
    fig.tight_layout()
    return _save(fig, out_dir, "F5_calibration", cfg)


def fig_permutation(perm: dict, out_dir: Path, cfg: dict) -> list[Path]:
    vals = perm.get("null_values") or []
    if not vals:
        return []
    plt = _plt(cfg)
    fig, ax = plt.subplots(figsize=(3.8, 3.0))
    ax.hist(vals, bins=min(30, max(8, len(vals) // 5)), color=_color(4), alpha=0.75,
            edgecolor="white", lw=0.5, label="置换零分布")
    obs = perm.get("observed")
    if obs is not None and np.isfinite(obs):
        ax.axvline(obs, color="#D55E00", lw=1.6, label=f"观测值 {obs:.3f}")
    ax.set_xlabel(perm.get("metric", "metric"))
    ax.set_ylabel("频数")
    ax.set_title(f"Y-randomization：p = {perm.get('p_value', float('nan')):.4f}\n"
                 f"（n_perm={perm.get('n_perm')}）", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.tick_params(labelsize=7.5)
    fig.tight_layout()
    return _save(fig, out_dir, "F6_y_randomization", cfg)


def _spectral_axis(ds) -> tuple[np.ndarray | None, np.ndarray | None]:
    """返回 (波数数组, 对应特征列下标)。优先用列名里的波数，其次退回 wavelength_axis。"""
    vals, idx = [], []
    for j, name in enumerate(ds.feature_names):
        try:
            vals.append(float(str(name)))
        except (TypeError, ValueError):
            continue
        idx.append(j)
    if len(idx) >= 2:
        order = np.argsort(vals)
        return np.asarray(vals, dtype=float)[order], np.asarray(idx, dtype=int)[order]
    if getattr(ds, "wavelength", None) is not None and len(ds.wavelength) == ds.n_features:
        return np.asarray(ds.wavelength, dtype=float), np.arange(ds.n_features)
    return None, None


def fig_importance(imp: pd.DataFrame, ds, out_dir: Path, cfg: dict,
                   top_k: int = 15) -> list[Path]:
    if imp.empty:
        return []
    plt = _plt(cfg)
    bands = imp[imp["kind"] == "band"].copy() if "kind" in imp else imp.copy()
    if bands.empty:
        bands = imp.copy()
    has_center = "center" in bands and np.isfinite(pd.to_numeric(bands["center"], errors="coerce")).any()

    labels = ([str(v) for v in bands["label"]] if "label" in bands
              else [f"f{i}" for i in range(len(bands))])
    wl0, _col0 = _spectral_axis(ds)
    if (not has_center) and wl0 is None:
        # 非光谱输入（工艺参数等表格数据）：没有波数轴，横轴必须用特征名。
        # 旧实现落到 band_index 分支，横轴只剩「波段序号」、读不出哪个特征重要，
        # 下半幅「平均谱」还是空 panel（只有一个「—」刻度），整张图等于废掉。
        y = bands["delta_mean"].to_numpy(dtype=float)
        xpos = np.arange(len(bands))
        fig, ax = plt.subplots(figsize=(4.8, max(2.6, 0.42 * len(bands) + 1.4)))
        ax.bar(xpos, y, color=_color(0), alpha=0.85, width=0.8)
        ax.axhline(0, color="#888888", lw=0.8)
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_xlabel("特征")
        ax.set_ylabel("ΔR²（置换后下降）")
        ax.set_title("特征级置换重要性：打乱该特征后的性能损失", fontsize=9)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=7.5)
        fig.tight_layout()
        return _save(fig, out_dir, "F7_importance_map", cfg)

    fig, axes = plt.subplots(2, 1, figsize=(4.8, 4.2), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1]})
    ax, ax2 = axes
    wl, col_idx = _spectral_axis(ds)

    if has_center:
        x = pd.to_numeric(bands["center"], errors="coerce").to_numpy(dtype=float)
        order = np.argsort(x)
        x = x[order]
        y = bands["delta_mean"].to_numpy(dtype=float)[order]
        spacing = float(np.median(np.diff(x))) if x.size > 2 else 1.0
        width = abs(spacing) * 0.85 if np.isfinite(spacing) and spacing != 0 else 1.0
        ax.bar(x, y, width=width, color=_color(0), alpha=0.85)
        ax.axhline(0, color="#888888", lw=0.8)
        xlabel = r"波数 (cm$^{-1}$)"
        invert = True
    else:
        x = bands["band_index"].to_numpy(dtype=float)
        y = bands["delta_mean"].to_numpy(dtype=float)
        ax.bar(x, y, color=_color(0), alpha=0.85, width=0.85)
        xlabel = "波段序号"
        invert = False

    ax.set_ylabel("ΔR²（置换后下降）")
    ax.set_title("波段级置换重要性：打乱整段波长后的性能损失", fontsize=9)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=7.5)

    if wl is not None and col_idx is not None:
        mean_spec = np.nanmean(np.asarray(ds.X, dtype=float)[:, col_idx], axis=0)
        ax2.plot(wl, mean_spec, lw=0.7, color="#444444")
        ax2.set_ylabel("平均强度", fontsize=8)
        ax2.set_xlim(float(wl.min()), float(wl.max()))
        if invert:
            ax2.invert_xaxis()
    else:
        ax2.set_ylabel("—", fontsize=8)
    ax2.set_xlabel(xlabel)
    ax2.tick_params(labelsize=7)
    fig.tight_layout()
    return _save(fig, out_dir, "F7_importance_map", cfg)


def fig_compare(ci: pd.DataFrame, summary: pd.DataFrame, task: str,
                out_dir: Path, cfg: dict) -> list[Path]:
    metric = "rmse" if task == "regression" else "auc"
    if summary.empty or f"{metric}_mean" not in summary:
        return []
    plt = _plt(cfg)
    df = summary.sort_values(f"{metric}_mean", ascending=(metric == "rmse"))
    models = df["model"].tolist()
    means = df[f"{metric}_mean"].to_numpy(dtype=float)
    sds = df[f"{metric}_sd"].to_numpy(dtype=float) if f"{metric}_sd" in df else np.zeros_like(means)
    fig, ax = plt.subplots(figsize=(3.8, max(2.4, 0.42 * len(models) + 1.2)))
    ypos = np.arange(len(models))
    ax.errorbar(means, ypos, xerr=sds, fmt="o", ms=5, lw=1.2, capsize=3,
                color=_color(0), ecolor="#777777")
    # 若已有 bootstrap CI，用更细的线叠一层（区间可能更宽，两种口径都展示）
    if not ci.empty and "metric" in ci:
        sub = ci[ci["metric"] == metric]
        for _, row in sub.iterrows():
            if row["model"] in models:
                i = models.index(row["model"])
                ax.plot([row["ci_low"], row["ci_high"]], [i, i], lw=3, alpha=0.35, color=_color(1))
    ax.set_yticks(ypos)
    ax.set_yticklabels(models, fontsize=8)
    ax.set_xlabel("RMSE（越小越好）" if metric == "rmse" else "AUROC（越大越好）")
    ax.set_title("模型比较\n点=repeat均值，细线=±sd，粗线=95% CI", fontsize=9)
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=7.5)
    fig.tight_layout()
    return _save(fig, out_dir, "F8_model_comparison", cfg)


# --------------------------------------------------------------------------- #
def render_all(ctx, best_model: str | None = None) -> dict[str, list[str]]:
    """从 run 目录读取产物，渲染全部可用的图。缺数据的图自动跳过并记录。"""
    cfg = ctx.cfg
    ds = ctx.ds
    run = ctx.run_dir
    fig_dir = ensure_dir(run / "figures")
    task = ds.task
    made: dict[str, list[str]] = {}
    skipped: list[str] = []

    oof_p = run / "cv" / "oof_predictions.csv"
    folds_p = run / "cv" / "fold_results.csv"
    oof = pd.read_csv(oof_p) if oof_p.exists() else pd.DataFrame()
    folds = pd.read_csv(folds_p) if folds_p.exists() else pd.DataFrame()
    if oof.empty or not best_model:
        warn("缺少 OOF 预测，无法出图")
        return {}

    def _do(code: str, fn, *args):
        try:
            paths = fn(*args)
        except Exception as exc:  # noqa: BLE001
            warn(f"{code} 生成失败：{type(exc).__name__}: {exc}")
            skipped.append(code)
            return
        if paths:
            made[code] = [str(p) for p in paths]
            info(f"  {code} -> {Path(paths[0]).name}")
        else:
            skipped.append(code)

    _do("F1", fig_pred_obs, oof, best_model, task, fig_dir, cfg, ds.label_map)
    _do("F2", fig_residual, oof, best_model, fig_dir, cfg, task)
    if not folds.empty:
        _do("F3", fig_cv_dist, folds, task, fig_dir, cfg)

    ad_p = run / "ad" / "ad_scores.csv"
    ad_scores = pd.read_csv(ad_p) if ad_p.exists() else pd.DataFrame()
    if not ad_scores.empty:
        _do("F4", fig_williams, ad_scores, oof, best_model, fig_dir, cfg)

    _do("F5", fig_calibration, oof, best_model, task, fig_dir, cfg, ds.label_map)

    perm_p = run / "perm" / "permutation.json"
    if perm_p.exists():
        perm = json.loads(perm_p.read_text(encoding="utf-8"))
        _do("F6", fig_permutation, perm, fig_dir, cfg)

    imp_p = run / "interp" / "importance.csv"
    if imp_p.exists():
        imp = pd.read_csv(imp_p)
        _do("F7", fig_importance, imp, ds, fig_dir, cfg)

    summ_p = run / "cv" / "metrics_summary.csv"
    ci_p = run / "cv" / "metrics_ci.csv"
    if summ_p.exists():
        summary = pd.read_csv(summ_p)
        ci = pd.read_csv(ci_p) if ci_p.exists() else pd.DataFrame()
        _do("F8", fig_compare, ci, summary, task, fig_dir, cfg)

    if skipped:
        info(f"跳过 {len(skipped)} 张图（缺少对应产物）：{skipped}")
    return made
