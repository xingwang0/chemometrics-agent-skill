# Copyright 2026 The research-ml-modeling Authors
# SPDX-License-Identifier: Apache-2.0
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

from . import present
from .utils import as_path, configure_matplotlib, ensure_dir, info, warn, write_json

__all__ = ["render_all", "PALETTE", "FIGURE_INDEX", "figure_qc", "QC_ISSUES"]

STYLE_DIR = as_path(__file__).resolve().parent.parent.parent / "assets" / "style"

# 兜底配色与 ``assets/style/palette.json`` 必须逐项一致。只写 8 色时第 9、10 个模型
# 会静默回绕到前两色，图上两个模型同色而无任何提示——所以两边都列满 10 色。
_FALLBACK_PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9",
                     "#D55E00", "#F0E442", "#666666", "#44AA99", "#AA4499"]

_FALLBACK_SPECS = {
    "figsize": {"F1": [3.6, 3.4], "F2": [3.6, 3.2], "F3": [3.4, 3.2], "F4": [3.8, 3.2],
                "F5": [3.6, 3.2], "F6": [3.8, 3.0], "F7": [4.8, 4.2], "F8": [3.8, 2.4]},
    "fontsize": {"annotation": 6.8, "annotation_small": 6.4, "tick": 7.5,
                 "tick_small": 7.0, "title": 9.0, "legend": 7.0},
}


def _read_style_json(name: str, default: dict) -> dict:
    p = STYLE_DIR / name
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8")) or default
    except Exception as exc:  # noqa: BLE001
        warn(f"样式文件 {name} 解析失败，已用内置默认值：{type(exc).__name__}")
        return default


_PALETTE_JSON = _read_style_json("palette.json", {"colors": _FALLBACK_PALETTE})
PALETTE = list(_PALETTE_JSON.get("colors") or _FALLBACK_PALETTE)
if len(PALETTE) < len(_FALLBACK_PALETTE):
    warn(f"palette.json 只有 {len(PALETTE)} 色，回退到内置 10 色，避免多模型同色")
    PALETTE = list(_FALLBACK_PALETTE)
SEMANTIC = dict(_PALETTE_JSON.get("semantic") or {})

FS = _read_style_json("figure_specs.json", _FALLBACK_SPECS)


def _fsize(code: str, fallback: tuple[float, float]) -> tuple[float, float]:
    raw = (FS.get("figsize") or {}).get(code)
    try:
        return (float(raw[0]), float(raw[1])) if raw else fallback
    except (TypeError, ValueError, IndexError):
        return fallback


def _font(key: str, fallback: float) -> float:
    try:
        return float((FS.get("fontsize") or {}).get(key, fallback))
    except (TypeError, ValueError):
        return fallback


#: 图件完整性检查的违规记录。``render_all`` 结束时落盘为 ``figures/figure_qc.json``，
#: 自检断言该文件为空——"图有没有误导性"因此从口头约定变成可失败的断言。
QC_ISSUES: dict[str, list[str]] = {}

_BAD_COLORMAPS = {"jet", "rainbow", "gist_rainbow", "gist_ncar", "hsv", "gist_stern"}

FIGURE_INDEX = {
    "F1": "OOF 性能图（回归=预测-实测散点；分类=ROC，多分类为逐类 OVR）；统计量为 OOF 池化口径，与表 T2 的 repeat 均值口径不可混引",
    "F2": "残差图（仅回归）；偏差/±1.96SD 亦为 OOF 池化口径",
    "F3": "交叉验证分布（逐折；黑菱形为折均值，与 T2 的重复间均值略有差异）",
    "F4": ("Williams plot（适用域；纵轴为标准化残差，未按杠杆校正，仅用于诊断，"
           "不单独作为域外判定依据）"),
    "F5": "校准 / 一致性（回归=Bland-Altman；分类=可靠性曲线，多分类为逐类 OVR；OOF 池化口径）",
    "F6": "Y-randomization 零分布",
    "F7": "重要性-波数映射",
    "F8": "模型比较（95% CI；† = 该行折均值不可引用，见图内标注）",
}


def _unit(cfg: dict):
    """目标变量单位。多目标 fan-out 时 cfg 里只有一个目标，取它对应的键。"""
    from .io_data import resolve_target_field

    targets = (cfg.get("task") or {}).get("target") or []
    single = str(targets[0]) if len(targets) == 1 else None
    return resolve_target_field(cfg, "target_unit", single)


def _ax_label(cfg: dict, kind: str) -> str:
    """轴标签：有单位就带上，没有就显式写"单位未指定"。

    图里省掉单位是最常见的一类误导——读者无法判断 RMSE=0.56 是 % 还是 mg/kg，
    也无法判断该模型能不能用。宁可图丑一点，也不能让它读起来没有量纲。
    """
    return present.axis_label(kind, _unit(cfg), "mpl")


def _numeric_limits(ax, axis: str) -> tuple[float, float] | None:
    vals: list[float] = []
    for ln in ax.lines:
        arr = ln.get_ydata() if axis == "y" else ln.get_xdata()
        try:
            vals.extend(np.asarray(arr, dtype=float).ravel().tolist())
        except (TypeError, ValueError):
            continue
    for col in ax.collections:
        try:
            off = np.asarray(col.get_offsets(), dtype=float)
        except (TypeError, ValueError):
            continue
        if off.size:
            vals.extend((off[:, 1] if axis == "y" else off[:, 0]).tolist())
    from matplotlib.patches import Rectangle
    for patch in ax.patches:
        if isinstance(patch, Rectangle):
            if axis == "y":
                vals.extend([patch.get_y(), patch.get_y() + patch.get_height()])
            else:
                vals.extend([patch.get_x(), patch.get_x() + patch.get_width()])
    vals = [v for v in vals if np.isfinite(v)]
    return (min(vals), max(vals)) if vals else None


def _needs_axis_label(ax, axis: str) -> bool:
    """类别轴（模型名、特征名）不必有轴标题；数值轴必须有。"""
    labels = ax.get_xticklabels() if axis == "x" else ax.get_yticklabels()
    texts = [t.get_text().strip() for t in labels if t.get_text().strip()]
    if not texts:
        return True
    for t in texts:
        try:
            float(t)
        except ValueError:
            return False
    return True


def figure_qc(fig, code: str) -> list[str]:
    """图件完整性检查（Q1–Q5）。返回违规说明，空列表表示通过。

    这些规则不是审美偏好，每一条都对应一种"让数据看起来更好"的常用手法：
    双 y 轴、截断柱状图的纵轴、彩虹色标、只画箱体不画散点、只给点估计不给区间。
    """
    issues: list[str] = []
    axes = list(fig.axes)

    # Q1 叠加坐标轴（twinx/twiny）——两个刻度轴可以随意缩放，让任意两条曲线"看起来相关"
    pos = [tuple(round(v, 6) for v in a.get_position().bounds) for a in axes]
    if len(pos) != len(set(pos)):
        issues.append("Q1 存在叠加坐标轴（twinx/twiny）：双 y 轴可让任意两条曲线看起来相关")

    # Q3 非感知均匀色标
    for a in axes:
        for im in a.get_images():
            name = str(getattr(im.get_cmap(), "name", ""))
            if name in _BAD_COLORMAPS:
                issues.append(f"Q3 使用了非感知均匀色标 {name}（应改用 viridis 一类）")

    for a in axes:
        from matplotlib.patches import Rectangle
        bars = [p for p in a.patches
                if isinstance(p, Rectangle) and abs(float(p.get_height())) > 0]
        lim = a.get_ylim()
        rng = _numeric_limits(a, "y")
        # Q2 只针对柱状图：柱长编码大小，截断纵轴会直接放大差异。
        # 箱线图与散点图按数据自动缩放是正常做法，不在此列。
        if bars and rng is not None and lim[0] > 0 and rng[0] < lim[0] * (1 - 1e-9):
            if not getattr(a, "_sml_zero_declared", False):
                issues.append(
                    f"Q2 柱状图 y 轴被截断（显示 {lim[0]:.3g} 起，数据下界 {rng[0]:.3g}）"
                    "且未声明；柱长编码大小，截断会放大差异")
        # Q4 箱线图必须叠散点：只画箱体看不到折间分布，也看不到失败折
        if getattr(a, "_sml_boxplot", False) and not a.collections:
            issues.append("Q4 箱线图未叠加散点：只画箱体会掩盖折间分布与失败折")
        # Q5 数值轴缺轴标题。共享 x 轴的分面里只有最下面一格需要 x 标签
        # （F7 的上下两幅共享波数轴），把上面那格也算成"缺标签"是误报。
        shares_x_below = any(
            other is not a and a.get_shared_x_axes().joined(a, other)
            and other.get_position().y0 < a.get_position().y0
            for other in axes)
        for axis in ("x", "y"):
            if axis == "x" and shares_x_below:
                continue
            label = (a.get_xlabel() if axis == "x" else a.get_ylabel()).strip()
            if not label and _needs_axis_label(a, axis):
                issues.append(f"Q5 {axis} 轴缺标签：读者无法判断该轴是什么量")

    if code == "F8":
        from matplotlib.container import ErrorbarContainer
        if not any(isinstance(c, ErrorbarContainer) for a in axes for c in a.containers):
            issues.append("Q5 F8 缺少误差线：只画点估计无法比较模型")
    return issues


def _plt(cfg: dict):
    style = STYLE_DIR / "paper.mplstyle"
    return configure_matplotlib(dpi=int(cfg.get("report", {}).get("dpi", 300)), style_path=style)


def _save(fig, out_dir: Path, code: str, cfg: dict) -> list[Path]:
    issues = figure_qc(fig, code)
    if issues:
        QC_ISSUES[code] = issues
        warn(f"{code} 图件完整性检查未通过：{'；'.join(issues)}")
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
    fig, ax = plt.subplots(figsize=_fsize("F1", (3.6, 3.4)))
    if task == "regression":
        y, p = sub["y_true"].to_numpy(), sub["y_pred"].to_numpy()
        ax.scatter(y, p, s=14, alpha=0.65, edgecolors="none", color=_color(0), label="OOF 预测")
        lo = float(min(y.min(), p.min()))
        hi = float(max(y.max(), p.max()))
        pad = 0.04 * (hi - lo if hi > lo else 1.0)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ls="--", lw=1, color="#444444", label="1:1 线")
        from .metrics import regression_metrics
        m = regression_metrics(y, p)
        u = present.unit_suffix(_unit(cfg), "mpl")
        ax.text(0.04, 0.96,
                f"$R^2$ = {present.fmt('r2', m['r2'])}\n"
                f"RMSE = {present.fmt('rmse', m['rmse'])} ({u})\n"
                f"RPD = {present.fmt('rpd', m['rpd'])}\n"
                f"斜率 = {present.fmt('slope', m['slope'])}\n"
                f"（OOF 池化，见表 T2 口径说明）",
                transform=ax.transAxes, va="top", fontsize=_font("annotation", 6.8),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbbbbb", alpha=0.9))
        ax.set_xlabel(_ax_label(cfg, "observed"))
        ax.set_ylabel(_ax_label(cfg, "predicted"))
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
                        label=f"{_class_name(c, label_map)}: AUROC={present.fmt('auc', auc_c)}")
                aucs.append(auc_c)
            if not aucs:
                warn(f"{model} 的 OOF 里没有同时含正负样本的类别，跳过 ROC")
                return []
            ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#888888")
            ax.text(0.04, 0.96, f"宏平均 AUROC = {present.fmt('auc', float(np.mean(aucs)))}\n（OOF 池化，OVR）",
                    transform=ax.transAxes, va="top", fontsize=_font("annotation", 6.8),
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bbbbbb", alpha=0.9))
            ax.legend(loc="lower right", fontsize=_font("annotation_small", 6.4), frameon=False)
        else:
            score = (pd.to_numeric(sub["proba"], errors="coerce").to_numpy(dtype=float)
                     if "proba" in sub else sub["y_pred"].to_numpy(dtype=float))
            if not np.isfinite(score).all():
                score = sub["y_pred"].to_numpy(dtype=float)
            if len(np.unique(y)) > 1:
                fpr, tpr, _ = roc_curve(y, score)
                ax.plot(fpr, tpr, lw=1.6, color=_color(0),
                        label=f"AUROC = {present.fmt('auc', roc_auc_score(y, score))}（OOF 池化）")
                ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#888888")
                ax.legend(loc="lower right", fontsize=_font("legend", 7.2), frameon=False)
        ax.set_xlabel("假阳性率")
        ax.set_ylabel("真阳性率")
    ax.set_title(f"OOF 性能（{model}）", fontsize=_font("title", 9))
    ax.tick_params(labelsize=_font("tick", 7.5))
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
    fig, ax = plt.subplots(figsize=_fsize("F2", (3.6, 3.2)))
    ax.scatter(y, resid, s=14, alpha=0.65, edgecolors="none", color=_color(2))
    ax.axhline(0, color="#333333", lw=1)
    for k, ls in ((1.96, "--"), (-1.96, "--")):
        ax.axhline(k * sd, color="#D55E00", lw=1, ls=ls)
    u = present.unit_suffix(_unit(cfg), "mpl")
    ax.text(0.02, 0.98,
            f"±1.96SD = ±{present.fmt('rmse', 1.96 * sd)} ({u})\n"
            f"偏差 = {present.fmt('ba_bias', float(resid.mean()))}（预测−实测）\n（OOF 池化）",
            transform=ax.transAxes, va="top", fontsize=_font("legend", 7.0),
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
    ax.set_xlabel(_ax_label(cfg, "observed"))
    ax.set_ylabel(f"残差（预测−实测；{present.unit_suffix(_unit(cfg), 'mpl')}）")
    ax.set_title("残差图（检查系统偏差与异方差）", fontsize=_font("title", 9))
    ax.tick_params(labelsize=_font("tick", 7.5))
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
    fig, ax = plt.subplots(figsize=(max(_fsize("F3", (3.4, 3.2))[0], 0.75 * len(models) + 2.2),
                                   _fsize("F3", (3.4, 3.2))[1]))
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
    ax._sml_boxplot = True
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
    ax.set_ylabel(present.LABELS_ZH.get(metric, metric))
    # 各模型的有效折数可能不同（退化折被剔除），标题里取最大值并说明，别把第一个模型的折数当全局
    n_folds = max(len(d) for d in data) if data else 0
    ax.set_title(f"交叉验证稳定性（最多 {n_folds} 个折）\n黑菱形=均值", fontsize=_font("title", 9))
    ax.tick_params(labelsize=_font("tick", 7.5))
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
    fig, ax = plt.subplots(figsize=_fsize("F4", (3.8, 3.2)))
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
    # h* 必须连同"用的哪个 n"一起标出来：逐行打分时 n 是**拟合行数**（每个谱图一个点），
    # 不是独立样品数，也不是图上散点的个数——OOF 会被 merge 成 repeat×行 份，
    # 拿 len(sc) 当 n 会把 180 行说成 540 行，正好又是一次"两个 n 混用"。
    n_fit = int(sc["row_id"].nunique()) if "row_id" in sc else len(sc)
    ax.text(h_thr, ax.get_ylim()[1],
            f" h* = {present.fmt('leverage', h_thr)}（3(k+1)/{n_fit}）",
            va="top", fontsize=_font("legend", 7), color="#D55E00")
    ax.set_xlabel("Leverage（h）")
    ax.set_ylabel(ylabel)
    # 纵轴是「标准化残差」而非真正的学生化残差（未按 √(1−h) 校正），高杠杆点的残差
    # 因而被系统性低估。这一轮不改统计定义（属方法学变更），但必须写在标题里，
    # 否则读者会把它当成标准 Williams plot 的结论。
    # 本次判定是否真的用了 leverage：smoke 预设只有 T²+kNN，图上的 h* 仍会画出来，
    # 但那条线不参与判域。不说明等于把"诊断参考线"冒充"判定边界"。
    lev_on = "leverage" in ((cfg.get("ad") or {}).get("methods") or [])
    title = "Williams plot（纵轴未按杠杆校正，仅作诊断）\n右上/右下 = 域外样本"
    if not lev_on:
        title += "\n（leverage 未参与本次判定，h* 仅为参考线）"
    ax.set_title(title, fontsize=_font("title", 9))
    ax.tick_params(labelsize=_font("tick", 7.5))
    fig.tight_layout()
    return _save(fig, out_dir, "F4_williams", cfg)


def fig_calibration(oof: pd.DataFrame, model: str, task: str, out_dir: Path, cfg: dict,
                    label_map: dict | None = None) -> list[Path]:
    plt = _plt(cfg)
    sub = oof[oof["model"] == model]
    if sub.empty:
        return []
    fig, ax = plt.subplots(figsize=_fsize("F5", (3.6, 3.2)))
    if task == "regression":
        from .calib import bland_altman
        y, p = sub["y_true"].to_numpy(), sub["y_pred"].to_numpy()
        ba = bland_altman(y, p)
        diff = p - y
        mean_val = (p + y) / 2
        ax.scatter(mean_val, diff, s=14, alpha=0.65, edgecolors="none", color=_color(3))
        bias_txt = present.fmt("ba_bias", ba["bias"])
        if float(ba["bias"]) > 0:
            bias_txt = "+" + bias_txt
        ax.axhline(ba["bias"], color="#0072B2", lw=1, label=f"偏差 {bias_txt}（预测−实测）")
        ax.axhline(ba["loa_low"], ls="--", lw=1, color="#D55E00")
        ax.axhline(ba["loa_high"], ls="--", lw=1, color="#D55E00")
        ax.fill_between([mean_val.min(), mean_val.max()], ba["loa_low"], ba["loa_high"],
                        color="#D55E00", alpha=0.08)
        u = present.unit_suffix(_unit(cfg), "mpl")
        ax.set_xlabel(f"均值（实测, 预测）({u})")
        ax.set_ylabel(f"预测 − 实测 ({u})")
        ax.legend(fontsize=_font("legend", 7), frameon=False, loc="upper right")
        ax.set_title("Bland-Altman 一致性分析", fontsize=_font("title", 9))
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
            ax.text(0.04, 0.96,
                    f"宏平均 ECE = {present.fmt('ece', float(np.mean(eces)))}\n（OOF 池化，OVR）",
                    transform=ax.transAxes, va="top", fontsize=_font("annotation", 6.8),
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bbbbbb", alpha=0.9))
            ax.legend(fontsize=_font("annotation_small", 6.4), frameon=False, loc="lower right")
            ax.set_xlabel("预测概率（该类）")
            ax.set_ylabel("实测该类占比")
            ax.set_title("校准曲线（逐类 OVR）", fontsize=_font("title", 9))
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
                ax.text(0.04, 0.96, f"ECE = {present.fmt('ece', ece)}\n（OOF 池化）",
                        transform=ax.transAxes, va="top", fontsize=_font("legend", 7.0))
                ax.legend(fontsize=_font("legend", 7), frameon=False, loc="lower right")
            ax.set_xlabel("预测概率")
            ax.set_ylabel("实测阳性率")
            ax.set_title("校准曲线", fontsize=_font("title", 9))
    ax.tick_params(labelsize=_font("tick", 7.5))
    fig.tight_layout()
    return _save(fig, out_dir, "F5_calibration", cfg)


def fig_permutation(perm: dict, out_dir: Path, cfg: dict) -> list[Path]:
    vals = perm.get("null_values") or []
    if not vals:
        return []
    plt = _plt(cfg)
    fig, ax = plt.subplots(figsize=_fsize("F6", (3.8, 3.0)))
    ax.hist(vals, bins=min(30, max(8, len(vals) // 5)), color=_color(4), alpha=0.75,
            edgecolor="white", lw=0.5, label="置换零分布")
    obs = perm.get("observed")
    if obs is not None and np.isfinite(obs):
        ax.axvline(obs, color="#D55E00", lw=1.6,
                   label=f"观测值 {present.fmt(str(perm.get('metric') or 'r2'), obs)}")
    ax.set_xlabel(present.LABELS_ZH.get(str(perm.get("metric")), str(perm.get("metric"))))
    ax.set_ylabel("频数")
    # p 值用 <0.001 的写法：写 0.0000 会被读成"p 精确等于 0"，而置换检验的 p
    # 有硬下限 1/(1+n_perm)，根本不可能到 0。
    ax.set_title(f"Y-randomization：p = {present.fmt_p(perm.get('p_value'))}\n"
                 f"（n_perm={perm.get('n_perm')}）", fontsize=_font("title", 9))
    ax.legend(fontsize=_font("legend", 7), frameon=False, loc="upper left")
    ax.tick_params(labelsize=_font("tick", 7.5))
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
        fig, ax = plt.subplots(figsize=(_fsize("F7", (4.8, 4.2))[0],
                                       max(2.6, 0.42 * len(bands) + 1.4)))
        ax.bar(xpos, y, color=_color(0), alpha=0.85, width=0.8)
        ax.axhline(0, color="#888888", lw=0.8)
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_xlabel("特征")
        ax.set_ylabel("ΔR²（置换后下降）")
        ax.set_title("特征级置换重要性：打乱该特征后的性能损失", fontsize=_font("title", 9))
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=_font("tick", 7.5))
        fig.tight_layout()
        return _save(fig, out_dir, "F7_importance_map", cfg)

    fig, axes = plt.subplots(2, 1, figsize=_fsize("F7", (4.8, 4.2)), sharex=True,
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
        xlabel = present.axis_label(
            "wavenumber", ((cfg.get("data") or {}).get("wavelength_axis") or {}).get("unit") or "cm-1",
            "mpl")
        invert = True
    else:
        x = bands["band_index"].to_numpy(dtype=float)
        y = bands["delta_mean"].to_numpy(dtype=float)
        ax.bar(x, y, color=_color(0), alpha=0.85, width=0.85)
        xlabel = "波段序号"
        invert = False

    ax.set_ylabel("ΔR²（置换后下降）")
    ax.set_title("波段级置换重要性：打乱整段波长后的性能损失", fontsize=_font("title", 9))
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=_font("tick", 7.5))

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
    ax2.tick_params(labelsize=_font("tick_small", 7.0))
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
    fig, ax = plt.subplots(figsize=(_fsize("F8", (3.8, 2.4))[0],
                                   max(_fsize("F8", (3.8, 2.4))[1], 0.42 * len(models) + 1.2)))
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
    # 有效折不足的模型，其 *_mean 只来自极少数折（留一法下就是 1 个折）。
    # T2 里标 † 提示"不可引用"，图上却并排画出来而不标，等于把不可比的数画成可比的。
    pooled = ([str(b) == "pooled" for b in df["metric_basis"]]
              if "metric_basis" in df else [False] * len(models))
    ylabels = [f"{m} †" if p else m for m, p in zip(models, pooled)]
    ax.set_yticks(ypos)
    ax.set_yticklabels(ylabels, fontsize=8)
    if metric == "rmse":
        ax.set_xlabel(f"RMSE（越小越好；{present.unit_suffix(_unit(cfg), 'mpl')}）")
    else:
        ax.set_xlabel("AUROC（越大越好）")
    title = "模型比较\n点=repeat均值，细线=±sd，粗线=95% CI"
    if any(pooled):
        title += "\n† 该行折均值不可引用，须改看池化值（表 T2）"
    ax.set_title(title, fontsize=_font("title", 9))
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=_font("tick", 7.5))
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
    QC_ISSUES.clear()

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
    # 图件 QC 落盘：自检读这个文件断言为空。放在这里而不是 _save 里，是为了一次写全，
    # 避免每张图都重写一遍文件。
    write_json(fig_dir / "figure_qc.json", QC_ISSUES)
    if QC_ISSUES:
        warn(f"{len(QC_ISSUES)} 张图未通过完整性检查（详见 figures/figure_qc.json）："
             f"{sorted(QC_ISSUES)}")
    return made
