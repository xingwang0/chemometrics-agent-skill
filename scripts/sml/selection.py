"""从 CV 产物里挑"代表模型"及其超参。多个 stage（perm / interp / report）共用。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ConfigError

__all__ = ["pick_best_model", "fixed_params_for", "spec_for_model",
           "load_summary", "targets_in_summary", "significance_vs_runner_up",
           "PRIMARY_BY_TASK"]

PRIMARY_BY_TASK = {"regression": "r2", "classification": "auc"}

# 判定阈值：必须和正文里报的 BH 校正 p 用同一个 α，否则结论与数字互相打脸。
ALPHA = 0.05


def load_summary(run_dir: str | Path) -> pd.DataFrame:
    p = Path(run_dir) / "cv" / "metrics_summary.csv"
    if not p.exists():
        raise ConfigError("E-STAGE-001", f"缺少 {p}，请先跑 stage_cv")
    return pd.read_csv(p)


def targets_in_summary(df: pd.DataFrame) -> list[str]:
    if df.empty or "target" not in df.columns:
        return []
    return [str(t) for t in df["target"].dropna().unique()]


def _basis_aware_score(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """加一列 ``_sel_score``：``metric_basis=pooled`` 的行用池化指标，其余用折均值。

    **所有"挑模型/挑次优"的入口都必须走这里**。旧版 `significance_vs_runner_up`
    自己按 ``{metric}_mean`` 排序，而 `pick_best_model` 是 basis-aware 的 —— 当最优
    模型是 pooled 口径时，那句"显著优于次优"就比错了对象（拿折均值的亚军去比池化
    的冠军）。这条口径只决定"拿谁去比"，不参与任何拟合，所以统一它不会改任何数值。
    """
    df = df.copy()
    col = f"{metric}_mean"
    basis = (df["metric_basis"].astype(str) if "metric_basis" in df.columns
             else pd.Series("fold", index=df.index))
    if metric in df.columns:
        df["_sel_score"] = df[col].astype(float).where(~basis.eq("pooled"), df[metric].astype(float))
    else:
        df["_sel_score"] = df[col].astype(float)
    return df


def pick_best_model(run_dir: str | Path, task: str,
                    target: str | None = None) -> tuple[str, dict[str, Any]]:
    """按主指标（回归 R²、分类 AUC）挑排名第一的模型。

    多目标时**必须**给 ``target``：``metrics_summary`` 是多个目标纵向拼接的表，
    不同目标的指标尺度与难度都不同，全表排序得到的是"跨目标最强"，
    而不是"每个目标各自的最优"。取 ``iloc[0]`` 更糟——那是拼接顺序里的第一行。
    """
    df = load_summary(run_dir)
    metric = PRIMARY_BY_TASK.get(task, "r2")
    col = f"{metric}_mean"
    if col not in df:
        raise ConfigError("E-STAGE-002", f"CV 汇总缺少 {col}")

    names = targets_in_summary(df)
    if len(names) > 1:
        if target is None:
            raise ConfigError(
                "E-STAGE-004",
                f"CV 结果含 {len(names)} 个目标（{', '.join(names)}），无法判定「最优模型」指的是哪个目标。\n"
                f"  - 每个目标的指标尺度不同，全表排序只会选出「跨目标最强」的模型，不是各目标的最优；\n"
                f"    报告里的 CI 也会被后一个目标覆盖前一个。\n"
                f"  修法：用 run_pipeline（多目标会自动按目标拆成 "
                f"<out>/targets/<目标>/ 独立子目录，各自出报告），\n"
                f"        或显式指定单目标：--set task.target=[\"{names[0]}\"]")
        df = df[df["target"].astype(str) == str(target)]
        if df.empty:
            raise ConfigError("E-STAGE-005",
                              f"目标 {target!r} 不在 CV 结果里（现有：{', '.join(names)}）")

    ascending = metric not in ("r2", "auc")   # R²/AUC 越大越好
    # 口径：stage_cv 在「有效折 < 3」时把这些行标成 metric_basis=pooled，
    # 这些行的 *_mean 只是极少数折的平均（退化折被 nanmean 跳过），不能用来选模。
    df = _basis_aware_score(df, metric)
    best = df.sort_values("_sel_score", ascending=ascending).iloc[0]
    basis_best = "pooled" if str(best.get("metric_basis", "fold")) == "pooled" else "fold"
    prep_note = best.get("prep_mix") if "prep_mix" in df.columns else best.get("prep")
    best_info: dict[str, Any] = {metric: float(best["_sel_score"]), "prep": prep_note,
                                 "basis": basis_best,
                                 "target": None if "target" not in df.columns else str(best["target"])}
    if basis_best == "pooled":
        best_info[f"{metric}_fold_mean"] = float(best.get(col, float("nan")))
        for k in ("n_folds_ok", "n_folds_degenerate"):
            if k in df.columns:
                best_info[k] = int(best[k])
    for extra in (f"{metric}_sd", "rmse_mean", "rpd_mean"):
        if extra in df:
            best_info[extra] = float(best[extra])
    return str(best["model"]), best_info


def significance_vs_runner_up(run_dir: str | Path, task: str, best_model: str,
                              target: str | None = None,
                              legacy_runner: bool = False) -> dict[str, Any]:
    """最优模型与**同目标**次优模型是否显著不同（配对 bootstrap + BH 校正）。

    没有这一条，报告就会无条件写「最优模型 X」：当 X 与 Y 的差值 CI 跨 0 时，
    这句话在论文里等于给出了数据不支持的排序结论。

    ``legacy_runner=True`` 保留旧口径（只按 ``*_mean`` 挑次优），仅用于对照与回滚；
    默认走 `_basis_aware_score`，与 `pick_best_model` 同一把尺子。
    """
    try:
        summary = load_summary(run_dir)
    except ConfigError:
        return {"available": False, "reason": "缺少 CV 汇总"}
    if target is not None and "target" in summary.columns:
        summary = summary[summary["target"].astype(str) == str(target)]
    metric = PRIMARY_BY_TASK.get(task, "r2")
    col = f"{metric}_mean"
    if col not in summary.columns or summary.empty:
        return {"available": False, "reason": f"缺少 {col}"}

    others = summary[summary["model"].astype(str) != str(best_model)]
    if legacy_runner:
        others = others[np.isfinite(pd.to_numeric(others[col], errors="coerce"))]
        if others.empty:
            return {"available": False, "reason": "只有一个可用模型，无可比对象"}
        runner = str(others.sort_values(col, ascending=False).iloc[0]["model"])
    else:
        # 与选模同一把尺子：pooled 行用池化指标，其余用折均值
        ranked = _basis_aware_score(others, metric)
        ranked = ranked[np.isfinite(pd.to_numeric(ranked["_sel_score"], errors="coerce"))]
        if ranked.empty:
            return {"available": False, "reason": "只有一个可用模型，无可比对象"}
        ascending = metric not in ("r2", "auc")
        runner = str(ranked.sort_values("_sel_score", ascending=ascending).iloc[0]["model"])

    p = Path(run_dir) / "cv" / "comparison.csv"
    if not p.exists():
        return {"available": False, "runner_up": runner, "reason": "缺少 cv/comparison.csv"}
    comp = pd.read_csv(p)
    if target is not None and "target" in comp.columns:
        comp = comp[comp["target"].astype(str) == str(target)]
    sub = comp[(comp["metric"] == metric)
               & comp["model_a"].astype(str).isin([str(best_model), runner])
               & comp["model_b"].astype(str).isin([str(best_model), runner])]
    if sub.empty:
        return {"available": False, "runner_up": runner,
                "reason": f"comparison.csv 里没有 {best_model} vs {runner} 的 {metric} 行"
                          f"（检查 evaluation.metrics 是否包含 {metric}）"}
    row = sub.iloc[0]
    # delta 的方向统一成「best − runner」：metric 越大越好时 >0 表示 best 更好
    sign = 1.0 if str(row["model_a"]) == str(best_model) else -1.0
    delta = sign * float(row["delta"])
    lo, hi = sorted((sign * float(row["ci_low"]), sign * float(row["ci_high"])))
    p_bh = row.get("p_BH", row.get("p_raw", np.nan))
    p_raw = row.get("p_raw", np.nan)
    # comparison.csv 的 significant 只按「95% CI 是否跨 0」判定（等价单次比较 α=0.05），
    # 而正文同时报 BH 校正 p。家族 m 很大时 BH 会把 p 抬到 α 以上，
    # 于是会出现「显著优于次优 … BH 校正 p = 0.10」这种自相矛盾的句子。
    # 统一按 BH 结果定论，并把两套判据的分歧单独标出，供报告如实说明。
    sig_ci = bool(row.get("significant", (lo > 0 or hi < 0)))
    p_bh_ok = True if pd.isna(p_bh) else bool(float(p_bh) < ALPHA)
    significant = bool(sig_ci and p_bh_ok)
    bh_conflict = bool(sig_ci and not p_bh_ok)
    better = best_model if (significant and delta > 0) else (
        runner if significant else "无显著差异")
    return {
        "available": True, "runner_up": runner, "metric": metric,
        "delta": float(delta), "ci_low": float(lo), "ci_high": float(hi),
        "p_raw": None if pd.isna(p_raw) else float(p_raw),
        "p_bh": None if pd.isna(p_bh) else float(p_bh),
        "significant": significant, "significant_by_ci": sig_ci,
        "bh_conflict": bh_conflict, "alpha": ALPHA,
        "m_family": None if pd.isna(row.get("m_family", np.nan)) else int(row.get("m_family")),
        "better": better,
        "n_pairs": int(row.get("n_pairs", 0) or 0),
    }


def spec_for_model(specs, model_name: str, run_dir: str | Path | None = None):
    """按名字取 ModelSpec；取不到时给出可执行的修复建议，而不是 KeyError。

    典型触发场景：单独重跑下游 stage（interp / perm / ad）时，用了
    ``--config + --out`` 而漏掉当初的 ``--set models=...``——配置回退成预设默认值，
    CV 结果里的最优模型（例如 pcr）就不在当前 specs 里了。
    """
    by_name = {s.name: s for s in specs}
    if model_name in by_name:
        return by_name[model_name]
    raise ConfigError(
        "E-STAGE-003",
        f"CV 结果里的最优模型 {model_name!r} 不在当前配置的 models 里（当前：{sorted(by_name)}）。\n"
        f"  最常见原因：单独重跑下游 stage 时漏掉了当初的 --set 覆盖，配置回退成预设默认值。\n"
        f"  修法：改用 --run-dir {run_dir or '<run 目录>'} 复用 config.resolved.json，"
        f"或补齐同样的 --set。")


def fixed_params_for(run_dir: str | Path, model: str) -> dict[str, Any]:
    """取该模型在 CV 中最常被选中的那组超参（用于置换检验的固定配置）。"""
    p = Path(run_dir) / "cv" / "fold_results.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    sub = df[df["model"] == model]
    if "status" in sub:
        sub = sub[sub["status"].astype(str).isin(["ok", "degenerate"])]
    if sub.empty or "best_params" not in sub:
        return {}
    counts: dict[str, int] = {}
    for raw in sub["best_params"].astype(str):
        counts[raw] = counts.get(raw, 0) + 1
    top = max(counts, key=counts.get)
    try:
        return json.loads(top)
    except json.JSONDecodeError:
        return {}
