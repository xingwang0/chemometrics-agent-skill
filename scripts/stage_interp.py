# Copyright 2026 The research-ml-modeling Authors
# SPDX-License-Identifier: Apache-2.0
"""stage 5：可解释性。

顺序就是重要性顺序：
    1. 波段级置换重要性（主结果，回答"哪段谱区真正在起作用"）
    2. PLS VIP（化学计量学读者期待的指标，含 bootstrap 分位区间）
    3. 跨折系数稳定性（回答"选出的波段是否稳定"）
    4. SHAP（仅补充，且强制标注"高维共线下归因不唯一"）

产物：interp/importance.csv、vip.csv、coef_stability.csv、shap_summary.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sml.interp import (  # noqa: E402
    band_permutation_importance, build_band_groups, coef_stability,
    fit_fold_models, shap_values_safe, top_features, vip_scores,
)
from sml.io_data import impute_for_diagnostics  # noqa: E402
from sml.models import build_pipeline, resolve_models  # noqa: E402
from sml.selection import pick_best_model, spec_for_model  # noqa: E402
from sml.stage_common import build_parser, prepare  # noqa: E402
from sml.utils import info, warn, write_json  # noqa: E402


def _best_pls_components(ctx) -> int:
    """取 PLS 在 CV 中最常被选中的 n_components（找不到就用 10）。"""
    import json

    folds_path = ctx.run_dir / "cv" / "fold_results.csv"
    if not folds_path.exists():
        return 10
    df = pd.read_csv(folds_path)
    sub = df[df["model"].str.contains("pls", case=False, na=False)]
    counts: dict[int, int] = {}
    for raw in sub.get("best_params", pd.Series(dtype=str)).astype(str):
        try:
            params = json.loads(raw)
        except json.JSONDecodeError:
            continue
        k = params.get("model__n_components")
        if isinstance(k, (int, float)):
            counts[int(k)] = counts.get(int(k), 0) + 1
    return max(counts, key=counts.get) if counts else 10


def run_interp(ctx) -> dict:
    cfg = ctx.cfg
    ds = ctx.ds
    out_dir = ctx.sub("interp")
    if not cfg["interp"].get("enabled", True):
        info("interp.enabled=false，跳过可解释性分析")
        return {"skipped": True}

    folds_path = ctx.run_dir / "cv" / "fold_results.csv"
    folds = pd.read_csv(folds_path) if folds_path.exists() else None
    model_name, best = pick_best_model(ctx.run_dir, ds.task)
    spec = spec_for_model(resolve_models(cfg, ds.task), model_name, ctx.run_dir)

    band_size = int(cfg["interp"].get("band_size", 25))
    groups = build_band_groups(ds.feature_names, band_size)
    info(f"可解释性：模型 {model_name}，波段分组 {len(groups)} 组（band_size={band_size}）")

    result: dict = {"model": model_name, "band_size": band_size, "n_bands": len(groups),
                    "best_metric": best}

    # 1) 波段置换重要性
    fold_models = fit_fold_models(ctx, spec, folds)
    imp = band_permutation_importance(
        ctx, fold_models, groups,
        n_repeats=int(cfg["interp"].get("perm_repeats", 10)),
        seed=int(cfg["runtime"]["seed"]),
    )
    if not imp.empty:
        imp.to_csv(out_dir / "importance.csv", index=False, encoding="utf-8")
        result["mode"] = str(imp.attrs.get("mode", "test_fold"))
        result["degenerate"] = bool(imp.attrs.get("degenerate", False))
        if result["degenerate"]:
            result["degenerate_note"] = str(imp.attrs.get("note", ""))
            warn("特征级置换重要性全为 0（置换没有产生信息）：本次重要性不可用，"
                 "报告会标注为不可用而不是列出 0.0000")
        else:
            top = imp.head(int(cfg["interp"].get("top_k", 20)))
            result["top_bands"] = top[["label", "delta_mean", "share"]].to_dict(orient="records")
            positive = imp[imp["delta_mean"] > 0]
            result["n_positive_bands"] = int(len(positive))
            info(f"波段置换完成（{'全样本置换' if result['mode'] == 'all_rows' else '测试折置换'}）："
                 "贡献前 3 = "
                 + ", ".join(f"{r['label']}({r['delta_mean']:.4f})" for _, r in top.head(3).iterrows()))
    else:
        warn("波段置换重要性为空（可能是模型或数据结构不支持）")

    # 2) VIP（PLS 是化学计量学基线，始终单独算）
    if cfg["interp"].get("vip", True):
        k = _best_pls_components(ctx)
        try:
            y_vec = np.asarray(ds.targets.get(ctx.target, ds.y))
            # VIP/SHAP 是全量描述性分析，不是 CV 性能估计；缺失值用全量中位数
            # 仅用于这里，CV 内的插补仍严格在训练折内完成。
            Xdiag = impute_for_diagnostics(ds.X)
            vip = vip_scores(Xdiag, y_vec, k)
            # bootstrap 分位区间（按样品重采样），保持与 vip 相同的特征顺序
            rng = np.random.default_rng(int(cfg["runtime"]["seed"]))
            uniq = pd.unique(ds.groups)
            member = {g: np.flatnonzero(ds.groups == g) for g in uniq}
            boots = []
            for _ in range(100):
                picked = rng.integers(0, uniq.size, size=uniq.size)
                idx = np.concatenate([member[uniq[i]] for i in picked])
                try:
                    boots.append(vip_scores(Xdiag[idx], y_vec[idx], k))
                except Exception:  # noqa: BLE001
                    continue
            vip_df = pd.DataFrame({"feature": ds.feature_names, "vip": vip,
                                   "above_1": vip > 1.0})
            if boots:
                B = np.vstack(boots)
                vip_df["vip_q025"] = np.quantile(B, 0.025, axis=0)
                vip_df["vip_q975"] = np.quantile(B, 0.975, axis=0)
                vip_df["stable_above_1"] = vip_df["vip_q025"] > 1.0
            vip_df = vip_df.sort_values("vip", ascending=False).reset_index(drop=True)
            vip_df.to_csv(out_dir / "vip.csv", index=False, encoding="utf-8")
            result["vip"] = {"n_components": k, "n_above_1": int((vip > 1).sum()),
                             "top": vip_df.head(10)[["feature", "vip"]].to_dict(orient="records")}
            # 两种排序法给出不同首位时必须提示：读者只会看到一个「最重要」，
            # 而 VIP 与置换的重要性定义完全不同（VIP 含协方差结构，置换含模型依赖）。
            tops = result.get("top_bands") or []
            if tops and not vip_df.empty:
                top_perm, top_vip = str(tops[0].get("label")), str(vip_df.iloc[0]["feature"])
                if top_perm != top_vip:
                    result["vip_vs_permutation"] = {
                        "permutation_top": top_perm, "vip_top": top_vip,
                        "note": ("VIP 与置换重要性的首位不一致：前者偏向与响应协变的投影方向，"
                                 "后者衡量模型对该列的依赖；论文中应并列报告，不能只取一个"),
                    }
                    warn(f"VIP 首位（{top_vip}）与置换首位（{top_perm}）不一致，已在报告中标注")
            info(f"VIP 完成：{int((vip > 1).sum())} 个特征 VIP>1（PLS {k} 成分）")
        except Exception as exc:  # noqa: BLE001
            warn(f"VIP 计算失败：{type(exc).__name__}: {exc}")

    # 3) 系数稳定性
    if cfg["interp"].get("coef_stability", True):
        stab = coef_stability(fold_models, ds.feature_names)
        if not stab.empty:
            stab.to_csv(out_dir / "coef_stability.csv", index=False, encoding="utf-8")
            result["coef_stability"] = {
                "n_folds": int(stab["n_folds"].iloc[0]),
                "mean_sign_consistency": float(stab["sign_consistency"].mean()),
                "unstable_features": int((stab["sign_consistency"] < 0.8).sum()),
                "note": "符号一致率低说明该特征的作用方向随折变化，不应作为化学结论",
            }
            info(f"系数稳定性：平均符号一致率 {result['coef_stability']['mean_sign_consistency']:.2f}")
        else:
            info("该模型不支持跨折系数稳定性（如树模型/非线性模型），已跳过")

    # 4) SHAP（补充）
    shap_mode = str(cfg["interp"].get("shap", "auto")).lower()
    if shap_mode != "off" and spec.is_tree:
        pipe = build_pipeline(spec, cfg, ds.feature_names, seed=int(cfg["runtime"]["seed"]))
        y = np.asarray(ds.targets.get(ctx.target, ds.y))
        pipe.fit(ds.X, y)
        out = shap_values_safe(pipe, ds.X, seed=int(cfg["runtime"]["seed"]))
        if out is not None:
            vals, Xt = out
            n_feat = vals.shape[1]
            names = (ds.feature_names[:n_feat] if n_feat <= len(ds.feature_names)
                     else [f"pc{i}" for i in range(n_feat)])
            shap_df = pd.DataFrame({
                "feature": names,
                "mean_abs_shap": np.abs(vals).mean(axis=0),
            }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
            shap_df.to_csv(out_dir / "shap_summary.csv", index=False, encoding="utf-8")
            result["shap"] = {
                "n_features": int(n_feat),
                "caveat": "高维强共线下 SHAP 归因不唯一，仅作方向性参考，不可作为主结论",
            }
            info("SHAP 已作为补充结果输出（图上会加注归因不唯一的限制）")

    write_json(out_dir / "interp.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = build_parser("可解释性：波段置换重要性 / VIP / 系数稳定性 / SHAP")
    args = parser.parse_args(argv)
    ctx = prepare(args)
    run_interp(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
