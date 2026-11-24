# Copyright 2026 The research-ml-modeling Authors
# SPDX-License-Identifier: Apache-2.0
"""stage 4：适用域（AD）建模与有效性验证。

产物
    ad/ad_scores.csv   逐行 AD 打分与越界标记
    ad/ad_eval.json    域内/域外性能对比 + AD 是否真的有效
    ad/ad_model.joblib 可复用的 AD 模型（新样品先判域再预测）
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sml.ad import ad_summary_table, evaluate_ad_with_X, fit_ad  # noqa: E402
from sml.io_data import impute_for_diagnostics  # noqa: E402
from sml.stage_common import build_parser, prepare  # noqa: E402
from sml.utils import info, warn, write_json  # noqa: E402


def run_ad(ctx) -> dict:
    cfg = ctx.cfg
    ds = ctx.ds
    out_dir = ctx.sub("ad")
    if not cfg["ad"].get("enabled", True):
        info("ad.enabled=false，跳过适用域分析")
        return {"skipped": True}

    # AD 是全量描述性分析；缺失值用全量中位数仅用于这里，CV 性能仍由
    # Pipeline 在训练折内插补。两者口径不同，不能互相替代。
    X_ad = impute_for_diagnostics(ds.X)
    ad = fit_ad(X_ad, cfg, groups=ds.groups)
    info(f"适用域模型：PCA {ad.n_comp_} 成分（累计解释 "
         f"{np.sum(ad.pca.explained_variance_ratio_):.1%}），阈值取 {ad.quantile:.0%} 经验分位")

    scores = ad.score(X_ad)
    scores.insert(0, "row_id", ds.row_ids)
    scores.insert(1, "group", ds.groups)
    if ds.batch is not None:
        scores.insert(2, "batch", ds.batch)
    scores.to_csv(out_dir / "ad_scores.csv", index=False, encoding="utf-8")

    summary = ad_summary_table(scores, ad.degenerate_)
    summary.to_csv(out_dir / "ad_summary.csv", index=False, encoding="utf-8")

    result: dict = {"ad": ad.describe(), "flag_rate_any": float(scores["ad_flag"].mean()),
                    "flag_rate_majority": float(scores["ad_majority"].mean()),
                    "methods_table": summary.to_dict(orient="records")}

    # 小样本退化判据：与 audit 阶段同一套公式，避免「audit 说没事、ad 才发现」
    n_groups_ad = int(len(set(np.asarray(ds.groups, dtype=object))))
    p_eff = int(ad.n_comp_)
    h_star = 3.0 * (p_eff + 1) / max(1, n_groups_ad)
    reliable = not (n_groups_ad < 5 * (p_eff + 1) or h_star >= 1.0)
    result["ad_reliable"] = bool(reliable)
    result["ad_reliability"] = {
        "n_groups": n_groups_ad, "n_components": p_eff,
        "leverage_threshold": float(h_star),
        "rule": "n >= 5(p+1) 且 h* < 1 才认为经验阈值可靠",
    }
    if not reliable:
        warn(f"适用域判据退化：n={n_groups_ad}，p={p_eff}，h*={h_star:.3f}。"
             f"AD 结果只能作探索性描述，不得宣称可外推")

    # 独立样品口径（论文用）：scores 是逐行打分，60 样品 × 3 重复测量 = 180 行，
    # 直接报行数会把域外样品数放大到 3 倍。这里同时给出两套数，下游一律取 group 口径。
    g = scores["group"].astype(str).to_numpy()
    n_groups = len(set(g))
    n_flag_any = len(set(g[scores["ad_flag"].to_numpy() == 1]))
    n_flag_major = len(set(g[scores["ad_majority"].to_numpy() == 1]))
    result.update({
        "n_groups": int(n_groups),
        "n_outside_groups_any": int(n_flag_any),
        "n_outside_groups_majority": int(n_flag_major),
        "flag_rate_any_groups": float(n_flag_any / max(1, n_groups)),
        "flag_rate_majority_groups": float(n_flag_major / max(1, n_groups)),
        "group_rule": "样品内任一测量被判越界，即该样品计为域外（保守口径）",
    })

    oof_path = ctx.run_dir / "cv" / "oof_predictions.csv"
    if oof_path.exists():
        oof = pd.read_csv(oof_path)
        per_model = {}
        for model_name, sub in oof.groupby("model"):
            ev = evaluate_ad_with_X(ad, X_ad, sub, ds.row_ids, ds.groups, ds.task)
            per_model[model_name] = ev
        result["per_model"] = per_model
        # 用第一个有结果的模型给出总结论
        conclusion = author_note = ""
        for ev in per_model.values():
            if ev.get("conclusion"):
                conclusion = ev["conclusion"]
                author_note = ev.get("author_note", "")
                break
        result["conclusion"] = conclusion
        result["author_note"] = author_note
        info(f"AD 有效性：{conclusion}")
        verdicts = {str(ev.get("ad_verdict")) for ev in per_model.values()}
        if "reversed" in verdicts:
            warn("适用域方向与预期相反（域外样品的误差反而更小）："
                 "不得用 AD 做样品筛选或外推声明，必须写入 limitations")
        elif any(ev.get("ad_effective") is False for ev in per_model.values()):
            warn("域内外性能无显著差异：报告中不得宣称「模型可安全外推」，需写入 limitations")
    else:
        result["conclusion"] = "缺少 OOF 预测，无法验证 AD 有效性，请先跑 stage_cv"
        result["author_note"] = "先跑 stage_cv 再跑 stage_ad；此状态下不要在正文写任何外推结论。"

    write_json(out_dir / "ad_eval.json", result)
    try:
        import joblib

        joblib.dump(ad, out_dir / "ad_model.joblib", compress=3)
    except Exception as exc:  # noqa: BLE001
        warn(f"AD 模型保存失败（不影响结果）：{exc}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = build_parser("适用域建模与有效性验证")
    args = parser.parse_args(argv)
    ctx = prepare(args)
    run_ad(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
