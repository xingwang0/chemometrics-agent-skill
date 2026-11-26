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

    # 小样本退化判据：这里的 n 是**独立样品数**——问的是"够不够撑起一个经验分位阈值"。
    # 它与逐行打分用的 h*（必须按拟合行数算，因为帽子矩阵建立在行上）不是同一个量。
    # 两个都给出来，并各带一行口径说明：同一个 run 里出现两个不相等的 h* 而不解释，
    # 读者只会认为其中一个算错了。
    diag = dict(getattr(ad, "diag_", {}) or {})
    n_groups_ad = int(diag.get("n_groups") or len(set(np.asarray(ds.groups, dtype=object))))
    n_rows_ad = int(diag.get("n_rows") or ad.n_train_)
    p_eff = int(ad.n_comp_)
    h_rows = float(diag.get("h_star_rows") or 3.0 * (p_eff + 1) / max(1, n_rows_ad))
    h_grp = float(diag.get("h_star_groups") or 3.0 * (p_eff + 1) / max(1, n_groups_ad))
    # 真正判域的那条线是 max(3(k+1)/n, 训练集经验分位)，不是理论线本身。
    # 旧实现把 h_star_rows 直接当 leverage_threshold 落盘，注释还写"与 ad_scores.csv 同值"，
    # 于是在 q95 主导时（经验分位 > 理论线）两个产物报出不同的阈值，读者会以为其中一个算错。
    lev_thr_real = float(diag.get("leverage_threshold", h_rows))
    lev_q95 = float(diag.get("leverage_q95", float("nan")))
    lev_rule = ("经验分位主导" if np.isfinite(lev_q95) and lev_q95 >= h_rows
                else "理论线 3(k+1)/n 主导") if np.isfinite(lev_q95) else "未知"
    reliable = not (n_groups_ad < 5 * (p_eff + 1) or h_grp >= 1.0)
    result["ad_reliable"] = bool(reliable)
    result["ad_reliability"] = {
        "n_groups": n_groups_ad, "n_rows": n_rows_ad, "n_components": p_eff,
        # 与 ad_scores.csv 的 leverage_thr 同值——这才是真正用来判域外的那条线
        "leverage_threshold": lev_thr_real,
        # 两个分量的原值都留档，否则读者无法判断这条线是怎么来的
        "leverage_threshold_theory": h_rows,
        "leverage_q95": lev_q95,
        "leverage_threshold_rule": lev_rule,
        "leverage_threshold_groups": h_grp,
        "leverage_max": float(diag.get("leverage_max", float("nan"))),
        "leverage_inert": bool(diag.get("leverage_inert", False)),
        "leverage_active": bool(diag.get("leverage_active", "leverage" in (ad.active_ or []))),
        "rule": ("n_groups >= 5(p+1) 且 3(p+1)/n_groups < 1 才认为经验阈值可靠；"
                 "leverage_threshold = max(3(k+1)/n_rows, 训练集 95% 分位)，"
                 "与 ad_scores.csv 的 leverage_thr 同值；leverage_threshold_theory 是其中的"
                 "理论线，leverage_threshold_groups 只用于本可靠性判断，不参与判域"),
    }
    if not reliable:
        warn(f"适用域判据退化：独立样品 n={n_groups_ad}，p={p_eff}，"
             f"h*(样品口径)={h_grp:.3f}。AD 结果只能作探索性描述，不得宣称可外推")
    # 只有 leverage 真的在投票集合里，"零触发"才是个值得说的事——
    # 否则它本来就没参与判定，提一句只会让读者以为它影响过结果。
    if result["ad_reliability"]["leverage_inert"] and result["ad_reliability"]["leverage_active"]:
        info(f"杠杆法本次零触发：数据最大杠杆 {diag.get('leverage_max', float('nan')):.4f} "
             f"≤ 阈值 {lev_thr_real:.4f}（{lev_rule}）。该法仍计入多数票分母，"
             f"但不会单独判出域外样品")

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
        verdicts = [str(ev.get("ad_verdict")) for ev in per_model.values()
                    if ev.get("ad_verdict")]
        # AD 判据本身是「样品级」的，但有效性检验要逐模型做。各模型结论不一致时，
        # 只报第一个模型会掩盖分歧（真实数据上出现过：2/5 显著有效、2/5 无差异、
        # 1/5 口径冲突）。这里直接给"按模型分组"的总结论，明细仍看 T5——
        # 不能把首个模型的结论拼在后面，那会让整体读起来像"AD 有效"。
        _VLAB = {"effective": "有效", "no_difference": "无显著差异",
                 "reversed": "方向与预期相反", "conflict": "口径冲突（秩与均值方向相反）"}
        n_eff = sum(v == "effective" for v in verdicts)
        if verdicts and len(set(verdicts)) > 1:
            by: dict[str, list[str]] = {}
            for m, ev in per_model.items():
                by.setdefault(str(ev.get("ad_verdict")), []).append(str(m))
            dist = "；".join(f"{_VLAB.get(k, k)} {len(v)} 个（{'、'.join(v)}）"
                            for k, v in sorted(by.items(), key=lambda kv: -len(kv[1])))
            result["verdict_mix"] = {k: len(v) for k, v in by.items()}
            result["conclusion"] = (
                f"**AD 有效性在各模型间不一致**：{len(verdicts)} 个模型中 {n_eff} 个判定「域外误差"
                f"显著更高」。逐模型判定：{dist}。因此只能写「部分模型上适用域能区分出更差的预测，"
                f"但不足以作为统一的域外筛选依据」，不能概称「AD 有效」；"
                f"逐模型 p 值与域内外误差见 T5 表。")
            info(f"AD 有效性（混合）：{result['conclusion']}")
        else:
            info(f"AD 有效性：{conclusion}")
        if "conflict" in set(verdicts):
            warn("适用域有效性出现口径冲突（秩方向显示域外更差、均值方向相反）："
                 "不得宣称 AD 能筛出低可靠预测，需写入 limitations")
        if "reversed" in set(verdicts):
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
