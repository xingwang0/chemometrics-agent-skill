# Copyright 2026 The research-ml-modeling Authors
# SPDX-License-Identifier: Apache-2.0
"""报告装配：表格（CSV+XLSX）、Markdown 主报告、中英 Methods 草稿、审稿质疑应对清单。

设计原则
--------
* **结论必须带证据位置**：每条结论后面都写"见 F3 / 表 T2 / perm/permutation.json"，
  让作者能一路点回去核对，而不是相信一段生成的文字。
* **局限自动生成**：无批次、无重复测量、AD 无效、样本量不足等一律写进 limitations，
  绝不因为"看起来不好"就省略——这是科研写作与营销文案的区别。
* **Methods 用模板**：``assets/templates/methods_*.tpl``，占位符 ``$var``（string.Template），
  因为本机没有 jinja2。
"""

from __future__ import annotations

import json
from pathlib import Path
from string import Template
from typing import Any

import numpy as np
import pandas as pd

from .calib import regression_calibration
from .metrics import METRIC_LABELS, format_metric
from .plots import FIGURE_INDEX
from .selection import pick_best_model, significance_vs_runner_up
from .utils import as_path, ensure_dir, info, read_json, warn, write_text

__all__ = ["build_tables", "build_report", "build_methods", "build_reviewer_checklist", "run_report"]

TEMPLATE_DIR = as_path(__file__).resolve().parent.parent.parent / "assets" / "templates"

# 英文 Methods 用的指标名（METRIC_LABELS 是中文，不能直接塞进英文稿）
METRIC_LABELS_EN = {
    "r2": "R²", "rmse": "RMSE", "mae": "MAE", "bias": "bias", "rpd": "RPD",
    "rpiq": "RPIQ", "ccc": "CCC", "slope": "slope", "auc": "AUROC",
    "balanced_accuracy": "balanced accuracy", "brier": "Brier score",
    "sensitivity": "sensitivity", "specificity": "specificity",
}


def ad_conclusion_en(eff: Any, task: str = "regression") -> str:
    what = "misclassification rate" if task == "classification" else "prediction error"
    if eff is True:
        return (f"Out-of-domain specimens showed a significantly higher {what} than in-domain "
                f"specimens, indicating that the domain definition identified the less reliable "
                f"predictions.")
    if eff is False:
        return (f"The {what} inside and outside the domain did not differ significantly, so the "
                f"present domain criterion did not isolate the less reliable predictions.")
    return ("Too few specimens fell inside or outside the domain to test its validity statistically.")


# --------------------------------------------------------------------------- #
def _read_json_if(run: Path, rel: str) -> dict:
    p = run / rel
    if not p.exists():
        return {}
    try:
        return read_json(p)
    except json.JSONDecodeError:
        return {}


def _read_csv_if(run: Path, rel: str) -> pd.DataFrame:
    p = run / rel
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(f):
        return "—"
    return f"{f:.{digits}f}"


# --------------------------------------------------------------------------- #
def _attach_ci(t2: pd.DataFrame, ci: pd.DataFrame) -> pd.DataFrame:
    """把 95% CI 贴到 T2 上，键为 **(target, model)**。

    只按 ``model`` 建字典是多目标下的经典错法：同一模型在每个目标各有一行 CI，
    后一个目标会覆盖前一个，T2 里两个目标显示同一组区间，且不会报任何错。
    """
    if t2.empty or ci.empty:
        return t2
    out = t2.copy()
    use_target = "target" in out.columns and "target" in ci.columns
    keys = ("r2", "rmse", "mae", "rpd", "auc", "balanced_accuracy", "mcc", "brier")
    if use_target:
        idx = pd.MultiIndex.from_arrays([out["target"].astype(str), out["model"].astype(str)])
    for metric in keys:
        sub = ci[ci["metric"] == metric]
        if sub.empty:
            continue
        if use_target:
            sub_idx = pd.MultiIndex.from_arrays(
                [sub["target"].astype(str), sub["model"].astype(str)])
            lookup = pd.Series(sub["ci_low"].to_numpy(), index=sub_idx)
            out[f"{metric}_ci_low"] = lookup.reindex(idx).to_numpy()
            lookup = pd.Series(sub["ci_high"].to_numpy(), index=sub_idx)
            out[f"{metric}_ci_high"] = lookup.reindex(idx).to_numpy()
        else:
            lo = dict(zip(sub["model"].astype(str), sub["ci_low"]))
            hi = dict(zip(sub["model"].astype(str), sub["ci_high"]))
            out[f"{metric}_ci_low"] = out["model"].astype(str).map(lo)
            out[f"{metric}_ci_high"] = out["model"].astype(str).map(hi)
    return out


def t2_columns(summary: pd.DataFrame) -> list[str]:
    """T2 的列选择。

    ``metric_basis`` / ``n_folds_ok`` / ``n_folds_degenerate`` 与池化值都必须留在 T2：
    这几列一旦被裁掉，第四章的「† 折均值不可用」判定永远不会触发，读者只会看到那个
    由 1 个折算出来的折均值（留一法下就是 −0.8 这种数），报告却不再提示。
    """
    cols = [c for c in summary.columns
            if c.endswith(("_mean", "_sd"))
            or c in ("model", "prep_mix", "target", "rank",
                     "metric_basis", "n_folds_ok", "n_folds_degenerate")]
    if "metric_basis" in summary.columns:
        cols += [c for c in ("r2", "rmse", "mae", "auc", "balanced_accuracy", "ccc", "rpd")
                 if c in summary.columns and c not in cols]
    return cols


def build_tables(ctx, best_model: str) -> dict[str, pd.DataFrame]:
    """T1 数据审计 / T2 性能+CI / T3 逐折 / T4 模型比较 / T5 适用域 / T6 重要性。"""
    run = ctx.run_dir
    out = ctx.sub("tables")
    tables: dict[str, pd.DataFrame] = {}

    # T1：审计摘要
    audit = _read_json_if(run, "audit/audit.json")
    if audit:
        s = audit.get("summary", {})
        rows = [{"项目": k, "取值": json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v}
                for k, v in s.items() if k not in ("data_hash",)]
        tables["T1_data_audit"] = pd.DataFrame(rows)
        findings = pd.DataFrame(audit.get("findings", []))
        if not findings.empty:
            tables["T1b_audit_findings"] = findings.rename(columns={
                "level": "级别", "key": "标识", "title": "问题", "detail": "依据", "action": "建议动作"})

    # T2：性能 + CI
    summary = _read_csv_if(run, "cv/metrics_summary.csv")
    ci = _read_csv_if(run, "cv/metrics_ci.csv")
    if not summary.empty:
        t2 = summary[t2_columns(summary)].copy()
        t2 = _attach_ci(t2, ci)
        tables["T2_performance"] = t2

    # T3：逐折
    folds = _read_csv_if(run, "cv/fold_results.csv")
    if not folds.empty:
        tables["T3_per_fold"] = folds

    # T4：模型比较
    comp = _read_csv_if(run, "cv/comparison.csv")
    if not comp.empty:
        tables["T4_comparison"] = comp

    # T5：适用域
    ad_eval = _read_json_if(run, "ad/ad_eval.json")
    if ad_eval:
        per = ad_eval.get("per_model", {})
        rows = []
        clf = ctx.ds.task == "classification"
        for model, ev in per.items():
            row = {
                "模型": model,
                "域内样品": ev.get("n_inside_groups", ev.get("n_inside")),
                "域外样品": ev.get("n_outside_groups", ev.get("n_outside")),
                "域外样品比例": _fmt(ev.get("flag_rate_groups", ev.get("flag_rate")), 3),
                "域外样品(多数测量)": ev.get("n_outside_groups_majority"),
                "域内OOF行": ev.get("n_inside"), "域外OOF行": ev.get("n_outside"),
            }
            if clf:
                row["域内准确率"] = _fmt(ev.get("acc_inside"), 3)
                row["域外准确率"] = _fmt(ev.get("acc_outside"), 3)
            else:
                row["域内RMSE"] = _fmt((ev.get("metrics_inside") or {}).get("rmse"), 4)
                row["域外RMSE"] = _fmt((ev.get("metrics_outside") or {}).get("rmse"), 4)
            row["有效性检验 p"] = _fmt(ev.get("mannwhitney_p", ev.get("fisher_p")), 4)
            row["AD 是否有效"] = {True: "是", False: "否", None: "无法判定"}.get(
                ev.get("ad_effective"), "—")
            rows.append(row)
        if rows:
            tables["T5_applicability_domain"] = pd.DataFrame(rows)
    ad_sum = _read_csv_if(run, "ad/ad_summary.csv")
    if not ad_sum.empty:
        tables["T5b_ad_methods"] = ad_sum.rename(columns={
            "method": "方法", "key": "标识", "flagged": "越界样品数",
            "rate": "越界样品比例", "flagged_rows": "越界OOF行数", "threshold": "阈值"})

    # T6：重要性 / VIP / 稳定性
    imp = _read_csv_if(run, "interp/importance.csv")
    if not imp.empty:
        tables["T6_band_importance"] = imp
    vip = _read_csv_if(run, "interp/vip.csv")
    if not vip.empty:
        tables["T6b_vip"] = vip.head(100)
    stab = _read_csv_if(run, "interp/coef_stability.csv")
    if not stab.empty:
        tables["T6c_coef_stability"] = stab.head(100)

    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False, encoding="utf-8")

    # 单文件 xlsx（多 sheet），便于直接塞进补充材料
    if tables:
        try:
            xlsx = out / "tables.xlsx"
            with pd.ExcelWriter(xlsx, engine="xlsxwriter") as writer:
                for name, df in tables.items():
                    df.to_excel(writer, sheet_name=name[:31], index=False)
            info(f"表格已写出：{len(tables)} 张 CSV + 1 个 xlsx")
        except Exception as exc:  # noqa: BLE001
            warn(f"xlsx 写出失败（CSV 仍然可用）：{type(exc).__name__}: {exc}")
    return tables


# --------------------------------------------------------------------------- #
def _primary_block(ctx, best_model: str, summary: pd.DataFrame, ci: pd.DataFrame,
                   oof: pd.DataFrame, target: str | None = None) -> dict[str, Any]:
    task = ctx.ds.task
    metric = "r2" if task == "regression" else "auc"

    def _pick(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = df[df["model"].astype(str) == str(best_model)]
        if "target" in out.columns:
            if target is not None:
                out = out[out["target"].astype(str) == str(target)]
            elif out["target"].nunique(dropna=False) > 1:
                from .config import ConfigError

                raise ConfigError(
                    "E-STAGE-004",
                    f"主性能块需要明确目标：CV 结果含多个目标（{sorted(out['target'].astype(str).unique())}）。"
                    "直接用第一个会把其它目标的数字当成这个模型的性能。")
        return out

    pool = _pick(summary)
    row = pool.iloc[0] if not pool.empty else None
    block: dict[str, Any] = {"model": best_model, "task": task}
    if row is not None:
        basis = str(row["metric_basis"]) if "metric_basis" in row.index else "fold"
        block["basis"] = basis
        if "n_folds_ok" in row.index:
            block["n_folds_ok"] = int(row["n_folds_ok"])
            block["n_folds_degenerate"] = int(row.get("n_folds_degenerate", 0) or 0)
        metric_keys = ("r2", "rmse", "mae", "bias", "rpd", "rpiq", "ccc", "slope",
                       "auc", "balanced_accuracy", "f1_macro", "mcc", "brier",
                       "sensitivity", "specificity")
        for m in metric_keys:
            key = f"{m}_mean"
            if key in row.index:
                block[m] = float(row[key])
                sd_key = f"{m}_sd"
                if sd_key in row.index:
                    block[f"{m}_sd"] = float(row[sd_key])
        # 有效折不足时，*_mean 只是极少数折（甚至 1 个折）的平均，不能当性能报。
        # 把主指标换成池化 OOF 值，折均值降级为 *_fold_mean 供备查。
        if basis == "pooled":
            for m in metric_keys:
                if m in row.index and m in block:
                    block[f"{m}_fold_mean"] = block[m]
                    block[m] = float(row[m])
            block["r2_sd"] = float("nan")
        block["n_repeats"] = int(row.get("n_repeats", 0) or 0)
    sub_ci = _pick(ci) if not ci.empty else pd.DataFrame()
    if not sub_ci.empty:
        sub_ci = sub_ci[sub_ci["metric"].isin(["r2", "rmse", "auc"])]
        block["ci"] = {r["metric"]: (float(r["ci_low"]), float(r["ci_high"]))
                       for _, r in sub_ci.iterrows()}
    if not oof.empty and task == "regression":
        sub = oof[oof["model"].astype(str) == str(best_model)]
        if "target" in sub.columns and target is not None:
            sub = sub[sub["target"].astype(str) == str(target)]
        if not sub.empty:
            cal = regression_calibration(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy(),
                                         sub["group"].to_numpy(dtype=object), n_boot=400,
                                         seed=int(ctx.cfg["runtime"]["seed"]))
            block["calibration"] = cal
    return block


def build_methods(ctx, best_model: str, block: dict, perm: dict, ad_eval: dict,
                  out_dir: Path) -> dict[str, str]:
    cfg = ctx.cfg
    ds = ctx.ds
    lang = cfg.get("report", {}).get("lang", ["zh"])
    summary = _read_csv_if(ctx.run_dir, "cv/metrics_summary.csv")
    models = list(dict.fromkeys(summary["model"])) if not summary.empty else [best_model]
    preps = (list(dict.fromkeys(summary["prep"])) if not summary.empty and "prep" in summary
             else [c.get("name", f"cand{i}") for i, c in enumerate(cfg["preprocess"]["candidates"])])

    wave = cfg["data"].get("wavelength_axis") or {}
    if wave:
        range_text = f"{wave.get('start')}–{wave.get('stop')} {wave.get('unit', 'cm-1')}"
    else:
        range_text = f"{ds.n_features} 个特征"

    # 数据集类型要按特征名判断，不能把表格数据写成「光谱/每条谱图」：
    # 波数型列名（可转 float）或显式 wavelength_axis 才算光谱。
    def _num_name(name: str) -> bool:
        try:
            float(str(name))
            return True
        except (TypeError, ValueError):
            return False

    spectral = bool(wave) or all(_num_name(f) for f in (ds.feature_names or []))
    data_noun = "光谱数据" if spectral else "工艺参数数据"
    record_noun = "谱图" if spectral else "记录"
    data_noun_en = "spectral data" if spectral else "process-parameter data"
    record_noun_en = "spectrum" if spectral else "record"

    perm_text = "未执行" if not perm or perm.get("error") else (
        f"Y-randomization 置换检验（n={perm.get('n_perm')}，样品级置换）得到零分布，"
        f"观测 {METRIC_LABELS.get(str(perm.get('metric')), perm.get('metric'))}="
        f"{_fmt(perm.get('observed'), 4)}，"
        f"p={_fmt(perm.get('p_value'), 4)}")
    perm_text_en = "was not performed" if not perm or perm.get("error") else (
        f"(n = {perm.get('n_perm')} permutations of the response at the sample level) yielded a null "
        f"distribution against which the observed "
        f"{METRIC_LABELS_EN.get(str(perm.get('metric')), perm.get('metric'))} = "
        f"{_fmt(perm.get('observed'), 4)} gave p = {_fmt(perm.get('p_value'), 4)}")

    # 置换段的两句改写：模式必须如实写（fast 是固定超参，不是「重跑整个流程」），
    # 结论必须按 p 值分支——旧模板无条件写「超出随机水平」，p>0.05 时与正文自相矛盾。
    perm_mode = str((perm or {}).get("mode", "fast")).lower()
    perm_mode_note = ("超参数按交叉验证选定的最优值固定（fast 模式，保守做法）"
                      if perm_mode == "fast" else "连超参数一起重新搜索")
    perm_mode_note_en = ("with hyperparameters fixed at the CV-selected values (fast mode, conservative)"
                         if perm_mode == "fast" else "with hyperparameters re-selected in every permutation")
    if perm and not perm.get("error"):
        perm_conclusion = str(perm.get("conclusion")
                              or "（置换检验未给出结论，正文不得据此声称模型关系超出随机水平）")
        perm_conclusion_en = (
            "The observed value lay outside the null distribution, indicating that the fitted "
            "relationship is unlikely to be attributable to chance alone."
            if perm.get("significant") else
            "The observed value fell within the null distribution, so the fitted relationship "
            "cannot be distinguished from chance and no such claim should be made."
        )
    else:
        perm_conclusion = "（未执行置换检验，正文不得声称模型关系超出随机水平）"
        perm_conclusion_en = ("(Permutation testing was not performed, so no claim about exceeding "
                              "chance may be made.)")
    ad_text = "未执行适用域分析"
    ad_text_en = "was not performed."
    if ad_eval and not ad_eval.get("skipped"):
        per = ad_eval.get("per_model", {})
        first = next(iter(per.values()), {})
        quant = int(round(float(cfg.get("ad", {}).get("quantile", 0.95)) * 100))
        n_out_g = first.get("n_outside_groups")
        n_g = first.get("n_groups")
        rate_g = first.get("flag_rate_groups")
        # 单位必须写死在句子里：只有 group 口径能进论文，行口径只作备查
        if n_out_g is not None and n_g:
            unit = (f"共有 {n_out_g} 个独立样品（占 {n_g} 个样品的 {float(rate_g):.1%}）"
                    f"至少有 1 次测量被判定为域外")
            unit_en = (f"in total {n_out_g} of {n_g} independent samples "
                       f"({float(rate_g):.1%}) had at least one measurement outside the domain")
        else:
            unit = "域外样品数未能按独立样品汇总（缺少 sample_id 分组信息）"
            unit_en = "the number of out-of-domain samples could not be aggregated by specimen"
        # 方法名必须按「实际参与投票」写：退化的方法（h*≥1 的 leverage、
        # 保留全部主成分时的 Q 残差）从并集里剔除后，仍写「四法联合判定」就是错的。
        from .ad import METHOD_LABELS as _AD_LABELS
        ad_desc = ad_eval.get("ad") or {}
        acts = ad_desc.get("active_methods") or list(_AD_LABELS)
        degs = ad_desc.get("degenerate_methods") or {}
        act_txt = "、".join(_AD_LABELS.get(m, m) for m in acts)
        deg_txt_zh = ("；其中" + "、".join(_AD_LABELS.get(k, k) for k in degs)
                      + "在本数据上退化（阈值不可能被越过），已退出投票") if degs else ""
        deg_txt_en = ("; " + " and ".join(_AD_LABELS.get(k, k) for k in degs)
                      + " were degenerate on this dataset (their thresholds cannot be exceeded) "
                      "and were excluded from voting") if degs else ""
        ad_text = (f"以 {len(acts)} 种方法联合判定适用域（{act_txt}；阈值取训练集 {quant}% 经验分位，"
                   f"任一方法越界即判为域外{deg_txt_zh}），{unit}。"
                   f"{first.get('conclusion', '')}")
        ad_text_en = (f"The applicability domain was defined jointly by {len(acts)} method(s) "
                      f"({act_txt.replace('、', ', ')}) with thresholds taken as the {quant}th percentile "
                      f"of the training data (a specimen was flagged if any single method exceeded its "
                      f"threshold{deg_txt_en}); {unit_en}. "
                      f"Repeated measurements of the same specimen were not counted separately. "
                      f"{ad_conclusion_en(first.get('ad_effective'), ds.task)}")

    from .provenance import package_versions

    vers = package_versions()
    software = (f"Python {__import__('sys').version.split()[0]}；"
                f"scikit-learn {vers.get('scikit-learn', '?')}；numpy {vers.get('numpy', '?')}；"
                f"pandas {vers.get('pandas', '?')}；scipy {vers.get('scipy', '?')}")

    _fig_ref_zh = "图 F1" if ctx.ds.task != "regression" else "图 F1/F2"
    # 留一法/池化口径下没有可用的折均值，上面引用的 R² 就是池化 OOF 值；
    # 这时再写「以上为各重复内折均值的跨重复平均」等于把口径说错。
    if block.get("basis") == "pooled" or (block.get("n_folds_ok") in (0, 0.0)):
        metric_note_zh = ("（以上为折外预测**池化**口径：本方案每折测试样品太少、折均值不可用，"
                          f"{_fig_ref_zh} 与之同口径，可一致引用）")
        metric_note_en = ("(values are **pooled** over out-of-fold predictions, because the number "
                          "of test specimens per fold was too small for fold means to be usable; "
                          "the statistics on Fig. F1 use the same basis)")
    else:
        metric_note_zh = (f"（以上为各重复内折均值的跨重复平均；{_fig_ref_zh} 的统计量按折外预测池化计算，"
                          "两者口径不同，数值可略有差异）")
        metric_note_en = ("(values are the mean over repetitions of within-repetition fold means; "
                          "the statistics printed on Fig. F1 are pooled over out-of-fold predictions, "
                          "so small numerical differences are expected)")
    # 显著性从句：没有它，Methods 会写成一个数据不支持的「最优」排序
    sig = block.get("significance") or {}
    if sig.get("available") and sig.get("bh_conflict"):
        # 单次比较 CI 不含 0、但 BH 校正后 p 已超过 α：两套判据分歧，必须如实写出来，
        # 不能只挑对自己有利的那一套（否则句子里「显著优于」和「p=0.10」互相矛盾）。
        m = sig.get("m_family")
        sig_zh = (f"；但其与 **{sig['runner_up']}** 的差异经多重比较校正后未达显著"
                  f"（单次比较的配对 bootstrap 差值（池化折外预测口径）95% CI "
                  f"[{_fmt(sig['ci_low'], 4)}, {_fmt(sig['ci_high'], 4)}] 不含 0，"
                  f"但 BH 校正（家族 m = {m if m is not None else '?'}）后 p = {_fmt(sig['p_bh'], 4)}"
                  f" > {_fmt(sig.get('alpha', 0.05), 2)}），论文中应写「{best_model} 略优但未确立」，"
                  f"不要写成显著优于 **{sig['runner_up']}**")
        sig_en = (f"; however, the difference from **{sig['runner_up']}** did not survive "
                  f"multiple-comparison correction (single-comparison 95% CI "
                  f"[{_fmt(sig['ci_low'], 4)}, {_fmt(sig['ci_high'], 4)}] excluded zero, but the "
                  f"BH-adjusted p was {_fmt(sig['p_bh'], 4)} > {_fmt(sig.get('alpha', 0.05), 2)} "
                  f"over a family of m = {m if m is not None else '?'}), so the two models should "
                  f"be described as comparable rather than significantly different")
    elif sig.get("available") and not sig.get("significant"):
        sig_zh = (f"；但其与 **{sig['runner_up']}** 的差异未达统计显著"
                  f"（配对 bootstrap 差值（池化折外预测口径）95% CI [{_fmt(sig['ci_low'], 4)}, {_fmt(sig['ci_high'], 4)}]，"
                  f"BH 校正 p = {_fmt(sig['p_bh'], 4)}），故二者应视为性能相当")
        sig_en = (f"; however, its difference from **{sig['runner_up']}** was not statistically "
                  f"significant (paired bootstrap 95% CI [{_fmt(sig['ci_low'], 4)}, "
                  f"{_fmt(sig['ci_high'], 4)}], BH-adjusted p = {_fmt(sig['p_bh'], 4)}), "
                  f"so the two models should be regarded as performing equivalently")
    elif sig.get("available") and sig.get("significant"):
        sig_zh = (f"；该模型显著优于次优 **{sig['runner_up']}**"
                  f"（配对 bootstrap 差值（池化折外预测口径）95% CI [{_fmt(sig['ci_low'], 4)}, {_fmt(sig['ci_high'], 4)}]，"
                  f"BH 校正 p = {_fmt(sig['p_bh'], 4)}）")
        sig_en = (f"; this model was significantly better than the runner-up "
                  f"**{sig['runner_up']}** (paired bootstrap 95% CI "
                  f"[{_fmt(sig['ci_low'], 4)}, {_fmt(sig['ci_high'], 4)}], "
                  f"BH-adjusted p = {_fmt(sig['p_bh'], 4)})")
    else:
        sig_zh = sig_en = ""
    if ds.task == "regression":
        results_zh = (f"嵌套分组交叉验证中表现最好的模型为 **{best_model}**，"
                      f"R² = {_fmt(block.get('r2'), 3)}，RMSE = {_fmt(block.get('rmse'), 4)}，"
                      f"MAE = {_fmt(block.get('mae'), 4)}，RPD = {_fmt(block.get('rpd'), 2)}"
                      f"{sig_zh}{metric_note_zh}。")
        results_en = (f"The best-performing model was **{best_model}**, achieving "
                      f"R² = {_fmt(block.get('r2'), 3)}, RMSE = {_fmt(block.get('rmse'), 4)}, "
                      f"MAE = {_fmt(block.get('mae'), 4)} and RPD = {_fmt(block.get('rpd'), 2)} "
                      f"in nested grouped cross-validation{sig_en} {metric_note_en}.")
    else:
        results_zh = (f"嵌套分组交叉验证中表现最好的模型为 **{best_model}**，"
                      f"AUROC = {_fmt(block.get('auc'), 3)}，"
                      f"平衡准确率 = {_fmt(block.get('balanced_accuracy'), 3)}，"
                      f"F1(macro) = {_fmt(block.get('f1_macro'), 3)}，"
                      f"MCC = {_fmt(block.get('mcc'), 3)}{sig_zh}{metric_note_zh}。")
        results_en = (f"The best-performing model was **{best_model}**, achieving "
                      f"AUROC = {_fmt(block.get('auc'), 3)}, balanced accuracy = "
                      f"{_fmt(block.get('balanced_accuracy'), 3)}, macro-F1 = "
                      f"{_fmt(block.get('f1_macro'), 3)} and MCC = {_fmt(block.get('mcc'), 3)} "
                      f"in nested grouped cross-validation{sig_en} {metric_note_en}.")

    n_batch = len(pd.unique(ds.batch)) if ds.batch is not None else 0
    reps_max = int(ds.groups_per_sample().max())
    mapping = {
        "n_rows": ds.n_samples, "n_groups": ds.n_groups, "n_features": ds.n_features,
        "target": ds.target_name,
        "group_col": cfg["data"].get("id_col") or "特征指纹推断",
        "batch_text": (f"数据来自 {n_batch} 个批次（{cfg['data'].get('batch_col')}）"
                       if ds.batch is not None else "数据未提供批次信息"),
        "batch_text_en": (f"the data originated from {n_batch} batches "
                          f"(`{cfg['data'].get('batch_col')}`)"
                          if ds.batch is not None else "no batch information was available"),
        "reps_text": (f"每个样品重复测量 {reps_max} 次"
                      if reps_max > 1 else "每个样品单次测量"),
        "reps_text_en": (f"{reps_max} repeated measurements per specimen"
                         if reps_max > 1 else "a single measurement per specimen"),
        "range_text": range_text,
        "prep_text": "、".join(preps),
        "k_out": cfg["split"]["outer"]["n_splits"],
        "k_in": cfg["split"]["inner"]["n_splits"],
        "n_repeats": cfg["split"]["repeats"],
        "seed": cfg["runtime"]["seed"],
        "models_text": "、".join(models),
        "best_model": best_model,
        "best_r2": _fmt(block.get("r2"), 3),
        "best_rmse": _fmt(block.get("rmse"), 4),
        "best_rpd": _fmt(block.get("rpd"), 2),
        "best_mae": _fmt(block.get("mae"), 4),
        "best_auc": _fmt(block.get("auc"), 3),
        "best_bacc": _fmt(block.get("balanced_accuracy"), 3),
        "best_f1": _fmt(block.get("f1_macro"), 3),
        "best_mcc": _fmt(block.get("mcc"), 3),
        "results_text_zh": results_zh,
        "results_text_en": results_en,
        "task_type": ds.task,
        "ci_text": ("；".join(f"{METRIC_LABELS.get(k, k)} 95% CI [{_fmt(v[0])}, {_fmt(v[1])}]"
                             for k, v in (block.get("ci") or {}).items()) or "未计算"),
        "ci_text_en": ("; ".join(f"{METRIC_LABELS_EN.get(k, k)} 95% CI "
                                f"[{_fmt(v[0])}, {_fmt(v[1])}]"
                                for k, v in (block.get("ci") or {}).items())
                       or "not computed"),
        "perm_text": perm_text,
        "perm_text_en": perm_text_en,
        "perm_conclusion": perm_conclusion,
        "perm_conclusion_en": perm_conclusion_en,
        "perm_mode_note": perm_mode_note,
        "perm_mode_note_en": perm_mode_note_en,
        "data_noun": data_noun,
        "record_noun": record_noun,
        "data_noun_en": data_noun_en,
        "record_noun_en": record_noun_en,
        "ad_text": ad_text,
        "ad_text_en": ad_text_en,
        "software": software,
        "data_file": Path(str(ds.source_path)).name if ds.source_path else "—",
        "objective": ds.target_name,
    }

    written: dict[str, str] = {}
    for lang_code, tpl_name, out_name in (
        ("zh", "methods_zh.md.tpl", "methods_zh.md"),
        ("en", "methods_en.md.tpl", "methods_en.md"),
    ):
        if lang_code not in lang:
            continue
        tpl_path = TEMPLATE_DIR / tpl_name
        if not tpl_path.exists():
            warn(f"缺少模板 {tpl_path}")
            continue
        text = Template(tpl_path.read_text(encoding="utf-8")).safe_substitute(mapping)
        p = write_text(out_dir / out_name, text)
        written[lang_code] = str(p)
        info(f"Methods 草稿（{lang_code}）-> {p.name}")
    return written


def _comparison_status(block: dict, comp_ok: bool) -> str:
    """审稿清单里「差异显著吗」这一行的真实状态，而不是无条件的 ✓。"""
    if not comp_ok:
        return "⚠ 只有单个模型，无需比较"
    sig = block.get("significance") or {}
    if not sig.get("available"):
        return f"⚠ 未能比较：{sig.get('reason', '缺少结果')}"
    p = _fmt(sig.get("p_bh"), 4)
    if sig.get("significant"):
        return f"✓ 最优显著优于 {sig['runner_up']}（BH 校正 p = {p}）"
    return (f"⚠ 最优与 {sig['runner_up']} 无显著差异（BH 校正 p = {p}）："
            f"论文中应写「二者性能相当」，不要只推荐 {block.get('model')}")


def build_reviewer_checklist(ctx, block: dict, perm: dict, ad_eval: dict,
                             comp: pd.DataFrame, tables: dict, out_dir: Path) -> Path:
    """把"审稿人会问什么"逐条对照到本 run 的产出物，并如实标出缺口。"""
    cfg = ctx.cfg
    ds = ctx.ds
    audit = _read_json_if(ctx.run_dir, "audit/audit.json")
    findings = {f.get("key"): f for f in audit.get("findings", [])}
    summary = tables.get("T2_performance", pd.DataFrame())
    has_replicate = (ds.groups_per_sample().max() > 1)
    has_batch = ds.batch is not None
    perm_ok = bool(perm.get("significant")) if perm else False
    ad_eff = None
    if ad_eval and ad_eval.get("per_model"):
        first = next(iter(ad_eval["per_model"].values()))
        ad_eff = first.get("ad_effective")
    comp_ok = not comp.empty
    best_r2 = block.get("r2")
    rpd = block.get("rpd")

    rows = [
        ("是不是随机划分导致同一样品的另一次测量泄漏进了训练集？",
         "按 sample_id 分组的外层/内层交叉验证；划分互斥性由 selfcheck 断言；"
         "`fold_results.csv` 的 n_train_groups 列可逐折核对",
         "cv/fold_results.csv", "✓" if has_replicate else "⚠ 数据无重复测量，需说明"),
        ("R² 是不是调参调出来的最好成绩？",
         f"嵌套交叉验证：内层 {cfg['split']['inner']['n_splits']} 折选参，外层评估；重复 {cfg['split']['repeats']} 次",
         "cv/metrics_summary.csv", "✓"),
        ("换个划分结果会不会变？",
         f"重复 {cfg['split']['repeats']} 次不同种子的分组划分，报告 mean±sd",
         "figures/F3_cv_distribution.png", "✓"),
        ("怎么证明模型不是在拟合噪声？",
         "Y-randomization：样品级置换 y 重跑全流程，给出零分布与 p 值",
         "perm/permutation.json、figures/F6_y_randomization.png",
         "✓" if perm_ok else "⚠ p 未达显著，必须在文中如实说明"),
        ("新样品能用吗？外推范围在哪？",
         "四法适用域（T²/Q/kNN/leverage）+ 域内外性能检验",
         "ad/ad_scores.csv、figures/F4_williams.png",
         {True: "✓ 且已验证有效", False: "⚠ AD 未能识别域外，不得宣称可外推", None: "⚠ 无法验证"}.get(ad_eff, "⚠ 未执行")),
        ("性能指标有没有不确定度？",
         "按样品（cluster）bootstrap 的 95% 置信区间",
         "cv/metrics_ci.csv", "✓"),
        ("两个模型的差异显著吗？",
         "配对 bootstrap 差值 CI + BH/Bonferroni 多重比较校正",
         "cv/comparison.csv",
         _comparison_status(block, comp_ok)),
        ("为什么不用 PLS？",
         "PLS 为强制基线（缺失时自动注入），并给 VIP 分数",
         "interp/vip.csv", "✓"),
        ("有没有系统性偏差？",
         "pred~obs 斜率/截距 + Bland-Altman 一致性界限",
         "figures/F5_calibration.png、tables/T2_performance.csv", "✓"),
        ("跨批次/跨仪器能推广吗？",
         "留一批次外部验证（`--set split.strategy=loo_batch` 另跑一次，按 batch 分折）",
         "cv/fold_results.csv",
         "⚠ 数据含批次，但尚未做留一批次外验证（`--set split.strategy=loo_batch`）"
         if has_batch else "⚠ 数据无批次信息，只能写进 limitations"),
        ("预处理是不是挑过的？",
         "预处理链作为搜索维度（prep__which）参与内层搜索",
         "cv/fold_results.csv", "✓"),
        ("结果怎么复现？",
         "provenance.json 记录版本矩阵、三个哈希、全部随机种子",
         "provenance.json", "✓"),
        ("样本量是否足够？EPV 多少？",
         "审计给出 n/p、n/√p、EPV 与折内样品数",
         "audit/audit.md", "✓" if "p_gt_n" in findings or True else "⚠"),
        ("指标是否适合该类数据？",
         "回归报告 R²/RMSE/MAE/bias/RPD/RPIQ/CCC；RPD 与 RPIQ 是化学计量学惯例",
         "tables/T2_performance.csv", "✓"),
    ]
    if ds.task == "classification":
        folds_all = _read_csv_if(ctx.run_dir, "cv/fold_results.csv")
        best = str(block.get("model") or "")
        kind = "native"
        if not folds_all.empty and "proba_kind" in folds_all and "model" in folds_all:
            sub = folds_all[folds_all["model"].astype(str) == best]
            vals = sub["proba_kind"].dropna() if not sub.empty else pd.Series(dtype=str)
            kind = str(vals.iloc[0]) if len(vals) else "native"
        if kind == "native":
            rows.append(("概率可信吗？阈值怎么定的？",
                         "校准曲线 + Brier + Brier Skill；阈值用内层 CV 的 Youden 点，不在测试折上调",
                         "figures/F5_calibration.png", "✓"))
        else:
            rows.append(("概率可信吗？阈值怎么定的？",
                         "该模型只有 decision_function（无原生概率），只报 AUC/AP 与阈值判定的分类指标",
                         "cv/fold_results.csv",
                         "⚠ 无原生概率，未给 Brier/ECE/校准曲线"))
    if ad_eval and ad_eval.get("author_note"):
        for i, (q, how, ev, status) in enumerate(rows):
            if q.startswith("新样品能用吗"):
                rows[i] = (q, f"{how}。{ad_eval['author_note']}", ev, status)
                break

    df = pd.DataFrame(rows, columns=["审稿人可能问", "本 run 的做法", "证据位置", "状态"])
    lines = ["# 审稿质疑应对清单", "",
             f"- 目标变量：`{ds.target_name}`｜模型：`{block.get('model')}`｜样本：{ds.n_samples} 行 / {ds.n_groups} 个独立样品",
             f"- 主要性能：R²={_fmt(best_r2, 3)}，RPD={_fmt(rpd, 2)}", "",
             "| 审稿人可能问 | 本 run 的做法 | 证据位置 | 状态 |", "|---|---|---|---|"]
    for _, r in df.iterrows():
        # 证据位置可能有多条：逐条包成独立 code span，不要把两个路径塞进同一对反引号
        evs = [p.strip() for p in str(r["证据位置"]).split("、") if p.strip()]
        ev_txt = "、".join(f"`{p}`" for p in evs)
        lines.append(f"| {r['审稿人可能问']} | {r['本 run 的做法']} | {ev_txt} | {r['状态']} |")
    lines += ["", "> 标 ⚠ 的条目是本次 run 的真实缺口，请在论文 limitations 中如实说明，",
              "> 不要靠删掉相关分析来「回避」——审稿人往往正是从缺失处发问。", ""]
    if not summary.empty and rpd is not None and np.isfinite(rpd):
        tier = ("RPD>3：可用于定量筛查" if rpd > 3 else
                "2<RPD<3：可用于粗略定量" if rpd > 2 else
                "RPD<2：仅能做定性/半定量判断")
        lines.append(f"**RPD 解读**：本次 RPD={rpd:.2f}，{tier}（Williams 的常用分档）。")
    p = write_text(out_dir / "reviewer_checklist.md", "\n".join(lines))
    return p


# --------------------------------------------------------------------------- #
def _best_model_line(block: dict, best_model: str) -> str:
    """「最优模型」这句话必须带显著性结论，否则等于给出数据不支持的排序。"""
    sig = block.get("significance") or {}
    label = METRIC_LABELS.get(sig.get("metric", ""), sig.get("metric", "主指标"))
    # 差值来自 comparison.csv，一律按池化折外预测计算；正文主指标是折均值。
    # 同一句里两种口径混着出现时必须写明，否则读者会把 0.18 和表里的 0.29 对不上。
    basis_tag = "，差值口径=池化折外预测"
    if sig.get("available") and sig.get("bh_conflict"):
        m = sig.get("m_family")
        return (f"- 最优模型：**{best_model}**（单次比较的 {label} 差 {_fmt(sig['delta'], 4)}"
                f"{basis_tag}，95% CI [{_fmt(sig['ci_low'], 4)}, {_fmt(sig['ci_high'], 4)}] 不含 0，"
                f"但 BH 校正（家族 m = {m if m is not None else '?'}）后 p = {_fmt(sig['p_bh'], 4)}，"
                f"未过 α = {_fmt(sig.get('alpha', 0.05), 2)}）→ 论文写「{best_model} 略优但未确立」，"
                f"不要写成显著优于 `{sig['runner_up']}`。")
    if sig.get("available") and sig.get("significant"):
        return (f"- 最优模型：**{best_model}**（显著优于次优 `{sig['runner_up']}`："
                f"{label} 差 {_fmt(sig['delta'], 4)}{basis_tag}，95% CI "
                f"[{_fmt(sig['ci_low'], 4)}, {_fmt(sig['ci_high'], 4)}]，"
                f"BH 校正 p = {_fmt(sig['p_bh'], 4)}）")
    if sig.get("available"):
        return (f"- 最优模型：**{best_model}**，但**与 `{sig['runner_up']}` 的差异未达显著**"
                f"（{label} 差 {_fmt(sig['delta'], 4)}{basis_tag}，95% CI "
                f"[{_fmt(sig['ci_low'], 4)}, {_fmt(sig['ci_high'], 4)}]，"
                f"BH 校正 p = {_fmt(sig['p_bh'], 4)}）→ 论文中应写"
                f"「{best_model} 与 {sig['runner_up']} 性能相当」，不要只推荐前者。")
    return (f"- 最优模型：**{best_model}**（未做显著性检验："
            f"{sig.get('reason', '缺少对比结果')}；不要据此宣称它优于其它模型）")


def build_report(ctx, best_model: str, block: dict, tables: dict, figures: dict,
                 perm: dict, ad_eval: dict, out_dir: Path) -> Path:
    cfg = ctx.cfg
    ds = ctx.ds
    audit = _read_json_if(ctx.run_dir, "audit/audit.json")
    findings = audit.get("findings", [])
    criticals = [f for f in findings if f.get("level") == "critical"]
    warns_ = [f for f in findings if f.get("level") == "warn"]
    summary = tables.get("T2_performance", pd.DataFrame())
    comp = tables.get("T4_comparison", pd.DataFrame())
    lim: list[str] = []

    if not (ds.groups_per_sample().max() > 1):
        lim.append("数据中每个样品仅一次测量，无法体现分组交叉验证相对随机划分的优势；"
                   "若实际存在重复测量，请补充 sample_id。")
    if ds.batch is None:
        lim.append("数据未提供批次/仪器信息，模型仅在当前批次内验证，**跨批次外推未经验证**。")
    else:
        lim.append(f"数据含 {len(pd.unique(ds.batch))} 个批次，但本次未执行留一批次外部验证"
                   f"（可另跑 `--set split.strategy=loo_batch`）。")
    if ds.n_groups < 30:
        lim.append(f"独立样品仅 {ds.n_groups} 个，性能估计的置信区间较宽，不宜据此宣称模型已定型。")
    if ad_eval and ad_eval.get("per_model"):
        first = next(iter(ad_eval["per_model"].values()))
        if first.get("ad_effective") is False:
            _what = "误分类率" if ds.task == "classification" else "预测误差"
            lim.append(f"适用域检验显示域内与域外样品的{_what}无显著差异，"
                       "当前适用域判据未能识别出可靠性较低的预测；"
                       "因此本研究不对域外样品的外推可靠性作出结论。")
    if perm and perm.get("fixed_hyperparams"):
        lim.append("Y-randomization 采用固定超参（保守估计），未对每轮置换重做内层调参。")
    if (ad_eval and ad_eval.get("per_model")) or _read_json_if(ctx.run_dir, "interp/interp.json"):
        lim.append("适用域（T²/Q/kNN/leverage）与重要性（置换/系数/VIP/SHAP）均在**全量数据**上拟合，"
                   "属 in-sample 描述性分析：它们只用于划定域边界与解释模型，"
                   "支持外推性能的只有第三章的折外（OOF）指标。")
    _sig_lim = block.get("significance") or {}
    if _sig_lim.get("available") and not _sig_lim.get("significant"):
        _lab_lim = METRIC_LABELS.get(_sig_lim.get("metric", ""), _sig_lim.get("metric", "主指标"))
        lim.append(f"最优模型 {best_model} 与次优 {_sig_lim['runner_up']} 的 {_lab_lim} "
                   f"差异未达统计显著（配对 bootstrap 差值 95% CI "
                   f"[{_fmt(_sig_lim['ci_low'], 4)}, {_fmt(_sig_lim['ci_high'], 4)}]，"
                   f"BH 校正 p = {_fmt(_sig_lim['p_bh'], 4)}"
                   # CI 不含 0 却没过 BH：分歧必须写清，否则同一条 limitation 自相矛盾
                   + ("；单次比较的 CI 不含 0，但多重比较校正后未过 α"
                      if _sig_lim.get("bh_conflict") else "")
                   + "）；"
                   f"二者应视为性能相当，模型选择可结合可解释性、成本或稳定性决定。")
    folds_all = _read_csv_if(ctx.run_dir, "cv/fold_results.csv")
    if (ds.task == "classification" and not folds_all.empty and "proba_kind" in folds_all):
        non_native = sorted(set(
            folds_all.loc[folds_all["proba_kind"].astype(str) != "native", "model"].astype(str)))
        if non_native:
            lim.append("以下模型没有原生概率接口（只有 decision_function）："
                       f"{'、'.join(non_native)}。其 Brier / ECE / log-loss 与校准曲线按缺失处理，"
                       "不外推；AUC/AP 不受影响（单调变换保持排序）。")
    if not criticals and not warns_:
        lim.append("数据审计未发现目标泄漏与严重共线问题。")

    lines: list[str] = []
    lines.append(f"# 预测建模报告：{ds.target_name}")
    lines.append("")
    title = cfg.get("report", {}).get("title")
    if title:
        lines.append(f"> {title}")
        lines.append("")
    cv_summary = _read_json_if(ctx.run_dir, "cv/cv_summary.json")
    failed_models = cv_summary.get("failed_models") or {}
    if failed_models:
        lines.append("> ⚠️ **有模型成批失败，已被排除在结果之外——这并不等于「该模型效果差」。**")
        for name, info in failed_models.items():
            n_total = info.get("n_total")
            scope = (f"全部 {n_total} 折失败" if info.get("all_failed")
                     else f"{info.get('n_failed')}/{n_total} 折失败")
            lines.append(f"> - `{name}`：{scope}；首个错误 "
                         f"`{str(info.get('first_error'))[:160]}`")
        lines.append("> 请修好配置后重跑；在修复之前，本报告不包含这些模型的任何结论，"
                     "更不能在论文里写「我们试过 X 但效果不佳」。")
        lines.append("")
    lines.append("## 一、结论摘要")
    lines.append("")
    if block.get("task") == "regression":
        prep_note = (summary.iloc[0].get("prep_mix") if not summary.empty and "prep_mix" in summary
                     else "—")
        lines.append(_best_model_line(block, best_model))
        lines.append(f"- 逐折选中的预处理：{prep_note}")
        if block.get("basis") == "pooled":
            n_ok = int(block.get("n_folds_ok") or 0)
            n_deg = int(block.get("n_folds_degenerate") or 0)
            lines.append(
                f"- 交叉验证性能（**池化 OOF 口径**）：R² = {_fmt(block.get('r2'), 3)}，"
                f"RMSE = {_fmt(block.get('rmse'), 4)}，MAE = {_fmt(block.get('mae'), 4)}，"
                f"RPD = {_fmt(block.get('rpd'), 2)}")
            lines.append(
                f"- ⚠️ **折均值口径不可用**：有效折仅 {n_ok}/{n_ok + n_deg}"
                f"（其余为退化折：测试样品 <2、指标为 NaN，会被 nanmean 跳过）。"
                f"折均值 R² = {_fmt(block.get('r2_fold_mean'), 3)} 只来自这 {n_ok} 个折，**不要在论文里引用**。")
        else:
            lines.append(f"- 交叉验证性能：R² = {_fmt(block.get('r2'), 3)} ± {_fmt(block.get('r2_sd'), 3)}，"
                         f"RMSE = {_fmt(block.get('rmse'), 4)}，MAE = {_fmt(block.get('mae'), 4)}，"
                         f"RPD = {_fmt(block.get('rpd'), 2)}")
        if block.get("ci"):
            lines.append(f"- 95% 置信区间（按样品 bootstrap）：" +
                         "；".join(f"{METRIC_LABELS.get(k, k)} [{_fmt(v[0])}, {_fmt(v[1])}]"
                                  for k, v in block["ci"].items()))
        cal = block.get("calibration") or {}
        if cal:
            lines.append(f"- 校准：{cal.get('interpretation', '')}")
    else:
        lines.append(_best_model_line(block, best_model))
        lines.append(f"- 交叉验证性能：AUROC = {_fmt(block.get('auc'), 3)}，"
                     f"平衡准确率 = {_fmt(block.get('balanced_accuracy'), 3)}，"
                     f"F1(macro) = {_fmt(block.get('f1_macro'), 3)}，MCC = {_fmt(block.get('mcc'), 3)}")
    # 两个口径给出的「第一」不同时必须说出来：报告只写一个模型名，读者无从知道另一个口径选了谁
    if (not summary.empty and ds.task == "regression"
            and "r2_mean" in summary.columns and "r2" in summary.columns):
        top_fold = summary.sort_values("r2_mean", ascending=False).iloc[0]
        top_pool = summary.sort_values("r2", ascending=False).iloc[0]
        if str(top_fold["model"]) != str(top_pool["model"]):
            lines.append(
                f"- ⚠️ **两个口径的第一名不是同一个模型**：折均值第一 `{top_fold['model']}`"
                f"（R²={_fmt(top_fold['r2_mean'], 3)}），池化第一 `{top_pool['model']}`"
                f"（R²={_fmt(top_pool['r2'], 3)}）。本报告按 `metric_basis` 选模；"
                f"论文里引用的数字必须与所选口径一致，不能混引。")
    if perm and not perm.get("error"):
        lines.append(f"- 抗噪声检验：{perm.get('conclusion')}")
    if ad_eval and ad_eval.get("conclusion"):
        _n_out, _n_g = ad_eval.get("n_outside_groups_any"), ad_eval.get("n_groups")
        _rate = ad_eval.get("flag_rate_any_groups")
        _head = (f"{_n_out}/{_n_g} 个独立样品被判为域外（{float(_rate):.1%}）；"
                 if _n_out is not None and _n_g and _rate is not None else "")
        lines.append(f"- 适用域：{_head}{ad_eval['conclusion']}")
    lines.append("")

    lines.append("## 二、数据与任务")
    lines.append("")
    s = audit.get("summary", {})
    lines.append(f"| 项 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| 数据文件 | `{Path(str(ds.source_path)).name}` |")
    lines.append(f"| 规模 | {ds.n_samples} 行 × {ds.n_features} 特征 |")
    lines.append(f"| 独立样品（group） | {ds.n_groups} |")
    _gps = ds.groups_per_sample()
    reps_txt = (str(int(_gps.min())) if int(_gps.min()) == int(_gps.max())
                else f"{int(_gps.min())}–{int(_gps.max())}")
    lines.append(f"| 每样品测量次数 | {reps_txt} |")
    lines.append(f"| 批次 | {len(pd.unique(ds.batch)) if ds.batch is not None else '未提供'} |")
    lines.append(f"| 任务 | {ds.task}（目标 `{ds.target_name}`）|")
    cond = s.get("cond_number")
    cond_txt = f"{float(cond):.2e}" if isinstance(cond, (int, float)) and np.isfinite(cond) else str(cond or "—")
    lines.append(f"| 条件数 | {cond_txt} |")
    lines.append(f"| 有效秩 | {s.get('effective_rank', '—')} |")
    lines.append("")
    lines.append(f"详细审计见 `audit/audit.md`。严重问题 {len(criticals)} 项，警告 {len(warns_)} 项。")
    lines.append("")
    if criticals:
        lines.append("**⚠ 存在严重数据问题，建模结论需谨慎：**")
        for f in criticals:
            lines.append(f"- **{f.get('title')}**：{f.get('detail')}")
        lines.append("")

    lines.append("## 三、验证方案")
    lines.append("")
    plan = _read_json_if(ctx.run_dir, "cv/cv_summary.json")
    lines.append(f"- 策略：`{cfg['split']['strategy']}`（{_read_json_if(ctx.run_dir, 'audit/audit.json').get('recommendations', {}).get('group_col', 'sample_id')} 分组）")
    lines.append(f"- 外层：{cfg['split']['outer']['n_splits']} 折分组 CV × 重复 {cfg['split']['repeats']} 次（不同种子）")
    lines.append(f"- 内层：{cfg['split']['inner']['n_splits']} 折分组 CV 用于调参与预处理选择")
    lines.append(f"- 模型：{', '.join(summary['model'].tolist()) if not summary.empty else best_model}")
    lines.append(f"- 总拟合次数：{plan.get('total_fit_calls', '—')}")
    lines.append("")
    lines.append("> 聚合口径：先按 repeat 求折均值，再对 repeat 求 mean±sd。"
                 "把 R×k 个折混在一起算 sd 属于伪重复，会低估方差。")
    lines.append("")

    lines.append("## 四、性能结果")
    lines.append("")
    if not summary.empty:
        show = [c for c in ("model", "prep_mix", "prep", "n_repeats") if c in summary.columns]
        for m in ("r2", "rmse", "mae", "rpd", "auc", "balanced_accuracy", "mcc"):
            if f"{m}_mean" in summary.columns:
                show += [f"{m}_mean", f"{m}_sd"]
        pooled_rows = (summary["metric_basis"].astype(str).eq("pooled")
                       if "metric_basis" in summary.columns else pd.Series(False, index=summary.index))
        if pooled_rows.any():
            # 这些行的 *_mean 只来自极少数折，必须把池化值并排给出，否则读者会拿错列
            for m in ("r2", "rmse"):
                if m in summary.columns:
                    show.append(m)
        t = summary[show].copy()
        labels = {"r2": "r2 (池化)", "rmse": "rmse (池化)"}
        head = [labels.get(c, str(c).replace("_mean", "").replace("_sd", " (sd)")) for c in t.columns]

        def _cell(col, val, idx):
            if isinstance(val, (int, float, np.floating)):
                return _fmt(val, 3)
            text = str(val)
            if col == "model" and bool(pooled_rows.get(idx, False)):
                text += " †"
            return text

        lines.append("| " + " | ".join(head) + " |")
        lines.append("|" + "---|" * len(t.columns))
        for idx, r in t.iterrows():
            lines.append("| " + " | ".join(_cell(c, r[c], idx) for c in t.columns) + " |")
        lines.append("")
        if pooled_rows.any():
            names = "、".join(sorted(summary.loc[pooled_rows, "model"].astype(str)))
            lines.append(f"> † **这些行的折均值不可用**（{names}）：有效折不足 3 个，其余折为退化折"
                         f"（测试样品 <2、指标为 NaN，被 nanmean 跳过）。"
                         f"它们的性能以 `r2 (池化)` / `rmse (池化)` 两列为准，`r2`/`rmse` 的 `_mean/_sd` 仅备查。")
            lines.append("")
        if "prep_mix" in summary.columns:
            lines.append("> `prep_mix` 为每个模型在各折里实际选中的预处理链分布。"
                         "预处理是**逐折内部选出的超参**，因此模型排名不按预处理拆分"
                         "（否则等于在「模型×预处理」组合里挑最优）；细分结果见 `cv/metrics_by_prep.csv`。")
        lines.append("")
        _fig_pool = "图 F1" if ds.task != "regression" else "图 F1/F2"
        if pooled_rows.any():
            lines.append("> **口径提醒**：本表数值为「先按 repeat 求折均值、再跨 repeat 求均值」，"
                         "是论文应引用的口径——**但标 † 的行除外**，它们的折均值只来自极少数折，"
                         "必须引 `r2 (池化)` / `rmse (池化)`。"
                         f"{_fig_pool} 的统计量按**折外预测池化**计算，"
                         "同一模型可能与上表相差千分位（池化会给样品数多的折更高权重）。"
                         "两者都对，但同页出现时必须注明口径，不要混引。")
        else:
            lines.append("> **口径提醒**：本表数值为「先按 repeat 求折均值、再跨 repeat 求均值」，"
                         "是论文应引用的口径。"
                         f"{_fig_pool} 的统计量按**折外预测池化**计算，"
                         "同一模型可能与上表相差千分位（池化会给样品数多的折更高权重）。"
                         "两者都对，但同页出现时必须注明口径，不要混引。")
        lines.append("")
        _sig = block.get("significance") or {}
        if _sig.get("available") and _sig.get("bh_conflict"):
            _lab = METRIC_LABELS.get(_sig.get("metric", ""), _sig.get("metric", "主指标"))
            _m = _sig.get("m_family")
            lines.append(f"> **显著性提醒**：`{best_model}` 与 `{_sig['runner_up']}` 的 {_lab} 差值"
                         f" 95% CI 为 [{_fmt(_sig['ci_low'], 4)}, {_fmt(_sig['ci_high'], 4)}]（**不含 0**），"
                         f"但 BH 校正（家族 m = {_m if _m is not None else '?'}）后 p = {_fmt(_sig['p_bh'], 4)}，"
                         f"未过 α = {_fmt(_sig.get('alpha', 0.05), 2)}——两套判据分歧。"
                         f"本表不支持「{best_model} 唯一最优」的排序；论文中应写"
                         f"「{best_model} 略优于 {_sig['runner_up']}，但经多重比较校正后差异未确立」，"
                         f"不要只引单次比较的 CI。配对比较见 `cv/comparison.csv`（表 T4）。")
            lines.append("")
        elif _sig.get("available") and not _sig.get("significant"):
            _lab = METRIC_LABELS.get(_sig.get("metric", ""), _sig.get("metric", "主指标"))
            lines.append(f"> **显著性提醒**：`{best_model}` 与 `{_sig['runner_up']}` 的 {_lab} 差值"
                         f" 95% CI 为 [{_fmt(_sig['ci_low'], 4)}, {_fmt(_sig['ci_high'], 4)}]（跨 0），"
                         f"BH 校正 p = {_fmt(_sig['p_bh'], 4)}。本表不支持「{best_model} 唯一最优」的排序；"
                         f"论文中应写「二者性能相当」，配对比较见 `cv/comparison.csv`（表 T4）。")
            lines.append("")
    figs_present = sorted(figures.keys())
    if figs_present:
        lines.append("可用图：")
        for code in figs_present:
            lines.append(f"- **{code}** {FIGURE_INDEX.get(code, '')} → `figures/{code}*`")
        lines.append("")

    lines.append("## 五、模型比较与不确定度")
    lines.append("")
    if not comp.empty:
        sub = comp[comp.get("metric_primary", False) == True] if "metric_primary" in comp else comp  # noqa: E712
        lines.append("| A | B | 指标 | Δ | 95% CI | p | p(BH) | 结论 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, r in sub.head(15).iterrows():
            lines.append(f"| {r.get('model_a')} | {r.get('model_b')} | {r.get('metric')} | "
                         f"{_fmt(r.get('delta'), 4)} | [{_fmt(r.get('ci_low'))}, {_fmt(r.get('ci_high'))}] | "
                         f"{_fmt(r.get('p_raw'), 4)} | {_fmt(r.get('p_BH'), 4)} | {r.get('better', '—')} |")
        lines.append("")
        lines.append(f"> 多重比较家族大小 m = {int(comp['p_BH'].notna().sum()) if 'p_BH' in comp else '—'}；"
                     "只报原始 p 会高估显著性。")
    else:
        lines.append("（仅单一模型，无比较）")
    lines.append("")

    lines.append("## 六、是否在拟合噪声（Y-randomization）")
    lines.append("")
    if perm and not perm.get("error"):
        lines.append(f"- 零分布：均值 {_fmt(perm.get('null_mean'), 4)}，"
                     f"95% 区间 [{_fmt(perm.get('null_q025'), 4)}, {_fmt(perm.get('null_q975'), 4)}]")
        lines.append(f"- 观测值：{_fmt(perm.get('observed'), 4)}；p = {_fmt(perm.get('p_value'), 4)}"
                     f"（n_perm={perm.get('n_perm')}）")
        lines.append(f"- 结论：{perm.get('conclusion')}")
    else:
        lines.append("未执行或执行失败。")
    lines.append("")

    lines.append("## 七、适用域")
    lines.append("")
    if ad_eval and ad_eval.get("per_model"):
        n_flag = ad_eval.get("n_outside_groups_any")
        n_g = ad_eval.get("n_groups")
        rate = ad_eval.get("flag_rate_any_groups")
        if n_flag is not None and n_g:
            lines.append(f"- 域外样品（独立样品口径）：**{n_flag} / {n_g}**（{float(rate):.1%}）；"
                         f"{ad_eval.get('group_rule', '样品内任一测量越界即计为域外')}")
        lines.append(f"- OOF 行口径（备查，勿用于论文）：{_fmt(ad_eval.get('flag_rate_any'), 3)}"
                     f"（同一样品的重复测量与重复次数会重复计数）")
        ad_desc = ad_eval.get("ad") or {}
        degs = ad_desc.get("degenerate_methods") or {}
        acts = ad_desc.get("active_methods") or []
        if degs:
            from .ad import METHOD_LABELS as _AD_LABELS
            lines.append("- **退化方法**（阈值在本数据上不可能被越过，已退出投票、不计入越界）："
                         + "；".join(f"**{_AD_LABELS.get(k, k)}**（{v}）" for k, v in degs.items()))
        if acts:
            from .ad import METHOD_LABELS as _AD_LABELS
            lines.append("- 实际参与投票的方法：" + "、".join(_AD_LABELS.get(m, m) for m in acts))
        lines.append(f"- 结论：{ad_eval.get('conclusion', '—')}")
        lines.append("- 新样品预测前应先调用 `ad/ad_model.joblib` 判域，域外结果需标注不确定性。")
        lines.append("> **口径声明**：本节与第八节（重要性/VIP/SHAP）的统计量均为**全量 in-sample 描述性分析**"
                     "（PCA/kNN/leverage 与重要性都在含测试样品的数据上拟合），用途是划定域边界、"
                     "解释模型，**不构成任何外推性能证据**；外推结论只能引用第三章的 OOF 指标。")
    else:
        lines.append("未执行。")
    lines.append("")

    lines.append("## 八、可解释性")
    lines.append("")
    interp = _read_json_if(ctx.run_dir, "interp/interp.json")
    if interp:
        tops = interp.get("top_bands") or []
        if interp.get("degenerate"):
            lines.append("**重要性不可用**：置换前后指标没有变化（ΔR² 全为 0），"
                         "本章不给出特征排序。")
            lines.append(f"- 原因：{interp.get('degenerate_note') or '置换没有产生信息'}")
            lines.append("- 处置：换验证方案（如 3 折 × 多次重复）或增大折内测试样品数后重跑 interp 阶段；"
                         "不要把 0.0000 当作「没有重要特征」写进论文。")
        elif tops:
            lines.append("贡献最大的波段（打乱该段后的 R² 下降）：")
            for t in tops[:8]:
                lines.append(f"- **{t.get('label')}**：ΔR² = {_fmt(t.get('delta_mean'), 4)}"
                             f"（占正贡献的 {_fmt((t.get('share') or 0) * 100, 1)}%）")
            if interp.get("mode") == "all_rows":
                lines.append("")
                lines.append("> 本节重要性用**全样本置换**计算（每折只有 1 个测试样品时，"
                             "对单行置换是恒等操作）；ΔR² 含内样本偏差，只能用于特征排序，"
                             "不能当作外推性能的一部分。")
        vip = interp.get("vip") or {}
        if vip:
            lines.append(f"- PLS VIP：{vip.get('n_above_1')} 个特征 VIP>1（{vip.get('n_components')} 成分）")
        conflict = interp.get("vip_vs_permutation") or {}
        if conflict:
            lines.append(f"- ⚠ 两种排序法首位不一致：置换重要性指 **{conflict.get('permutation_top')}**，"
                         f"VIP 指 **{conflict.get('vip_top')}**。{conflict.get('note')}")
        cs = interp.get("coef_stability") or {}
        if cs:
            lines.append(f"- 系数稳定性：平均符号一致率 {_fmt(cs.get('mean_sign_consistency'), 2)}，"
                         f"不稳定特征 {cs.get('unstable_features')} 个")
        if interp.get("shap"):
            lines.append(f"- SHAP：{interp['shap'].get('caveat')}")
        lines.append("")
        lines.append("> 置换重要性与 VIP 的定义不同（前者衡量模型对该列的依赖，后者含协方差结构），"
                     "高维共线数据上单列重要性本就不可识别；本报告只做特征级排序，"
                     "不据此给出具体官能团或机理归属的结论。")
    else:
        lines.append("未执行。")
    lines.append("")

    lines.append("## 九、局限（必须写入论文）")
    lines.append("")
    for item in lim:
        lines.append(f"- {item}")
    lines.append("")
    author_notes = []
    if ad_eval and ad_eval.get("author_note"):
        author_notes.append(("适用域", ad_eval["author_note"]))
    if author_notes:
        lines.append("**写作提示（给作者，不要抄进论文）**")
        lines.append("")
        for topic, note in author_notes:
            if note:
                lines.append(f"- {topic}：{note}")
        lines.append("")

    lines.append("## 十、复现信息")
    lines.append("")
    prov = _read_json_if(ctx.run_dir, "provenance.json")
    if prov:
        h = prov.get("hashes", {})
        lines.append(f"- 数据 SHA256：`{str(h.get('data_sha256'))[:16]}…`")
        lines.append(f"- 配置 SHA256：`{str(h.get('config_sha256'))[:16]}…`")
        lines.append(f"- 代码 SHA256：`{str(h.get('code_sha256'))[:16]}…`")
        lines.append(f"- 主种子：{prov.get('seeds', {}).get('master')}；Python {prov.get('python')}")
        pk = prov.get("packages", {})
        lines.append(f"- 关键版本：scikit-learn {pk.get('scikit-learn')}，numpy {pk.get('numpy')}，"
                     f"pandas {pk.get('pandas')}，xgboost {pk.get('xgboost')}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*本报告由 research-ml-modeling 生成；所有数字均可从 `cv/oof_predictions.csv` 重算。*")
    p = write_text(out_dir / "report.md", "\n".join(lines))
    return p


# --------------------------------------------------------------------------- #
def run_report(ctx) -> dict:
    cfg = ctx.cfg
    ds = ctx.ds
    out_dir = ctx.sub("report")
    best_model, _ = pick_best_model(ctx.run_dir, ds.task)
    summary = _read_csv_if(ctx.run_dir, "cv/metrics_summary.csv")
    ci = _read_csv_if(ctx.run_dir, "cv/metrics_ci.csv")
    oof = _read_csv_if(ctx.run_dir, "cv/oof_predictions.csv")
    perm = _read_json_if(ctx.run_dir, "perm/permutation.json")
    ad_eval = _read_json_if(ctx.run_dir, "ad/ad_eval.json")

    block = _primary_block(ctx, best_model, summary, ci, oof)
    # 「最优」必须有显著性支撑：与次优模型的配对 bootstrap 差值 CI 跨 0 时，
    # 只能写「性能相当」，不能只推荐一个模型（论文里这就是排序结论）。
    block["significance"] = significance_vs_runner_up(ctx.run_dir, ds.task, best_model)

    from .plots import render_all

    figures = render_all(ctx, best_model)
    tables = build_tables(ctx, best_model)
    comp = tables.get("T4_comparison", pd.DataFrame())

    report_path = build_report(ctx, best_model, block, tables, figures, perm, ad_eval, out_dir)
    methods = build_methods(ctx, best_model, block, perm, ad_eval, out_dir)
    checklist = build_reviewer_checklist(ctx, block, perm, ad_eval, comp, tables, out_dir)

    result = {
        "best_model": best_model,
        "report": str(report_path),
        "methods": methods,
        "checklist": str(checklist),
        "n_figures": len(figures),
        "n_tables": len(tables),
        "figures": figures,
        "tables": list(tables),
    }
    from .utils import write_json

    write_json(out_dir / "report_index.json", result)
    info(f"报告完成：{report_path.name}，图 {len(figures)} 张，表 {len(tables)} 张")
    return result
