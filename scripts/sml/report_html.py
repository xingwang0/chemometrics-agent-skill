# Copyright 2026 The research-ml-modeling Authors
# SPDX-License-Identifier: Apache-2.0
"""单文件响应式 HTML 报告。

设计约束（每一条都对应一类具体的误读）
------------------------------------
* **零外链、零 JS**：CSS 全部内联。审稿系统、离线机器、邮件附件里都能原样打开；
  一旦依赖 CDN 字体或 ECharts，三年后打开就是白屏。
* **图必须引用原始 PNG**：``src="../figures/F1_*.png"``，不缩小、不转 base64。
  页面上的图和投稿用的图必须是同一个字节，否则"图文不符"根本查不出来。
* **每个数字带 ``data-raw``**：悬停显示全精度原值与来源文件。展示层只负责让人读懂，
  不做唯一真源；要复算请回 ``cv/oof_predictions.csv``。
* **状态不靠颜色**：徽标同时给符号 + 文字，色盲与黑白打印都不丢信息。
* **正文复用**：``_best_line`` / ``_limitations`` 来自 ``report.build_report``，
  不在这里重写一遍——两套措辞各自演化，半年后必然互相矛盾。
"""

from __future__ import annotations

import html
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

from . import present
from .utils import as_path, ensure_dir, info, read_json, write_text

__all__ = ["build_html"]

_CSS = """
:root{--ink:#1a1a1a;--muted:#5b5b5b;--line:#dcdcdc;--bg:#f6f6f4;--card:#fff;
--accent:#185FA5;--warn:#A32D2D;--caution:#854F0B;--ok:#0F6E56}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-size:15px;line-height:1.65;
font-family:-apple-system,"Segoe UI","Microsoft YaHei","Source Han Sans SC",system-ui,sans-serif}
.wrap{max-width:1200px;margin:0 auto;padding:0 24px 72px}
a{color:var(--accent)}
header{padding:36px 0 14px;border-bottom:1px solid var(--line)}
h1{font-size:26px;font-weight:600;margin:0 0 6px}
h2{font-size:19px;font-weight:600;margin:44px 0 12px;padding-bottom:6px;
border-bottom:1px solid var(--line)}
h3{font-size:15px;font-weight:600;margin:22px 0 8px}
p,li{max-width:80ch}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0 0}
.chip{font:12px/1.4 ui-monospace,Consolas,monospace;background:var(--card);
border:1px solid var(--line);border-radius:999px;padding:3px 10px;color:var(--muted)}
.grid{display:grid;grid-template-columns:208px minmax(0,1fr);gap:40px;align-items:start;
margin-top:22px}
nav{position:sticky;top:20px;font-size:13px}
nav ol{list-style:none;margin:0;padding:0}
nav a{display:block;padding:3px 0;color:var(--muted);text-decoration:none}
nav a:hover{color:var(--accent)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(148px,1fr));gap:12px;
margin:16px 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.kpi .k{font-size:12px;color:var(--muted)}
.kpi .v{font-size:22px;font-weight:600;font-variant-numeric:tabular-nums;margin-top:2px}
.kpi .u{font-size:12px;color:var(--muted)}
.banner{border-left:4px solid var(--warn);background:#FCEBEB;padding:12px 16px;
border-radius:0 8px 8px 0;margin:14px 0}
.banner.caution{border-color:var(--caution);background:#FAEEDA}
.banner.info{border-color:var(--accent);background:#E6F1FB}
.banner ul{margin:6px 0 0;padding-left:20px}
.badge{display:inline-block;font-size:12px;border-radius:999px;padding:1px 8px;
border:1px solid;margin:0 6px 6px 0;white-space:nowrap}
.badge .mark{font-weight:600;margin-right:4px}
.sev-info{color:var(--accent);border-color:#85B7EB;background:#E6F1FB}
.sev-caution{color:var(--caution);border-color:#FAC775;background:#FAEEDA}
.sev-blocking{color:var(--warn);border-color:#F09595;background:#FCEBEB}
.tw{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--card);
margin:14px 0}
table{border-collapse:collapse;width:100%;font-size:13px}
caption{caption-side:top;text-align:left;padding:10px 12px;font-size:13px;color:var(--muted);
border-bottom:1px solid var(--line)}
th,td{padding:7px 10px;border-bottom:1px solid #eee;text-align:left;white-space:nowrap}
thead th{background:#fafaf8;font-weight:600;position:sticky;top:0;border-bottom:1px solid var(--line)}
td.num{text-align:right;font-variant-numeric:tabular-nums}
td.txt{white-space:normal;min-width:9em;max-width:44ch}
td:first-child,th:first-child{position:sticky;left:0;background:var(--card);z-index:1}
thead th:first-child{background:#fafaf8;z-index:2}
tbody tr:hover td{background:#fbfbf9}
figure{margin:18px 0;background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:14px}
figure img{width:100%;height:auto;display:block}
figcaption{font-size:13px;color:var(--muted);margin-top:10px;line-height:1.55}
.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}
.kv{width:100%;font-size:13px}
.kv th{width:32%;color:var(--muted);font-weight:400;white-space:normal;vertical-align:top}
.kv td{white-space:normal}
.note{font-size:13px;color:var(--muted)}
.unit-missing{color:var(--warn);font-weight:600}
code{font:12px ui-monospace,Consolas,monospace;background:#f1efe8;padding:1px 5px;
border-radius:4px}
footer{margin-top:56px;padding-top:20px;border-top:1px solid var(--line);font-size:13px;
color:var(--muted)}
.skip{position:absolute;left:-9999px}
@media(max-width:1099px){
.grid{grid-template-columns:1fr;gap:0}
nav{position:static;overflow-x:auto;white-space:nowrap;padding:10px 0;
border-bottom:1px solid var(--line)}
nav ol{display:flex;gap:16px}
nav a{padding:0}
}
@media(max-width:760px){
body{font-size:14px}
.wrap{padding:0 16px 48px}
h1{font-size:21px}h2{font-size:17px}
.kpis{grid-template-columns:repeat(auto-fit,minmax(118px,1fr))}
.kpi .v{font-size:18px}
figure{padding:10px}
}
@media print{
@page{size:A4;margin:14mm 12mm}
body{background:#fff;font-size:10.5pt}
nav,.skip{display:none}
.grid{display:block}
.wrap{max-width:none;padding:0}
h2{break-after:avoid}
tr,figure,.kpi{break-inside:avoid}
thead{display:table-header-group}
.tw{overflow:visible;border:none}
th,td{padding:4px 6px;font-size:9pt}
td:first-child,th:first-child{position:static}
figure{border:none;padding:0}
a{color:inherit;text-decoration:none}
}
"""

_COL_ZH = {
    "model": "模型", "target": "目标", "prep_mix": "逐折选中的预处理",
    "metric_basis": "口径", "n_folds_ok": "有效折数", "n_folds_degenerate": "退化折数",
    "rank": "排名", "rmse_gap_to_best": "与最优的 RMSE 差", "n_repeats": "重复次数",
    "n_oof_rows": "OOF 行数", "prep": "预处理", "status": "状态", "best_params": "最优超参",
    "method": "方法", "key": "标识", "flagged": "越界样品数", "rate": "越界样品比例",
    "flagged_rows": "越界 OOF 行数", "threshold": "阈值",
    "n": "有效行数", "n_train": "训练行数", "n_test": "测试行数",
    "n_train_groups": "训练样品数", "fold": "折", "repeat": "重复", "scoring": "评分函数",
    "proba_kind": "概率类型", "pred_n_extreme": "越界预测数",
    "best_inner_score": "内层最优分", "value_a": "模型 A 值", "value_b": "模型 B 值",
    "model_a": "模型 A", "model_b": "模型 B", "metric": "指标", "better": "更优者",
    "significant": "显著", "m_family": "家族大小", "n_pairs": "配对数",
    "metric_primary": "主指标", "level": "级别", "title": "问题", "detail": "依据",
    "action": "建议动作", "feature": "特征", "vip": "VIP", "above_1": "VIP>1",
    "band": "波段", "label": "标签", "center": "中心波数", "delta_mean": "ΔR² 均值",
    "delta_sd": "ΔR² 标准差", "kind": "类型", "row_id": "行标识", "group": "样品",
    "batch": "批次", "t2": "T²", "q": "Q 残差", "knn_dist": "kNN 距离",
    "leverage": "杠杆值", "ad_votes": "投票数", "ad_flag": "任一越界", "ad_majority": "多数越界",
}
_SUFFIX_ZH = (("_fold_mean", "（折均值）"), ("_mean", "（均值）"), ("_sd", "（SD）"),
              ("_ci_low", "（95%CI 下限）"), ("_ci_high", "（95%CI 上限）"))


# --------------------------------------------------------------------------- #
# 基础渲染
# --------------------------------------------------------------------------- #
def _esc(v: Any) -> str:
    return html.escape("" if v is None else str(v), quote=True)


def _text(v: Any) -> str:
    """脏值边界：None / NaN / pandas NA 一律压成破折号。"""
    if v is None:
        return "—"
    if isinstance(v, float) and not math.isfinite(v):
        return "—"
    s = str(v).strip()
    return "—" if s in {"", "nan", "NaN", "None", "<NA>", "NaT", "NaT"} else s


def _is_num(v: Any) -> bool:
    if isinstance(v, bool) or isinstance(v, str):
        return False
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _col_label(col: Any) -> str:
    name = str(col)
    if name in _COL_ZH:
        return _COL_ZH[name]
    if any(ord(ch) > 127 for ch in name):
        # 已经是中文列名（"域内RMSE"、"域外样品比例"），直接照用。
        # 走 spec 的 label 会把它退化成"RMSE"/"比例"，丢掉"域内/域外"这个关键限定。
        return name
    spec = present.spec_for_column(name)
    base = spec.label_zh if spec else name
    for suf, zh in _SUFFIX_ZH:
        if name.endswith(suf):
            return f"{base}{zh}"
    return base


def _col_unit(col: Any, unit: Any) -> str:
    spec = present.spec_for_column(col)
    if spec is None or spec.unit is not None:
        return ""            # 显式无量纲：不标
    return present.unit_suffix(unit, "html")


def _td(v: Any, col: Any, unit: Any) -> str:
    spec = present.spec_for_column(col)
    if not _is_num(v):
        if isinstance(v, bool):
            return f"<td>{'是' if v else '否'}</td>"
        # 长文本列（审计发现的「依据」「建议动作」）必须允许换行，
        # 否则 nowrap 会把一张表拉成两米宽，读者只能左右拖。
        return f'<td class="txt">{_esc(_text(v))}</td>'
    raw = float(v)
    if not math.isfinite(raw):
        # nan 在表里必须写成破折号。落成 "nan" 会被当成一个字符值渲染出来，
        # 而且 grep 不到任何报错——读者只会看到一格 "nan" 混在数字中间。
        return f'<td class="num">{present.MISSING}</td>'
    if spec is not None:
        text = present.fmt(spec.id, raw)
        field = spec.id
    else:
        text = str(int(raw)) if raw.is_integer() else f"{raw:.4g}"
        field = str(col)
    tip = f"原始值 {raw!r}"
    if spec is not None and spec.unit is None:
        u = present.unit_suffix(unit, "md")
        tip += f" · 单位 {u}"
    tip += f" · 字段 {field}"
    return (f'<td class="num" data-field="{_esc(field)}" data-raw="{_esc(repr(raw))}" '
            f'title="{_esc(tip)}">{_esc(text)}</td>')


def _table_html(df, caption: str, note: str = "", unit: Any = None,
                max_rows: int | None = None, first_col_strong: bool = False) -> str:
    if df is None or getattr(df, "empty", True):
        return ""
    rows = df.head(max_rows) if max_rows else df
    cols = list(rows.columns)
    head = "".join(
        f'<th scope="col">{_esc(_col_label(c))}{_col_unit(c, unit)}</th>' for c in cols)
    body = []
    for _, r in rows.iterrows():
        cells = "".join(_td(r[c], c, unit) for c in cols)
        body.append(f"<tr>{cells}</tr>")
    note_html = f'<div class="note">{_esc(note)}</div>' if note else ""
    return (f'<div class="tw"><table><caption>{_esc(caption)}</caption>'
            f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>{note_html}")


def _kv(rows: list[tuple[str, str]]) -> str:
    body = "".join(
        f'<tr><th scope="row">{_esc(k)}</th><td>{v}</td></tr>' for k, v in rows)
    return f'<table class="kv"><tbody>{body}</tbody></table>'


def _figure(code: str, figures: dict, note: str = "") -> str:
    paths = (figures or {}).get(code) or []
    png = next((p for p in paths if str(p).lower().endswith(".png")), None)
    if not png:
        return ""
    rel = "../figures/" + Path(str(png)).name
    caption = _FIG_CAPTION.get(code, code)
    alt = f"{code} {caption}"
    if note:
        alt += f"；{note}"
    return (f'<figure><img src="{_esc(rel)}" alt="{_esc(alt)}" loading="lazy">'
            f"<figcaption><b>{code}</b> {_esc(caption)}"
            + (f'<br><span class="note">{_esc(note)}</span>' if note else "")
            + "</figcaption></figure>")


_FIG_CAPTION = {
    "F1": "OOF 预测-实测散点（分类为 ROC）",
    "F2": "残差图（仅回归）",
    "F3": "交叉验证分布（逐折）",
    "F4": "Williams plot（适用域，纵轴未按杠杆校正）",
    "F5": "校准 / 一致性（回归=Bland-Altman）",
    "F6": "Y-randomization 零分布",
    "F7": "重要性-波数映射",
    "F8": "模型比较（点=均值，细线=±SD，粗线=95% CI）",
}


def _md_inline(text: Any) -> str:
    """把报告正文里那点 Markdown（``**粗体**``、`` `代码` ``、行首 ``-``）转成 HTML。

    这些句子由 ``report.build_report`` 产出、与 ``report.md`` 共用同一份文本。
    在 HTML 里原样显示 ``**ridge**`` 会被读者当成排版错误，而复制一份措辞到渲染层
    又会让两版慢慢说不一样的话——所以只做最小转换，不重写文字。
    """
    s = _esc(re.sub(r"^[-*]\s+", "", str(text or "").strip()))
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    return s


def _badges(badges: list[present.Badge], target: str = "html") -> str:
    out = []
    for b in badges:
        body = (f'<span class="mark">{_esc(b.mark)}</span>{_esc(b.label_zh)}'
                if target == "html" else f"{b.mark} {b.label_zh}")
        if target == "html":
            out.append(f'<span class="badge sev-{b.severity}" title="{_esc(b.explain)}">{body}</span>')
        else:
            out.append(body)
    return "".join(out) if target == "html" else "、".join(out)


# --------------------------------------------------------------------------- #
# 各部分
# --------------------------------------------------------------------------- #
def _evidence(block: dict, ds, perm: dict, ad_eval: dict) -> present.Evidence:
    return present.evidence_for(block, ds, perm, ad_eval)


def _kpis(block: dict, task: str, unit: Any) -> str:
    keys = (("r2", "R²", False), ("rmse", "RMSE", True), ("mae", "MAE", True), ("rpd", "RPD", False)) \
        if task == "regression" else \
        (("auc", "AUROC", False), ("balanced_accuracy", "平衡准确率", False),
         ("f1_macro", "F1(macro)", False), ("mcc", "MCC", False))
    ci = block.get("ci") or {}
    cards = []
    for key, label, has_unit in keys:
        val = present.fmt(key, block.get(key))
        sub = ""
        if has_unit:
            sub = present.unit_suffix(unit, "html")
        if key in ci:
            lo, hi = ci[key]
            sub = f"{sub} 95%CI {present.fmt_ci(key, lo, hi)}".strip()
        cards.append(f'<div class="kpi"><div class="k">{_esc(label)}</div>'
                     f'<div class="v">{_esc(val)}</div>'
                     f'<div class="u">{sub}</div></div>')
    return f'<div class="kpis">{"".join(cards)}</div>'


def _alerts(ctx, block, audit: dict, cv_summary: dict, findings: list,
            unit: Any, qc: dict) -> str:
    items: list[tuple[str, str, list[str]]] = []
    if findings:
        errs = [f for f in findings if f.level == "error"]
        if errs:
            items.append(("warn", "措辞合规：检测到数据不支持的强断言",
                          [f"「{f.snippet}」→ {f.advice}" for f in errs]))
    if present.normalize_unit(unit)[1] == "missing" and ctx.ds.task == "regression":
        items.append(("caution", "目标单位未指定",
                      ["RMSE / MAE / bias 因此没有量纲。请在配置里设置 "
                       "<code>data.target_unit</code>（如 <code>%</code>、<code>mg/kg</code>），"
                       "否则读者无法判断模型能否用于实际判定。"]))
    failed = (cv_summary or {}).get("failed_models") or {}
    if failed:
        lines = []
        for name, meta in failed.items():
            n_total = meta.get("n_total")
            scope = (f"全部 {n_total} 折失败" if meta.get("all_failed")
                     else f"{meta.get('n_failed')}/{n_total} 折失败")
            lines.append(f"<code>{_esc(name)}</code>：{_esc(scope)}；首个错误 "
                         f"<code>{_esc(str(meta.get('first_error'))[:120])}</code>")
        items.append(("warn", "有模型成批失败，已排除在结果之外", lines))
    crit = [f for f in (audit.get("findings") or []) if f.get("level") == "critical"]
    if crit:
        items.append(("warn", "数据审计存在严重问题，建模结论需谨慎",
                      [f"<b>{_esc(f.get('title'))}</b>：{_esc(f.get('detail'))}" for f in crit]))
    if qc:
        items.append(("caution", "图件完整性检查未通过",
                      [f"<code>{_esc(k)}</code>：{_esc('；'.join(v))}" for k, v in qc.items()]))
    if str(block.get("basis", "fold")) == "pooled":
        items.append(("info", "本 run 的主指标为池化 OOF 口径",
                      ["有效折不足 3 个，折均值不可引用。图 F1/F2/F5 的统计量与表 T2 的"
                       "池化列同口径，但**不等于** <code>*_mean</code> 列。"]))
    if not items:
        return ""
    out = []
    for level, title, lines in items:
        cls = "" if level == "warn" else f" {level}"
        body = "".join(f"<li>{ln}</li>" for ln in lines)
        out.append(f'<div class="banner{cls}"><b>{_esc(title)}</b><ul>{body}</ul></div>')
    return "".join(out)


def _section_repro(ctx, prov: dict, unit: Any) -> str:
    h = (prov or {}).get("hashes") or {}
    seeds = (prov or {}).get("seeds") or {}
    pk = (prov or {}).get("packages") or {}
    unit_md = present.unit_suffix(unit, "md")
    rows = [
        ("数据 SHA256", f"<code>{_esc(str(h.get('data_sha256'))[:32])}…</code>"),
        ("配置 SHA256", f"<code>{_esc(str(h.get('config_sha256'))[:32])}…</code>"),
        ("代码 SHA256", f"<code>{_esc(str(h.get('code_sha256'))[:32])}…</code>"),
        ("主种子", _esc(seeds.get("master"))),
        ("Python", _esc((prov or {}).get("python_full") or (prov or {}).get("python"))),
        ("scikit-learn / numpy / pandas",
         f"{_esc(pk.get('scikit-learn'))} / {_esc(pk.get('numpy'))} / {_esc(pk.get('pandas'))}"),
        ("目标单位", ("<span class=\"unit-missing\">单位未指定</span>"
                  if present.normalize_unit(unit)[1] == "missing" else _esc(unit_md))),
    ]
    return _kv(rows)


def _section_units(ctx, unit: Any, block: dict) -> str:
    ds = ctx.ds
    unit_state = present.normalize_unit(unit)[1]
    u_md = present.unit_suffix(unit, "md")
    html = ["<p>本页所有数值都由 <code>sml/present.py</code> 的字段规格渲染："
            "同一个量在表、图、正文里只有一种精度与写法。CSV / XLSX 里的单元格仍是"
            "<b>未舍入的原始 float</b>，页面上看到的是显示层的舍入结果——"
            "两者不是同一层数据，复算请回到 <code>cv/</code> 下的 CSV。</p>"]
    basis = "池化 OOF（有效折不足，<code>*_mean</code> 不可引用）" if str(
        block.get("basis", "fold")) == "pooled" else \
        "折均值 ± SD（表 T2）与池化 OOF（图 F1/F2/F5）并存，引用时须与所引对象一致"
    rows = [
        ("目标变量", f"<code>{_esc(ds.target_name)}</code>（任务 {_esc(ds.task)}）"),
        ("目标单位", ("<span class=\"unit-missing\">单位未指定</span>"
                  if unit_state == "missing" else _esc(u_md))),
        ("独立样品数", f"{ds.n_groups}（论文必须报这个数，不是 {ds.n_samples} 条测量）"),
        ("口径", basis),
    ]
    html.append(_kv(rows))
    html.append("<h3>数字到原始文件的对应关系</h3>")
    links = [
        ("性能汇总（逐模型）", "../cv/metrics_summary.csv"),
        ("逐折明细", "../cv/fold_results.csv"),
        ("置信区间", "../cv/metrics_ci.csv"),
        ("模型两两比较", "../cv/comparison.csv"),
        ("逐行 OOF 预测（唯一真源）", "../cv/oof_predictions.csv"),
        ("置换检验", "../perm/permutation.json"),
        ("适用域评估", "../ad/ad_eval.json"),
        ("数据审计", "../audit/audit.md"),
    ]
    avail = [(k, p) for k, p in links if (ctx.run_dir / p.replace("../", "")).exists()]
    if avail:
        rows = [(k, f'<a href="{_esc(p)}">{_esc(p)}</a>') for k, p in avail]
        html.append(_kv(rows))
    else:
        html.append('<p class="note">配套文件不在本目录下（可能只拷走了 report 文件夹），'
                    "因此未生成溯源链接。</p>")
    return "".join(html)


def _section_tripod(ctx) -> str:
    try:
        from .standards import render_section
        return render_section(ctx)
    except Exception as exc:  # noqa: BLE001
        return (f'<p class="note">TRIPOD+AI 清单映射未启用'
                f"（{_esc(type(exc).__name__)}）。见 <code>references/standards-mapping.md</code>。</p>")


def _generated_at(prov: dict) -> str:
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if epoch and str(epoch).isdigit():
        return time.strftime("%Y-%m-%d %H:%M", time.gmtime(int(epoch)))
    created = str((prov or {}).get("created_at") or "")
    if created:
        return created.replace("T", " ")[:16]
    return time.strftime("%Y-%m-%d %H:%M")


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def build_html(ctx, best_model: str, tables: dict, block: dict, *,
               perm: dict | None = None, ad_eval: dict | None = None,
               figures: dict | None = None, out_dir: Path | None = None) -> Path:
    """生成 ``report/report.html``。失败不应阻断报告阶段（调用方负责兜住异常）。"""
    cfg = ctx.cfg
    ds = ctx.ds
    run = ctx.run_dir
    out = ensure_dir(out_dir or ctx.sub("report"))
    prov = read_json(run / "provenance.json") if (run / "provenance.json").exists() else {}
    audit = read_json(run / "audit/audit.json") if (run / "audit/audit.json").exists() else {}
    cv_summary = (read_json(run / "cv/cv_summary.json")
                  if (run / "cv/cv_summary.json").exists() else {})
    qc_p = run / "figures/figure_qc.json"
    qc = read_json(qc_p) if qc_p.exists() else {}

    unit = getattr(ds, "target_unit", None)
    unit_known = present.normalize_unit(unit)[1] != "missing"
    failed = (cv_summary or {}).get("failed_models") or {}
    badges = present.badges_for(block=block, perm=perm, ad_eval=ad_eval,
                                unit_known=unit_known, failed_models=failed)

    best_line = str(block.get("_best_line") or "")
    limitations = [str(x) for x in (block.get("_limitations") or [])]
    findings = present.guard_text("\n".join([best_line] + limitations),
                                  _evidence(block, ds, perm or {}, ad_eval or {}),
                                  strict=bool(cfg.get("report", {}).get("strict_text", False)))

    sections: list[tuple[str, str]] = []

    # 一、结论摘要
    body = [_kpis(block, ds.task, unit)]
    if best_line:
        body.append(f"<p>{_md_inline(best_line)}</p>")
    if badges:
        body.append(f"<p>{_badges(badges)}</p>")
    if perm and not perm.get("error") and perm.get("conclusion"):
        body.append(f"<p><b>抗噪声检验</b>：{_md_inline(perm.get('conclusion'))}</p>")
    if ad_eval and ad_eval.get("conclusion"):
        body.append(f"<p><b>适用域</b>：{_md_inline(ad_eval.get('conclusion'))}</p>")
    sections.append(("summary", "".join(body)))

    # 二、数据与任务
    s = audit.get("summary", {}) or {}
    gps = ds.groups_per_sample()
    reps = (str(int(gps.min())) if int(gps.min()) == int(gps.max())
            else f"{int(gps.min())}–{int(gps.max())}")
    rows = [
        ("数据文件", f"<code>{_esc(Path(str(ds.source_path)).name if ds.source_path else '—')}</code>"),
        ("规模", f"{ds.n_samples} 行 × {ds.n_features} 特征"),
        ("独立样品（group）", f"{ds.n_groups}"),
        ("每样品测量次数", _esc(reps)),
        ("批次", _esc(len(set(map(str, ds.batch))) if ds.batch is not None else "未提供")),
        ("任务", f"{_esc(ds.task)}（目标 <code>{_esc(ds.target_name)}</code>）"),
        ("条件数", _esc(present.fmt("cond_number", s.get("cond_number"))
                     if isinstance(s.get("cond_number"), (int, float)) else _text(s.get("cond_number")))),
        ("有效秩", _esc(s.get("effective_rank", "—"))),
    ]
    body = [_kv(rows)]
    findings_a = audit.get("findings") or []
    n_crit = len([f for f in findings_a if f.get("level") == "critical"])
    n_warn = len([f for f in findings_a if f.get("level") == "warn"])
    body.append(f'<p class="note">严重问题 {n_crit} 项，警告 {n_warn} 项；'
                f'详见 <a href="../audit/audit.md">audit/audit.md</a>。</p>')
    t1b = tables.get("T1b_audit_findings")
    if t1b is not None and not t1b.empty:
        body.append(_table_html(t1b, "表 T1b　数据审计发现", "级别按严重度排序；critical 必须先修复再建模",
                                max_rows=20))
    sections.append(("data", "".join(body)))

    # 三、验证方案
    sp = cfg["split"]
    plan = read_json(run / "cv/cv_summary.json") if (run / "cv/cv_summary.json").exists() else {}
    rows = [
        ("策略", f"<code>{_esc(sp['strategy'])}</code>"),
        ("分组键", f"<code>{_esc(cfg['data'].get('id_col') or '特征指纹推断')}</code>"),
        ("外层", f"{sp['outer']['n_splits']} 折分组 CV × 重复 {sp['repeats']} 次"),
        ("内层", f"{sp['inner']['n_splits']} 折分组 CV（调参 + 预处理选择）"),
        ("总拟合次数", _esc(plan.get("total_fit_calls", "—"))),
    ]
    sections.append(("design", _kv(rows) + "<p class=\"note\">预处理链作为超参数逐折选择，"
                     "测试折不参与任何拟合——这是本管线与「先全量预处理再切分」的关键区别。</p>"))

    # 四、性能结果
    t2 = tables.get("T2_performance")
    body = []
    if t2 is not None and not t2.empty:
        body.append(_table_html(
            t2, "表 T2　各模型性能（按 (target, model) 聚合）",
            "† = 该行有效折不足，*_mean 不可引用，请看同行的池化列。"
            "口径列 metric_basis / n_folds_ok 必须与性能一起读。", unit=unit))
    body.append(_figure("F1", figures, "统计量为 OOF 池化口径，与表 T2 的repeat 均值口径不可混引"))
    body.append(_figure("F3", figures, "逐折分布：箱体窄且散点集中才说明模型稳定"))
    t3 = tables.get("T3_per_fold")
    if t3 is not None and not t3.empty:
        body.append(_table_html(t3, "表 T3　逐折明细（前 30 行）",
                                f"共 {len(t3)} 行；审稿人要原始数据时给这份 CSV", unit=unit,
                                max_rows=30))
    sections.append(("performance", "".join(body)))

    # 五、模型比较
    t4 = tables.get("T4_comparison")
    body = []
    if t4 is not None and not t4.empty:
        body.append(_table_html(t4, "表 T4　模型两两配对 bootstrap 比较（前 20 行）",
                                "判断显著性以 p_BH 为准；p_raw 只用于说明家族大小。", unit=unit,
                                max_rows=20))
    sig = block.get("significance") or {}
    if sig.get("available"):
        body.append(f"<p>与次优模型 <b>{_esc(sig.get('runner_up'))}</b> 的差值为 "
                    f"{_esc(present.fmt_ci('delta', sig.get('ci_low'), sig.get('ci_high')))}，"
                    f"BH 校正 p = {_esc(present.fmt_p(sig.get('p_bh')))}"
                    f"（家族 m = {_esc(sig.get('m_family'))}）。</p>")
    else:
        body.append(f'<p class="note">未做显著性检验：{_esc(sig.get("reason") or "缺少对比结果")}。'
                    "在这种情况下不得声称某个模型优于其它模型。</p>")
    body.append(_figure("F8", figures, "点=repeat 均值，细线=±SD，粗线=95% CI"))
    sections.append(("compare", "".join(body)))

    # 六、是否在拟合噪声
    body = []
    if perm and not perm.get("error"):
        rows = [
            ("观测值", _esc(present.fmt(str(perm.get("metric") or "r2"), perm.get("observed")))),
            ("置换次数", _esc(perm.get("n_perm"))),
            ("p 值", _esc(present.fmt_p(perm.get("p_value")))),
            ("模式", "固定超参（保守）" if str(perm.get("mode")) == "fast" else "每轮重搜超参"),
            ("结论", _esc(perm.get("conclusion"))),
        ]
        body.append(_kv(rows))
        body.append(_figure("F6", figures, "观测值应落在零分布右尾之外"))
    else:
        body.append('<p class="note">本 run 未执行置换检验，因此不能声称模型关系超出随机水平。</p>')
    sections.append(("noise", "".join(body)))

    # 七、适用域
    body = []
    t5 = tables.get("T5_applicability_domain")
    if t5 is not None and not t5.empty:
        body.append(_table_html(
            t5, "表 T5　域内外性能对比（只报独立样品数）",
            "行口径（OOF 行数）仅为备查，不得写进论文正文。", unit=unit))
    t5b = tables.get("T5b_ad_methods")
    if t5b is not None and not t5b.empty:
        body.append(_table_html(t5b, "表 T5b　各判定方法的越界率与阈值", "退化方法的阈值不可能被越过，已退出投票"))
    body.append(_figure("F4", figures, "纵轴为标准化残差（未按 √(1−h) 校正），仅作诊断"))
    if ad_eval and ad_eval.get("author_note"):
        body.append(f'<div class="banner info"><b>写作提示（给作者，不要抄进论文）</b>'
                    f'<p>{_md_inline(ad_eval.get("author_note"))}</p></div>')
    sections.append(("domain", "".join(body)))

    # 八、可解释性
    body = []
    t6 = tables.get("T6_band_importance")
    if t6 is not None and not t6.empty:
        body.append(_table_html(t6, "表 T6　波段级置换重要性", "ΔR² 为打乱该波段后的性能下降",
                                max_rows=20))
    t6b = tables.get("T6b_vip")
    if t6b is not None and not t6b.empty:
        body.append(_table_html(t6b, "表 T6b　VIP（前 100 个特征）",
                                "VIP > 1 只是经验阈值，不等于统计显著", max_rows=20))
    body.append(_figure("F7", figures, "柱=重要性，底图=平均谱；两者共享波数轴"))
    sections.append(("explain", "".join(body)))

    # 九、局限
    if limitations:
        items = "".join(f"<li>{_md_inline(x)}</li>" for x in limitations)
        sections.append(("limits", f"<ul>{items}</ul>"))
    else:
        sections.append(("limits", '<p class="note">未生成局限清单（上游结果不完整）。</p>'))

    # 十、复现信息
    sections.append(("repro", _section_repro(ctx, prov, unit)))

    # 十一～十四
    gallery = "".join(_figure(c, figures) for c in
                      ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"))
    sections.append(("gallery", f'<div class="gallery">{gallery}</div>'
                     if gallery else '<p class="note">本 run 没有可用图。</p>'))
    sections.append(("legend", present.legend("html")))
    sections.append(("units", _section_units(ctx, unit, block)))
    sections.append(("tripod", _section_tripod(ctx)))

    order = ["summary", "data", "design", "performance", "compare", "noise",
             "domain", "explain", "limits", "repro"]

    nav = "".join(
        f'<li><a href="#s-{sid}">{i + 1}. {_esc(present.CHAPTERS[i][1])}</a></li>'
        for i, sid in enumerate(order))
    nav += ('<li><a href="#s-gallery">11. 图表</a></li><li><a href="#s-legend">12. 徽标</a></li>'
            '<li><a href="#s-units">13. 单位与口径</a></li><li><a href="#s-tripod">14. 报告清单</a></li>')

    body_html = []
    for i, sid in enumerate(order):
        num, title = present.CHAPTERS[i]
        content = dict(sections)[sid]
        body_html.append(f'<section id="s-{sid}"><h2>{num}、{_esc(title)}</h2>{content}</section>')
    for sid, label in (("gallery", "十一、图表"), ("legend", "十二、徽标图例"),
                       ("units", "十三、单位、口径与可追溯性"),
                       ("tripod", "十四、报告清单对照")):
        body_html.append(f'<section id="s-{sid}"><h2>{label}</h2>{dict(sections)[sid]}</section>')

    hashes = (prov or {}).get("hashes") or {}
    title = f"预测建模报告：{ds.target_name}"
    subtitle = cfg.get("report", {}).get("title") or ""
    chips = "".join([
        f'<span class="chip">run {_esc(run.name)}</span>',
        f'<span class="chip">data {_esc(str(hashes.get("data_sha256"))[:12])}</span>',
        f'<span class="chip">code {_esc(str(hashes.get("code_sha256"))[:12])}</span>',
        f'<span class="chip">seed {_esc((prov.get("seeds") or {}).get("master"))}</span>',
        f'<span class="chip">生成 {_esc(_generated_at(prov))}</span>',
    ])

    footer = [
        "<p><b>合规声明</b>。本报告由自动化管线生成，方法学实践参考 ASTM E1655、"
        "ISO 12099/21543 与 ICH Q2(R2)，但本管线未建立独立验证集与 reference method，"
        "<b>不声明符合</b>上述标准；亦未针对监管申报、临床诊断或生产实时控制做过验证。"
        "指标、单位与样本代表性须由作者核对。</p>",
        f"<p><b>完整性</b>。图件检查 {'通过' if not qc else '未通过（见上方告警）'}；"
        f"本页不含 JavaScript、不引用任何外部资源。</p>",
        f"<p><b>可追溯</b>。数据 <code>{_esc(hashes.get('data_sha256'))}</code><br>"
        f"配置 <code>{_esc(hashes.get('config_sha256'))}</code><br>"
        f"代码 <code>{_esc(hashes.get('code_sha256'))}</code></p>",
        "<p>页内所有数值均可从 <code>cv/oof_predictions.csv</code> 重算；"
        "若对不上，先怀疑表被手工改过。生成时间取自 run 创建时间，本页不参与三哈希。</p>",
    ]

    doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>{_CSS}</style>
</head>
<body>
<a class="skip" href="#main">跳到正文</a>
<div class="wrap">
<header>
<h1>{_esc(title)}</h1>
{f'<p class="note">{_esc(subtitle)}</p>' if subtitle else ''}
<div class="chips">{chips}</div>
</header>
{_alerts(ctx, block, audit, cv_summary, findings, unit, qc)}
<div class="grid">
<nav aria-label="目录"><ol>{nav}</ol></nav>
<main id="main">
{''.join(body_html)}
<footer>{''.join(footer)}</footer>
</main>
</div>
</div>
</body>
</html>
"""
    p = write_text(out / "report.html", doc)
    info(f"HTML 报告 -> {p.name}")
    return as_path(p)
