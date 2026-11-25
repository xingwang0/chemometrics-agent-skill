# Copyright 2026 The research-ml-modeling Authors
# SPDX-License-Identifier: Apache-2.0
"""TRIPOD+AI（BMJ 2024;385:e078378）报告清单的**映射层**。

边界（很重要）
--------------
本模块**不含任何清单条目的原文或译文**。27 条条目的措辞受版权保护，并且逐条翻译
会引入"看起来像标准、实际是我方改写"的风险。条目文本放在
``assets/standards/tripod_ai_2024.json``，由人工对照原文转录；代码只负责：
把 ``evidence`` 键映射到本次 run 的产物，给出 ✓ / ⚠ / N/A 三态，并渲染。

文件缺失或 ``items`` 为空时，本模块输出"清单未安装"的占位说明——**不生成任何条目**，
也不猜测条目内容。

三态的判定口径
--------------
* ``✓`` 证据文件存在且字段非空；
* ``⚠`` 需要作者补（仪器型号、样本代表性、亚组公平性等管线无法推断的信息），
  或证据缺失；
* ``N/A`` 必须由条目自带的 ``na_reason`` 说明理由，否则按 ⚠ 处理。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .utils import as_path

__all__ = ["ITEMS_PATH", "load_items", "checklist_rows", "render_section", "write_checklist"]

ITEMS_PATH = (as_path(__file__).resolve().parent.parent.parent
              / "assets" / "standards" / "tripod_ai_2024.json")

#: evidence 键 → 相对 run 目录的文件。值为 None 表示"管线无法提供，须作者补"。
EVIDENCE_FILES: dict[str, str] = {
    "audit": "audit/audit.json",
    "protocol": "cv/cv_summary.json",
    "models": "cv/models.json",
    "performance": "cv/metrics_summary.csv",
    "ci": "cv/metrics_ci.csv",
    "comparison": "cv/comparison.csv",
    "oof": "cv/oof_predictions.csv",
    "figures": "figures",
    "perm": "perm/permutation.json",
    "ad": "ad/ad_eval.json",
    "interp": "interp/interp.json",
    "provenance": "provenance.json",
    "model_card": "model/model_card.json",
}
#: 这些键永远无法由管线提供，只能由作者补——写成清单里的人名/仪器/伦理信息。
AUTHOR_ONLY = {"instrument", "sampling", "fairness", "ppi", "registration", "ethics"}


def load_items() -> list[dict[str, Any]]:
    if not ITEMS_PATH.exists():
        return []
    try:
        data = json.loads(ITEMS_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    items = data.get("items") if isinstance(data, dict) else None
    return [i for i in (items or []) if isinstance(i, dict) and i.get("id")]


def _evidence_state(run: Path, keys: list[str]) -> tuple[str, list[str], list[str]]:
    """返回 (状态, 已找到的证据, 缺失项说明)。"""
    found: list[str] = []
    missing: list[str] = []
    author = False
    for key in keys or []:
        k = str(key)
        if k in AUTHOR_ONLY:
            author = True
            missing.append(f"{k}（须作者补）")
            continue
        rel = EVIDENCE_FILES.get(k)
        if rel is None:
            missing.append(f"{k}（未知证据键）")
            continue
        p = run / rel
        if p.exists():
            found.append(rel)
        else:
            missing.append(rel)
    if author:
        return "author", found, missing
    if found and not missing:
        return "ok", found, missing
    return ("partial" if found else "missing"), found, missing


def checklist_rows(ctx) -> list[dict[str, str]]:
    """把条目映射成 ``{id, topic, status, mark, evidence, note}`` 行。"""
    rows: list[dict[str, str]] = []
    for item in load_items():
        na_reason = str(item.get("na_reason") or "").strip()
        state, found, missing = _evidence_state(ctx.run_dir, item.get("evidence") or [])
        if na_reason:
            rows.append({"id": str(item["id"]), "topic": str(item.get("topic_zh") or ""),
                         "status": "N/A", "mark": "—", "evidence": "",
                         "note": na_reason})
            continue
        mark = {"ok": "✓", "author": "⚠", "partial": "⚠", "missing": "⚠"}[state]
        note = "" if state == "ok" else "；".join(missing)
        rows.append({"id": str(item["id"]), "topic": str(item.get("topic_zh") or ""),
                     "status": state, "mark": mark,
                     "evidence": "、".join(found), "note": note})
    return rows


def render_section(ctx) -> str:
    rows = checklist_rows(ctx)
    if not rows:
        return ('<p class="note">TRIPOD+AI 清单尚未安装：'
                f'<code>{Path(ITEMS_PATH).name}</code> 的 <code>items</code> 为空。'
                "条目文本须人工对照 BMJ 2024;385:e078378 转录，代码不生成条目；"
                "转录方法见 <code>references/standards-mapping.md</code>。</p>")
    n_ok = sum(1 for r in rows if r["status"] == "ok")
    n_author = sum(1 for r in rows if r["status"] == "author")
    n_na = sum(1 for r in rows if r["status"] == "N/A")
    head = (f'<p>共 {len(rows)} 条：✓ {n_ok} 条已有管线证据；⚠ {n_author} 条需作者补；'
            f"— {n_na} 条声明不适用。<b>本表是对照自查，不构成「符合」或「认证」</b>："
            "TRIPOD+AI 原文明确它不规定建模方法、也不是质量评价工具。</p>")
    body = "".join(
        f'<tr><td>{r["id"]}</td><td>{r["topic"]}</td><td>{r["mark"]}</td>'
        f'<td>{r["evidence"] or "—"}</td><td>{r["note"] or ""}</td></tr>' for r in rows)
    table = ('<div class="tw"><table><caption>TRIPOD+AI 条目对照（三态：✓ 有证据 / ⚠ 需补 / — 不适用）</caption>'
             '<thead><tr><th scope="col">条目</th><th scope="col">主题</th>'
             '<th scope="col">状态</th><th scope="col">本次证据</th>'
             f'<th scope="col">待补</th></tr></thead><tbody>{body}</tbody></table></div>')
    return head + table


def write_checklist(ctx, out_dir: Path) -> Path | None:
    """写 ``report/tripod_ai_checklist.md``。无条目时不写文件（避免交出一份空清单）。"""
    rows = checklist_rows(ctx)
    if not rows:
        return None
    lines = ["# TRIPOD+AI 条目对照（自动生成）", "",
             "> 本表把本次 run 的产物映射到 TRIPOD+AI 条目，**不是**合规认证。",
             "> 条目文本来自 `assets/standards/tripod_ai_2024.json`（人工对照 BMJ 2024;385:e078378 转录）。",
             "> ✓ = 管线可提供证据；⚠ = 须作者补；— = 声明不适用（理由见末列）。", "",
             "| 条目 | 主题 | 状态 | 本次证据 | 待补 / 理由 |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f'| {r["id"]} | {r["topic"]} | {r["mark"]} | '
                     f'{r["evidence"] or "—"} | {r["note"] or ""} |')
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "tripod_ai_checklist.md"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return p
