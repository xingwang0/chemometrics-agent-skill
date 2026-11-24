# Copyright 2026 The research-ml-modeling Authors
# SPDX-License-Identifier: Apache-2.0
"""stage 6：报告装配（图 + 表 + Markdown + Methods 草稿 + 审稿清单）。

产物
    figures/F1..F8 (png+pdf)
    tables/T1..T6.csv + tables.xlsx
    report/report.md, methods_zh.md, methods_en.md, reviewer_checklist.md
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sml.report import run_report  # noqa: E402
from sml.stage_common import build_parser, prepare  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = build_parser("汇总报告：图表 / 表格 / Methods 草稿 / 审稿清单")
    args = parser.parse_args(argv)
    ctx = prepare(args)
    run_report(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
