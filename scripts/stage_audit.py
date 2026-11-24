# Copyright 2026 The research-ml-modeling Authors
# SPDX-License-Identifier: Apache-2.0
"""stage 1：数据审计。

产出 audit/audit.json、audit/audit.md，以及一份可直接抄的 CV 方案建议。
不训练任何模型，几秒内出结果——任何建模之前都应该先看它。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sml.audit import audit_dataset, render_audit_markdown  # noqa: E402
from sml.stage_common import build_parser, prepare  # noqa: E402
from sml.utils import info, write_json, write_text  # noqa: E402


def run_audit(ctx, raw_df=None) -> dict:
    rep = audit_dataset(ctx.ds, ctx.cfg, raw_df)
    out_dir = ctx.sub("audit")
    write_json(out_dir / "audit.json", rep.to_dict())
    write_text(out_dir / "audit.md", render_audit_markdown(rep))
    n_crit, n_warn = rep.count("critical"), rep.count("warn")
    info(f"审计完成：{n_crit} 项严重、{n_warn} 项警告 -> {out_dir / 'audit.md'}")
    return rep.to_dict()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser("数据审计（不训练模型）")
    args = parser.parse_args(argv)
    ctx = prepare(args)
    run_audit(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
