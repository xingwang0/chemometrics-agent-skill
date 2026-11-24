# Copyright 2026 The research-ml-modeling Authors
# SPDX-License-Identifier: Apache-2.0
"""各 stage 脚本的公共前置：命令行参数、配置解析、数据载入、run 目录与 provenance。

每个 stage_*.py 都通过这里接入，保证：
* 参数一致（都能用 --config/--data/--out/--set/--budget/--seed）
* 产物落在同一个 run 目录，stage 之间靠文件而非内存串联
* provenance 只写一次，--resume 时校验三哈希
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from . import provenance as prov_mod
from .config import apply_overrides, config_conflicts, resolve_config, stage_list
from .io_data import load_dataset
from .utils import ensure_dir, info, read_json, setup_console, write_json

__all__ = ["add_common_args", "build_parser", "prepare", "RunContext", "SUBDIRS"]

SUBDIRS = ["audit", "cv", "cv/models", "model", "perm", "ad", "interp",
          "figures", "tables", "report"]


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = parser.add_argument_group("公共参数")
    g.add_argument("--config", default=None,
                   help="JSON 配置文件（assets/config/*.json）；用 --run-dir 时可省略")
    g.add_argument("--data", default=None, help="数据文件路径，覆盖配置里的 data.path")
    g.add_argument("--out", default=None, help="run 输出目录；默认 runs/<时间戳>")
    g.add_argument("--set", dest="overrides", action="append", default=[],
                   help="点号路径覆盖，如 --set split.repeats=10（可多次）")
    g.add_argument("--set-remove", dest="removals", action="append", default=[],
                   help="从列表/字典中移除元素，如 --set-remove data.drop_cols=time（可多次）")
    g.add_argument("--budget", default=None, choices=["fast", "standard", "thorough"],
                   help="预算档位，统一缩放 repeats / 内层折数 / n_iter / n_perm / bootstrap")
    g.add_argument("--seed", type=int, default=None, help="主随机种子（默认取配置）")
    g.add_argument("--run-dir", default=None,
                   help="指定已存在的 run 目录（stage 单独运行时用来读取上游产物）")
    g.add_argument("--quiet", action="store_true", help="减少控制台输出")
    g.add_argument("--fail-fast", action="store_true", help="任一模失败立即中断（默认记 NaN 继续）")
    g.add_argument("--allow-config-change", action="store_true",
                   help="允许在已有 run 目录上换用另一套 task/data/models（会覆盖 config.resolved.json）")
    return parser


def build_parser(description: str, stages: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    if stages:
        parser.add_argument("--stages", default=None,
                            help="逗号分隔：audit,cv,interp,perm,ad,report")
        parser.add_argument("--dry-run", action="store_true", help="只做审计并预估耗时，不训练任何模型")
        parser.add_argument("--resume", action="store_true", help="三哈希一致时复用已有 stage 产物")
        parser.add_argument("--n-jobs", type=int, default=None, help="内层并行进程数（外层始终串行）")
    return parser


class RunContext:
    def __init__(self, cfg: dict, ds, run_dir: Path):
        self.cfg = cfg
        self.ds = ds
        self.run_dir = run_dir

    def sub(self, name: str) -> Path:
        return ensure_dir(self.run_dir / name)

    def file(self, rel: str) -> Path:
        return self.run_dir / rel

    def exists(self, rel: str) -> bool:
        p = self.run_dir / rel
        return p.exists() and p.stat().st_size > 0

    @property
    def task(self) -> str:
        return self.ds.task

    @property
    def target(self) -> str:
        return self.ds.target_name

    def load_json(self, rel: str, default: Any = None) -> Any:
        p = self.run_dir / rel
        if not p.exists():
            if default is None:
                raise FileNotFoundError(f"缺少上游产物 {rel}，请先跑对应 stage")
            return default
        return read_json(p)


def prepare(args: argparse.Namespace) -> RunContext:
    setup_console()
    from .utils import set_verbose

    set_verbose(not getattr(args, "quiet", False))
    if getattr(args, "quiet", False):
        from .utils import quiet_library_warnings

        quiet_library_warnings()
    if not getattr(args, "run_dir", None) and not getattr(args, "config", None):
        raise SystemExit("需要 --config <配置文件>，或用 --run-dir <已有 run 目录> 单独跑某个 stage")

    if args.run_dir:
        run_dir = ensure_dir(args.run_dir)
        cfg_file = run_dir / "config.resolved.json"
        if not cfg_file.exists():
            raise FileNotFoundError(f"{run_dir} 下没有 config.resolved.json，无法单独运行 stage")
        if getattr(args, "overrides", None) or getattr(args, "removals", None):
            from .utils import warn as _warn

            _warn(f"--run-dir 模式下 --set / --set-remove 会被忽略（配置直接取自 {cfg_file.name}）："
                  f"{list(getattr(args, 'overrides', None) or []) + list(getattr(args, 'removals', None) or [])}。"
                  f"要改配置请换 --out 新建 run，或先手工改 config.resolved.json")
        cfg = read_json(cfg_file)
        cfg["_warnings"] = []
        cfg.setdefault("runtime", {})["out"] = str(run_dir)
        ds, _ = load_dataset(cfg, args.data)
        info(f"复用 run 目录：{run_dir}")

        # 旧实现只在 --config 分支重建 provenance；--run-dir --resume 会直接读
        # 磁盘上的旧 provenance，代码/数据已变时仍然命中旧 stage key 而跳过。
        # 这里用当前代码与数据重新计算，并写回，让 resume key 反映真实状态。
        prov_path = run_dir / "provenance.json"
        prov = prov_mod.build_provenance(
            cfg, ds, scripts_dir=Path(__file__).resolve().parent.parent)
        if prov_path.exists():
            try:
                old = read_json(prov_path)
            except Exception:  # noqa: BLE001
                old = {}
            diff = prov_mod.resume_diff(old, prov)
            if diff:
                from .utils import warn as _warn

                _warn("上游产物失效：" + "；".join(diff))
            elif getattr(args, "resume", False):
                info("三哈希一致，--resume 可用")
        write_json(prov_path, prov)
        cfg["_provenance"] = prov
        return RunContext(cfg, ds, run_dir)

    cfg = resolve_config(
        path=args.config,
        overrides=args.overrides,
        budget=args.budget,
        seed=args.seed,
        data_path=args.data,
        out_dir=args.out,
        removals=getattr(args, "removals", None),
    )
    if getattr(args, "fail_fast", False):
        cfg["runtime"]["fail_fast"] = True
    if getattr(args, "n_jobs", None):
        cfg["runtime"]["n_jobs"] = int(args.n_jobs)

    ds, _ = load_dataset(cfg)

    out = args.out or cfg["runtime"].get("out")
    if not out:
        import time

        out = f"runs/{time.strftime('%Y%m%d_%H%M%S')}_{Path(str(ds.source_path)).stem}"
    run_dir = ensure_dir(out)
    for sub in SUBDIRS:
        ensure_dir(run_dir / sub)

    from .config import dump_resolved

    cfg_path = run_dir / "config.resolved.json"
    if cfg_path.exists():
        conflicts = config_conflicts(read_json(cfg_path), cfg)
        if conflicts and not getattr(args, "allow_config_change", False):
            from .config import ConfigError

            raise ConfigError(
                "E-CFG-030",
                f"同一个 run 目录不能混用两套配置（{run_dir}）：\n  - "
                + "\n  - ".join(conflicts)
                + "\n  上游 cv/ 产物是按旧配置算的，继续会写出与结果对不上账的报告。\n"
                  "  改法：换一个 --out；或用 --run-dir 复用旧配置；"
                  "确实要换就显式加 --allow-config-change（会覆盖 config.resolved.json）。")
        if conflicts:
            from .utils import warn as _warn

            _warn("按 --allow-config-change 覆盖已有配置：" + "；".join(conflicts))
    dump_resolved(cfg, cfg_path)

    prov_path = run_dir / "provenance.json"
    prov = prov_mod.build_provenance(cfg, ds, scripts_dir=Path(__file__).resolve().parent.parent)
    if prov_path.exists():
        old = read_json(prov_path)
        diff = prov_mod.resume_diff(old, prov)
        if diff:
            info("上游产物失效：" + "；".join(diff))
        elif getattr(args, "resume", False):
            info("三哈希一致，--resume 可用")
    write_json(prov_path, prov)
    cfg["_provenance"] = prov

    for w in cfg.get("_warnings", []):
        from .utils import warn

        warn(w)
    return RunContext(cfg, ds, run_dir)


def stage_stages(args: argparse.Namespace) -> list[str]:
    return stage_list(getattr(args, "stages", None))


def override(cfg: dict, pairs: list[str]) -> dict:
    return apply_overrides(cfg, pairs)
