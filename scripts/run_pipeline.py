# Copyright 2026 The research-ml-modeling Authors
# SPDX-License-Identifier: Apache-2.0
"""一键入口：从原始表格跑到论文级图表 + Methods 草稿 + 审稿质疑清单。

典型用法
--------
    # 1) 先看环境与本机 API 行为（首次必跑）
    python selfcheck.py

    # 2) 预演：只做审计并预估拟合次数/耗时，不训练
    python run_pipeline.py --config ../assets/config/regression_spectra.json \
        --data D:/data/NIR.xlsx --dry-run

    # 3) 正式跑
    python run_pipeline.py --config ../assets/config/regression_spectra.json \
        --data D:/data/NIR.xlsx --out D:/runs/nir_v1 --budget standard

    # 4) 换个随机种子复核稳定性（数据/配置/代码都没变，--resume 会自动跳过已完成 stage）
    python run_pipeline.py --config ... --data ... --out D:/runs/nir_v1 --seed 7

产物目录
--------
    config.resolved.json  provenance.json
    audit/  cv/  interp/  perm/  ad/  figures/  tables/  report/
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sml.config import ALL_STAGES, ConfigError  # noqa: E402
from sml.cv import describe_split_plan, estimate_fit_calls  # noqa: E402
from sml.models import resolve_models  # noqa: E402
from sml.provenance import resume_diff, resume_key  # noqa: E402
from sml.stage_common import build_parser, prepare, stage_stages  # noqa: E402
from sml.utils import (  # noqa: E402
    ensure_dir, human_duration, info, mem_guard, non_ascii, read_json, warn, write_json,
    write_text,
)

STAGE_OUTPUTS = {
    "audit": "audit/audit.json",
    "cv": "cv/oof_predictions.csv",
    "interp": "interp/interp.json",
    "perm": "perm/permutation.json",
    "ad": "ad/ad_eval.json",
    "report": "report/report.md",
}

# 单次拟合的兜底耗时（秒）：**只在实测标定失败时**使用。
# 旧版只用这张表做估算，它跟行数、波长数、树数、预处理链全无关：
# 光谱上 airPLS+SG 单次 fit 实测 24 s 却按 0.03 s 计，估出来的 20 分钟变成过夜。
# 现在 --dry-run 先在子样本上真跑一遍再外推（见 _calibrate_fit_seconds）。
FIT_COST = {"pls": 0.03, "pcr": 0.04, "ridge": 0.02, "lasso": 0.12, "enet": 0.15,
            "svr": 0.06, "knn": 0.005, "rf": 0.35, "et": 0.30, "gbr": 0.25,
            "xgb": 0.30, "lgbm": 0.12, "cat": 0.45, "gp": 0.20,
            "logreg": 0.04, "svc": 0.05, "ridge_clf": 0.02, "gnb": 0.005, "lda": 0.02}


def _calibrate_fit_seconds(ctx, specs, max_rows: int = 400, max_cols: int = 400,
                           reps: int = 3) -> dict[str, Any]:
    """在子样本上真实 fit，按 (行数 × 列数) 外推到全表，返回预处理与模型的单次耗时。

    为什么必须真跑：拟合成本主要由「预处理链 × 行数 × 列数 × 树数」决定，
    任何写死的经验系数都会在光谱数据上错 1–2 个数量级。这里只花几秒，
    换来可信的 ETA —— 它的意义是让 AI 在开跑前就能说清"这一轮要多久"。

    返回 ``{"prep": [每个预处理候选的秒], "model": {模型名: 秒}, "scale": ...}``；
    任何异常都退回 ``FIT_COST``（返回空 dict），绝不让预估失败拖垮 dry-run。
    """
    import time as _time

    from sml.models import build_pipeline
    from sml.preprocess import clear_prep_cache

    out: dict[str, Any] = {}
    try:
        ds, cfg = ctx.ds, ctx.cfg
        X = np.asarray(ds.X, dtype=float)
        n, p = int(X.shape[0]), int(X.shape[1])
        rng = np.random.default_rng(0)
        if n > max_rows:
            rows = np.sort(rng.choice(n, size=max_rows, replace=False))
        else:
            rows = np.arange(n)
        cols = (np.unique(np.linspace(0, p - 1, min(p, max_cols)).astype(int))
                if p > max_cols else np.arange(p))
        Xs = X[rows][:, cols]
        scale = (n / len(rows)) * (p / len(cols))
        tname = ds.target_names[0]
        y_all = np.asarray(ds.targets[tname])
        ys = y_all[rows]
        if ds.task != "regression":
            ys = np.array([str(v) if pd.notna(v) else "NA" for v in ys], dtype=object)
        cands = cfg["preprocess"]["candidates"]

        def _time_fit(spec, which: int, passthrough_model: bool = False) -> float:
            """冷启动计时：每次拟合前清空预处理缓存，否则第二次 0.001 s 的假读数会污染标定。"""
            best = float("inf")
            for _ in range(reps):
                pipe = build_pipeline(spec, cfg, ds.feature_names, seed=0)
                pipe.set_params(prep__which=which)
                if passthrough_model:
                    pipe.set_params(model="passthrough")
                clear_prep_cache()
                t0 = _time.perf_counter()
                pipe.fit(Xs, ys)
                best = min(best, _time.perf_counter() - t0)
            return best

        # 预处理段：模型 passthrough，只量 impute + prep + scale
        prep_cost = [_time_fit(specs[0], j, passthrough_model=True) for j in range(len(cands))]
        # 模型段：全流程减去最省的那条预处理链
        model_cost: dict[str, float] = {}
        for spec in specs:
            full = min(_time_fit(spec, j) for j in range(len(cands)))
            model_cost[spec.name] = max(0.0, full - min(prep_cost))
        out = {"prep": [c * scale for c in prep_cost],
               "model": {k: v * scale for k, v in model_cost.items()},
               "scale": scale, "n_sub": [len(rows), len(cols)]}
    except Exception as exc:  # noqa: BLE001 —— 预估失败不许中断流程
        warn(f"耗时标定失败（{type(exc).__name__}: {exc}），退回经验系数估算")
        return {}
    return out


def _stage_key(prov: dict) -> str:
    return resume_key(prov)


def _load_state(run_dir: Path) -> dict:
    p = run_dir / "_stages.json"
    if p.exists():
        try:
            return read_json(p)
        except json.JSONDecodeError:
            return {}
    return {}


def _should_skip(stage: str, state: dict, key: str, run_dir: Path, resume: bool) -> bool:
    if not resume:
        return False
    rec = state.get(stage) or {}
    out = run_dir / STAGE_OUTPUTS.get(stage, "")
    return bool(rec.get("key") == key and out.exists() and out.stat().st_size > 0)


def _mark_done(stage: str, state: dict, key: str, run_dir: Path, seconds: float) -> None:
    state[stage] = {"key": key, "done_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "seconds": round(seconds, 1)}
    write_json(run_dir / "_stages.json", state)


def _dry_run(ctx, stages: list[str]) -> int:
    from sml.audit import audit_dataset, render_audit_markdown

    info("— dry run：只做审计与成本预估，不训练任何模型 —")
    rep = audit_dataset(ctx.ds, ctx.cfg)
    out = ctx.sub("audit")
    write_json(out / "audit.json", rep.to_dict())
    from sml.utils import write_text

    write_text(out / "audit.md", render_audit_markdown(rep))

    specs = resolve_models(ctx.cfg, ctx.ds.task)
    est = estimate_fit_calls(ctx.ds, ctx.cfg, specs)
    plan = describe_split_plan(ctx.ds, ctx.cfg)

    per_sec = 0.0
    calib = _calibrate_fit_seconds(ctx, specs)
    n_preps = max(1, len(ctx.cfg["preprocess"]["candidates"]))
    prep_unit = 0.0
    if calib:
        prep_unit = float(np.mean(calib["prep"]))
        info("标定明细：子样本 {0}×{1}，外推系数 {2:.1f}×；预处理单次 "
             "{3}；模型单次 {4}".format(
                 calib["n_sub"][0], calib["n_sub"][1], calib["scale"],
                 ", ".join(f"{v:.2f}s" for v in calib["prep"]),
                 ", ".join(f"{k}={v:.2f}s" for k, v in calib["model"].items())))
        prep_unit = float(np.mean(calib["prep"]))
        # 预处理按「外层折 × 内层折 × 候选」的训练集各算一次，结果在候选/模型间共享：
        # 这正是 L3 折级缓存带来的结构，估算必须按这个结构算，否则会重复计数。
        unique_prep_fits = (max(1, plan["outer_n_splits"]) * max(1, plan["repeats"])
                            * (max(1, plan["inner_n_splits"]) + 1) * n_preps)
        per_sec += unique_prep_fits * prep_unit
    for spec in specs:
        calls = est["per_model"][spec.name]["fit_calls"]
        unit = (calib.get("model") or {}).get(spec.name) if calib else None
        if unit is None:
            unit = FIT_COST.get(spec.kind, 0.1)
        else:
            est["per_model"][spec.name]["sec_per_fit"] = round(unit + prep_unit, 3)
        per_sec += calls * unit
    if calib:
        mean_model = float(np.mean(list(calib["model"].values()))) if calib["model"] else 0.1
        perm_fits = est["permutation_fit_calls_est"]
        perm_sec = perm_fits * mean_model + (max(1, plan["repeats"])
                                             * max(1, plan["outer_n_splits"])) * prep_unit
    else:
        perm_sec = est["permutation_fit_calls_est"] * 0.08
    total = per_sec + (perm_sec if "perm" in stages else 0)
    # n_jobs=-1 在旧实现里被 max(1, -1) 吞成 1，导致 dry-run 按单核估算，
    # 在 8–16 核机器上高估 5–10 倍——这正是「预估 3 小时、实际 20 分钟」的主因。
    # 反过来，写死的 FIT_COST 又会在 airPLS/树模型上低估 1–2 个数量级。
    raw_jobs = int(ctx.cfg["runtime"].get("n_jobs", 1) or 1)
    n_jobs = (os.cpu_count() or 1) if raw_jobs < 0 else max(1, raw_jobs)
    if n_jobs > 1:
        total /= min(n_jobs, 16) * 0.8

    print()
    info(f"验证方案：{plan['strategy']}，外层 {plan['outer_n_splits']} 折 × 重复 {plan['repeats']} 次，"
         f"内层 {plan['inner_n_splits']} 折")
    info(f"分组键：{plan['group_col']}；{plan['note']}")
    info(f"将执行 stage：{', '.join(stages)}")
    if len(est.get("per_target") or {}) > 1:
        info("各目标有效行数（多目标按各自 y 的有效值建模）：" +
             "；".join(f"{t} {v['n_rows']} 行/{v['n_groups']} 组"
                      for t, v in est["per_target"].items()))
    print()
    print(f"{'模型':<14}{'每折候选':>10}{'内层折':>8}{'拟合次数':>12}{'单次拟合(s)':>14}")
    print("-" * 60)
    for name, row in est["per_model"].items():
        sec = row.get("sec_per_fit")
        sec_s = f"{sec:.2f}" if sec is not None else "—"
        print(f"{name:<14}{row['candidates_per_fold']:>10}{row['inner_folds']:>8}"
              f"{row['fit_calls']:>12,}{sec_s:>14}")
    print("-" * 60)
    print(f"{'合计':<14}{'':>10}{'':>8}{est['total_inner_fit_calls']:>12,}")
    if "perm" in stages:
        print(f"{'置换检验(估)':<14}{'':>10}{'':>8}{est['permutation_fit_calls_est']:>12,}")
    print()
    basis = "子样本冷启动实测标定（按行数×列数外推）" if calib else "经验系数（标定失败）"
    # 这行给的是**下限**：只算了拟合本身，没算 interp/perm 的带宽置换、报告绘图、
    # joblib 调度与折间开销。实测里这些常再翻 2–5 倍，所以说"下限"而不是装成精确值。
    info(f"拟合耗时下限（{basis}，串行加总后按 n_jobs 折算）：约 {human_duration(total)}"
         f"（n_jobs={n_jobs}；interp/perm 附加计算与调度开销另计，实际通常再高 2–10 倍；"
         f"本 run 目录已有实测时以下一行「参考」为准）")
    prev = _load_state(ctx.run_dir)
    if prev:
        parts = [f"{k} {v.get('seconds')}s" for k, v in prev.items()
                 if isinstance(v, dict) and v.get("seconds")]
        if parts:
            info("参考：本 run 目录上次实测 —— " + "；".join(parts))
    if "perm" not in stages:
        info("本次未包含 perm 阶段，置换检验耗时未计入——但严谨性红线要求至少跑一次 perm")
    info(f"若觉得太慢：--budget fast，或 --set split.repeats=3、--set permutation.n_perm=50")
    info(f"审计报告已写出：{out / 'audit.md'}")
    crit = rep.count("critical")
    if crit:
        warn(f"审计发现 {crit} 项严重问题，请先看 audit.md 再正式跑")
    print()
    info("确认无误后去掉 --dry-run 即可开始")
    return 0


# --------------------------------------------------------------------------- #
# 多目标：按目标拆成独立子 run
# --------------------------------------------------------------------------- #
def _target_slug(name: str, idx: int) -> str:
    safe = re.sub(r"[^0-9A-Za-z_.\-\u4e00-\u9fff]+", "_", str(name)).strip("_")
    return f"{idx:02d}_{safe or 'target'}"


def _peek_plan(args) -> tuple[list[str], list[str]]:
    """只解析配置、不载数据：返回 (目标列表, 已有的 exclude_from_features)。"""
    if not getattr(args, "config", None):
        return [], []
    from sml.config import resolve_config

    cfg = resolve_config(path=args.config, overrides=args.overrides, budget=args.budget,
                         seed=args.seed, data_path=args.data, out_dir=args.out)
    raw = (cfg.get("task") or {}).get("target") or []
    targets = [str(t) for t in raw] if isinstance(raw, (list, tuple)) else [str(raw)]
    excludes = [str(c) for c in ((cfg.get("data") or {}).get("exclude_from_features") or [])]
    return targets, excludes


def _summarize_subrun(out: Path) -> dict:
    rec: dict = {}
    try:
        s = pd.read_csv(out / "cv" / "metrics_summary.csv")
        task = ("classification"
                if "auc_mean" in s.columns and "r2_mean" not in s.columns else "regression")
        # 与报告同一套选模口径：有效折不足时用池化 OOF，而不是只看折均值。
        from sml.selection import pick_best_model

        model, info = pick_best_model(out, task)
        metric = "auc" if task == "classification" else "r2"
        rec.update({"best_model": model, "metric": metric,
                    "value": float(info.get(metric, float("nan"))),
                    "basis": info.get("basis", "fold")})
    except Exception:  # noqa: BLE001
        pass
    try:
        rec["n_failed"] = read_json(out / "cv" / "cv_summary.json").get("n_failed")
    except Exception:  # noqa: BLE001
        pass
    return rec


def _write_target_index(out_root: Path, records: list[dict]) -> Path:
    ensure_dir(out_root)
    write_json(out_root / "targets_index.json",
               {"n_targets": len(records), "targets": records})
    multi = len(records) > 1
    lines = [("# 多目标运行索引" if multi else "# 运行索引"), "",
             ("每个目标在独立子目录里完整跑一遍（各自的数据审计、嵌套 CV、置换检验、适用域、"
              "图表与报告）。**不同目标的指标尺度与难度不同，不能放在同一张排名表里比**，"
              "因此这里只并列展示各自的最优模型，不做跨目标排序。"
              if multi else
              "单目标也使用与多目标一致的 `targets/<序号>_<目标>/` 结构，"
              "汇总脚本不需要再为两种目录结构各写一个分支。"), "",
             "| 目标 | 最优模型 | 主指标 | 失败折 | 子目录 |", "|---|---|---|---|---|"]
    for r in records:
        val = (f"{r['metric']} = {r['value']:.4f}" if r.get("value") is not None else "—")
        lines.append(f"| {r['target']} | {r.get('best_model', '—')} | {val} | "
                     f"{r.get('n_failed', '—')} | `targets/{Path(r['out']).name}/` |")
    lines += ["", "各目标的完整报告见对应子目录下的 `report/report.md`。"]
    p = write_text(out_root / "targets_summary.md", "\n".join(lines))
    return p


def _run_multi_target(args, targets: list[str], excludes: list[str] | None = None) -> int:
    out_root = (Path(args.out) if args.out
                else Path("runs") / f"{time.strftime('%Y%m%d_%H%M%S')}_targets")
    if len(targets) > 1:
        info(f"检测到 {len(targets)} 个目标：{', '.join(targets)}")
    info(f"按目标拆成独立子 run（各自出报告/图/表）：{out_root / 'targets'}")
    if len(targets) > 1:
        info("原因：不同目标的指标尺度与难度不同，全表排名会选出「跨目标最强」而不是各目标最优；"
             "报告与 CI 也会串目标。")
    else:
        info("单目标也走同一目录结构（targets/01_<目标>/），便于汇总脚本统一处理；"
             "要回到旧的根目录结构请加 --no-fanout。")

    records: list[dict] = []
    all_ok = True
    for i, tname in enumerate(targets, start=1):
        sub_out = out_root / "targets" / _target_slug(tname, i)
        argv = ["--config", str(args.config), "--out", str(sub_out)]
        if getattr(args, "data", None):
            argv += ["--data", str(args.data)]
        if getattr(args, "budget", None):
            argv += ["--budget", str(args.budget)]
        if getattr(args, "seed", None) is not None:
            argv += ["--seed", str(args.seed)]
        if getattr(args, "stages", None):
            argv += ["--stages", str(args.stages)]
        if getattr(args, "dry_run", False):
            argv += ["--dry-run"]
        if getattr(args, "resume", False):
            argv += ["--resume"]
        if getattr(args, "n_jobs", None):
            argv += ["--n-jobs", str(args.n_jobs)]
        if getattr(args, "quiet", False):
            argv += ["--quiet"]
        if getattr(args, "fail_fast", False):
            argv += ["--fail-fast"]
        for ov in (getattr(args, "overrides", None) or []):
            argv += ["--set", ov]
        for rm in (getattr(args, "removals", None) or []):
            argv += ["--set-remove", rm]
        # 子 run 的 task.target 只剩当前目标，io_data 便不知道兄弟目标的存在；
        # 必须显式把全部目标列排除出特征，否则会用"蛋白"预测"水分"。
        keep_out = list(dict.fromkeys([*(excludes or []), *targets]))
        if keep_out:
            argv += ["--set", "data.exclude_from_features="
                              + json.dumps(keep_out, ensure_ascii=False)]
        argv += ["--set", "task.target=" + json.dumps([tname], ensure_ascii=False)]
        argv += ["--no-fanout"]          # 单目标了，避免递归
        print()
        info(f"===== 目标 {i}/{len(targets)}：{tname} =====")
        rc = main(argv)
        all_ok = all_ok and rc == 0
        records.append({"target": tname, "out": str(sub_out), "exit": rc,
                        **_summarize_subrun(sub_out)})

    index_path = _write_target_index(out_root, records)
    print()
    info(f"多目标完成：{len(records)} 个目标，索引 {index_path}")
    for r in records:
        status = "ok" if r.get("exit") == 0 else f"失败(exit={r.get('exit')})"
        info(f"  ├ {r['target']}: {r.get('best_model', '—')} / "
             f"{r.get('metric', '—')}={r.get('value', float('nan')):.4f} [{status}]")
    return 0 if all_ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(
        "科研级机器学习预测建模：数据审计 → 嵌套分组CV → 置换检验 → 适用域 → 报告",
        stages=True,
    )
    parser.add_argument("--no-fanout", action="store_true",
                        help="不按目标拆子目录（单目标直接写在 run 根目录；多目标仅允许 audit,cv）")
    args = parser.parse_args(argv)

    # 按目标分流必须在建 run 目录之前：下游 stage（图/AD/置换/报告）都是单目标语义。
    # 单目标也默认走 targets/01_<目标>/，让目录结构统一；--no-fanout 可回到根目录。
    if not getattr(args, "run_dir", None):
        targets, excludes = _peek_plan(args)
        if targets:
            if not args.no_fanout:
                return _run_multi_target(args, targets, excludes)
            unsafe = (sorted(set(stage_stages(args)) & {"interp", "perm", "ad", "report"})
                      if len(targets) > 1 else [])
            if unsafe:
                raise ConfigError(
                    "E-CFG-031",
                    f"多目标 + --no-fanout 无法正确运行 {', '.join(unsafe)}：\n"
                    f"  - 一个 run 目录只有一套 figures/ 与 report/，多目标会互相覆盖；\n"
                    f"  - interp / perm / ad 只按第一个目标（{targets[0]}）计算，其余目标没有结果；\n"
                    f"  - 等到 report 阶段才拒绝，CV 的 {len(targets)} 个目标已经白跑。\n"
                    f"  修法：去掉 --no-fanout（自动按目标拆成 <out>/targets/<目标>/ 各自出报告）；\n"
                    f"        若只想看 CV 对照表：--stages audit,cv。")

    ctx = prepare(args)
    stages = stage_stages(args)[:]

    if getattr(args, "dry_run", False):
        return _dry_run(ctx, stages)

    cfg = ctx.cfg
    run_dir = ctx.run_dir
    prov = cfg.get("_provenance") or read_json(run_dir / "provenance.json")
    key = _stage_key(prov)
    state = _load_state(run_dir)

    if non_ascii(str(run_dir)) and int(cfg["runtime"].get("n_jobs", 1) or 1) > 1:
        warn("输出目录含非 ASCII 字符且开启多进程；若出现 pickle 报错，请改用英文目录或 --n-jobs 1")

    need_gb = float(cfg["runtime"].get("mem_guard_gb", 0) or 0)
    guard = mem_guard(need_gb, "本次建模") if need_gb > 0 else None
    if guard is not None:
        guard.__enter__()

    t_all = time.perf_counter()
    timings: dict[str, float] = {}
    ran: list[str] = []
    skipped: list[str] = []
    try:
        for stage in stages:
            if _should_skip(stage, state, key, run_dir, bool(getattr(args, "resume", False))):
                info(f"[{stage}] 三哈希一致且产物完整，跳过")
                skipped.append(stage)
                continue
            t0 = time.perf_counter()
            if stage == "audit":
                from stage_audit import run_audit

                run_audit(ctx)
            elif stage == "cv":
                from stage_cv import run_cv

                run_cv(ctx)
            elif stage == "interp":
                from stage_interp import run_interp

                run_interp(ctx)
            elif stage == "perm":
                from stage_perm import run_perm

                run_perm(ctx)
            elif stage == "ad":
                from stage_ad import run_ad

                run_ad(ctx)
            elif stage == "report":
                from sml.report import run_report

                run_report(ctx)
            else:
                warn(f"未知 stage {stage}，跳过")
                continue
            dt = time.perf_counter() - t0
            timings[stage] = round(dt, 1)
            _mark_done(stage, state, key, run_dir, dt)
            ran.append(stage)
            info(f"[{stage}] 完成，耗时 {human_duration(dt)}")
    finally:
        if guard is not None:
            guard.__exit__(None, None, None)

    total = time.perf_counter() - t_all
    # 把耗时写回 provenance，方便回看
    try:
        prov_path = run_dir / "provenance.json"
        prov_now = read_json(prov_path)
        prov_now["timing_seconds"] = {**prov_now.get("timing_seconds", {}), **timings}
        prov_now["total_fit_calls"] = int(cfg["runtime"].get("_total_fit_calls", 0)) or prov_now.get("total_fit_calls")
        write_json(prov_path, prov_now)
    except Exception:  # noqa: BLE001
        pass

    print()
    info(f"全部完成：执行 {len(ran)} 个 stage，跳过 {len(skipped)} 个，总耗时 {human_duration(total)}")
    info(f"产物目录：{run_dir}")
    for rel, label in (("audit/audit.md", "数据审计"), ("report/report.html", "HTML 报告"),
                       ("report/report.md", "主报告"),
                       ("report/methods_zh.md", "Methods 草稿"), ("report/reviewer_checklist.md", "审稿清单"),
                       ("cv/metrics_summary.csv", "性能汇总"), ("figures", "图表")):
        p = run_dir / rel
        if p.exists():
            info(f"  ├ {label}: {p}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConfigError as exc:
        # 配置类错误是可以照着改的，不要把整段 traceback 甩给用户
        print(f"\n[配置错误，未开始训练] {exc}\n", flush=True)
        raise SystemExit(2) from None
