# Copyright 2026 The research-ml-modeling Authors
# SPDX-License-Identifier: Apache-2.0
"""stage 2：嵌套分组重复交叉验证。

产出（全部落在 <run>/cv/）
    oof_predictions.csv   逐行 OOF 预测——全链路唯一真源
    fold_results.csv      每个 repeat×fold×model 一行，含逐折指标与最优参数
    metrics_summary.csv   先按 repeat 求折均值、再对 repeat 求 mean±sd
    metrics_ci.csv        按组 bootstrap 的 95% 置信区间
    comparison.csv        模型两两配对 bootstrap 差值 + BH/Bonferroni 校正
    models.json           生效的模型清单
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sml.compare import rank_models, compare_models  # noqa: E402
from sml.cv import effective_repeats, oof_group_metrics, run_nested_cv  # noqa: E402
from sml.metrics import (  # noqa: E402
    PRED_RANGE_FACTOR, aggregate_repeats, bootstrap_classification, bootstrap_regression,
    describe_pooled_rows, mark_metric_basis, oof_score_matrix,
)
from sml.models import model_summary, resolve_models  # noqa: E402
from sml.stage_common import build_parser, prepare  # noqa: E402
from sml.utils import info, warn, write_json  # noqa: E402


def _bootstrap_all(ctx, oof: pd.DataFrame) -> pd.DataFrame:
    """按 (target, model) 汇总的按组 bootstrap CI。

    预处理是**逐折内部选出的超参**，因此不按 prep 拆开算 CI——否则同一个模型会因为
    不同折选了不同预处理而被拆成几行，排名时等于在"模型×预处理"组合里挑最优
    （winner's curse）。逐预处理的细分结果另存 metrics_by_prep.csv。
    """
    cfg = ctx.cfg
    ds = ctx.ds
    boot = cfg["evaluation"]["bootstrap"]
    B = int(boot.get("n", 2000))
    alpha = float(boot.get("alpha", 0.05))
    seed = int(cfg["runtime"]["seed"])
    n_classes = 1 if ds.task == "regression" else int(len(ds.classes or [2]))
    rows = []
    keys = [k for k in ("target", "model") if k in oof.columns]
    for name, sub in oof.groupby(keys, dropna=False):
        name_tuple = name if isinstance(name, tuple) else (name,)
        base = dict(zip(keys, name_tuple))
        if ds.task == "regression":
            ci = bootstrap_regression(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy(),
                                      sub["group"].to_numpy(dtype=object),
                                      n_boot=B, alpha=alpha, seed=seed)
        else:
            score, kind = oof_score_matrix(sub, n_classes)
            ci = bootstrap_classification(sub["y_true"].to_numpy(), score,
                                          sub["group"].to_numpy(dtype=object),
                                          n_classes=n_classes,
                                          n_boot=min(B, 1000), alpha=alpha, seed=seed,
                                          proba_kind=kind)
        if not ci:
            warn(f"{base} 的 bootstrap 置信区间无法计算（独立样品过少）")
        for metric, (lo, hi) in ci.items():
            rows.append({**base, "metric": metric, "ci_low": lo, "ci_high": hi,
                         "method": "cluster_bootstrap_percentile", "n_boot": B,
                         "alpha": alpha})
    return pd.DataFrame(rows)


def _prep_mix(folds: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """统计每个模型在各折里实际选中的预处理分布，形如 ``raw×14、sg1_snv×1``。"""
    if "prep" not in folds:
        return pd.DataFrame(columns=keys + ["prep_mix", "n_prep_distinct"])
    rows = []
    for name, sub in folds.groupby(keys, dropna=False):
        name_tuple = name if isinstance(name, tuple) else (name,)
        vc = sub["prep"].value_counts()
        rows.append({**dict(zip(keys, name_tuple)),
                     "prep_mix": "、".join(f"{k}×{v}" for k, v in vc.items()),
                     "n_prep_distinct": int(vc.size)})
    return pd.DataFrame(rows)


def _failed_mask(folds: pd.DataFrame) -> pd.Series:
    """失败判定只能靠 status，不能写 ``status != "FAILED"``。

    cv.py 记的是 ``error:ValueError`` 这种带前缀的值，永远不会等于字面量
    "FAILED"——旧写法因此既没过滤掉失败折，也把 n_failed 恒算成 0
    （"0 失败"的错误结论就是这么来的）。
    """
    if "status" not in folds:
        return pd.Series(False, index=folds.index)
    status = folds["status"].astype(str)
    return ~status.isin(["ok", "degenerate"])


def _failure_summary(folds: pd.DataFrame, failed: pd.Series) -> dict[str, dict]:
    """按模型汇总失败情况，供报告置顶告警。"""
    if folds.empty or not failed.any():
        return {}
    out: dict[str, dict] = {}
    for model, sub in folds.groupby("model", dropna=False):
        mask = failed.loc[sub.index]
        if not mask.any():
            continue
        errs = sub.loc[mask, "error"] if "error" in sub else pd.Series(dtype=str)
        first = next((str(v) for v in errs.dropna() if str(v).strip()), "（未记录错误信息）")
        out[str(model)] = {
            "n_failed": int(mask.sum()),
            "n_total": int(len(sub)),
            "all_failed": bool(mask.all()),
            "first_error": first,
            "scoring": (str(sub.loc[mask, "scoring"].dropna().iloc[0])
                        if "scoring" in sub and len(sub.loc[mask, "scoring"].dropna()) else ""),
        }
    return out


def _prediction_range_summary(folds: pd.DataFrame, failed: pd.Series) -> dict[str, dict]:
    """按模型汇总"预测值跑出目标量纲"的折，供报告与 cv_summary 置顶告警。

    只统计未失败且列存在的折；分类任务没有 pred_n_extreme 列，直接返回空。
    这类折的指标（R² 可能是 −37）反映的是线性外推爆炸，不是模型能力——
    两者的处理方式完全不同（前者查数据/换模型，后者才是模型不好）。
    """
    if folds.empty or "pred_n_extreme" not in folds.columns:
        return {}
    extreme = pd.to_numeric(folds["pred_n_extreme"], errors="coerce").fillna(0)
    sub = folds[(~failed) & (extreme > 0)]
    if sub.empty:
        return {}
    out: dict[str, dict] = {}
    ok_total = folds[~failed]
    for model, m in sub.groupby("model", dropna=False):
        total = int((ok_total["model"].astype(str) == str(model)).sum())
        ratio = pd.to_numeric(m["pred_range_ratio"], errors="coerce")
        out[str(model)] = {
            "n_folds": int(len(m)),
            "n_folds_total": total,
            "max_ratio": (float(np.nanmax(ratio.to_numpy())) if ratio.notna().any()
                          else float("nan")),
            "n_rows_extreme": int(extreme.loc[m.index].sum()),
        }
    return out


def _best_by_primary(summary: pd.DataFrame, task: str) -> dict[str, dict] | None:
    """每个 target 各自的最优模型（多目标时 summary 是纵向拼接表，不能全表挑一个）。"""
    if summary.empty:
        return None
    metric = "r2" if task == "regression" else "auc"
    col = f"{metric}_mean"
    if col not in summary.columns:
        return None
    groups = (summary.groupby("target", dropna=False) if "target" in summary.columns
              else [(None, summary)])
    out: dict[str, dict] = {}
    for tname, sub in groups:
        sub = sub.copy()
        score = pd.to_numeric(sub[col], errors="coerce")
        # 与 pick_best_model / report 同一口径：有效折不足的行改用池化 OOF 主指标。
        if "metric_basis" in sub.columns and metric in sub.columns:
            pooled = sub["metric_basis"].astype(str).eq("pooled")
            score = score.where(~pooled, pd.to_numeric(sub[metric], errors="coerce"))
        sub["_sel"] = score
        sub = sub[np.isfinite(sub["_sel"])]
        if sub.empty:
            continue
        best = sub.sort_values("_sel", ascending=False).iloc[0]
        prep = best.get("prep_mix")
        basis = "pooled" if str(best.get("metric_basis", "fold")) == "pooled" else "fold"
        out[str(tname)] = {
            "model": str(best["model"]),
            metric: float(best["_sel"]),
            "basis": basis,
            "prep_mix": None if prep is None or pd.isna(prep) else str(prep),
        }
    return out or None


def run_cv(ctx) -> dict:
    cfg = ctx.cfg
    ds = ctx.ds
    out_dir = ctx.sub("cv")
    specs = resolve_models(cfg, ds.task)
    write_json(out_dir / "models.json", model_summary(specs))

    total_rounds = effective_repeats(cfg)
    # holdout_random 只有 1 个外层折；直接打印 outer.n_splits 会让人以为跑了 5 折
    n_outer_eff = (1 if cfg["split"]["strategy"] == "holdout_random"
                   else int(cfg["split"]["outer"]["n_splits"]))
    info(f"开始嵌套 CV：{len(specs)} 个模型 × {total_rounds} 次重复 × 外层 "
         f"{n_outer_eff} 折 × 内层 {cfg['split']['inner']['n_splits']} 折")

    counter = {"n": 0}

    def progress(msg: str) -> None:
        counter["n"] += 1
        if counter["n"] % 5 == 1:
            print(f"[sml]   {msg}", flush=True)

    oof, folds = run_nested_cv(ctx, specs, progress=progress)
    if oof.empty:
        raise RuntimeError("所有折都失败，没有产生任何 OOF 预测；请用 --fail-fast 查看首个错误")

    oof.to_csv(out_dir / "oof_predictions.csv", index=False, encoding="utf-8")
    folds.to_csv(out_dir / "fold_results.csv", index=False, encoding="utf-8")

    n_classes = 1 if ds.task == "regression" else int(len(ds.classes or [2]))
    metric_cols = [m for m in cfg["evaluation"]["metrics"]
                   if m in folds.columns and m not in ("n",)]
    if not metric_cols:
        metric_cols = [c for c in ("r2", "rmse", "mae") if c in folds.columns]

    failed = _failed_mask(folds)
    ok_folds = folds[~failed]
    fail_info = _failure_summary(folds, failed)
    for model, finfo in fail_info.items():
        if finfo["all_failed"]:
            warn(f"模型 {model!r} 的 {finfo['n_failed']}/{finfo['n_total']} 折全部失败，已从结果表剔除。"
                 f"这不是'效果差'，而是配置或依赖错误：{finfo['first_error']}")
        else:
            warn(f"模型 {model!r} 有 {finfo['n_failed']}/{finfo['n_total']} 折失败：{finfo['first_error']}")

    # 预测值越界（外推爆炸）：R² 只会显示"这个模型很差"，看不出是数值跑飞了。
    # 有这类折时必须在报告顶部说清楚，否则论文里会出现"某模型 R²=−38"这种被误读的结论。
    range_flags = _prediction_range_summary(folds, failed)
    for model, rinfo in range_flags.items():
        warn(f"模型 {model!r} 有 {rinfo['n_folds']}/{rinfo['n_folds_total']} 折的预测值超出训练集"
             f"目标范围 {rinfo['max_ratio']:.1f} 倍极差（阈值 {PRED_RANGE_FACTOR:.0f} 倍，"
             f"涉及 {rinfo['n_rows_extreme']} 个越界样品）：这些折的指标反映的是外推爆炸，"
             f"不是模型能力。先查目标尺度、极端样品与线性外推，不要直接把 R² 写进论文")
    # 主汇总口径：按 (target, model) —— 预处理是逐折内部选的超参，不参与排名
    model_keys = [k for k in ("target", "model") if k in ok_folds.columns]
    summary = aggregate_repeats(ok_folds, metric_cols, group_keys=tuple(model_keys))
    if not summary.empty:
        mix = _prep_mix(ok_folds, model_keys)
        if not mix.empty:
            summary = summary.merge(mix, on=model_keys, how="left")
        oof_metrics = oof_group_metrics(oof, ds.task, n_classes, group_keys=tuple(model_keys))
        keep_oof = [c for c in oof_metrics.columns if c not in summary.columns]
        summary = summary.merge(oof_metrics[model_keys + keep_oof], on=model_keys, how="left")
        # 有效折计数 + 口径判定：退化折（测试样品 <2，主指标 NaN）会被 nanmean 跳过，
        # 会造出「k 折的平均其实只来自 1 个折」这种假象（留一法下尤其明显）。
        # 判定逻辑统一在 sml.metrics.mark_metric_basis —— rebuild_tables 走的是同一份实现，
        # 否则「只重算表不重训」会把池化口径又丢回折均值，而且不报警。
        summary = mark_metric_basis(summary, ok_folds, group_keys=tuple(model_keys))
        pooled_msg = describe_pooled_rows(summary)
        if pooled_msg:
            warn(pooled_msg)
        summary = rank_models(summary, ds.task)
        if ds.task == "regression" and {"r2_mean", "r2"} <= set(summary.columns):
            groups = (summary.groupby("target", dropna=False)
                      if "target" in summary.columns else [("", summary)])
            for tname, sub in groups:
                sub = sub[np.isfinite(pd.to_numeric(sub["r2_mean"], errors="coerce"))]
                if sub.empty:
                    continue
                top_fold = sub.sort_values("r2_mean", ascending=False).iloc[0]
                top_pool = sub.sort_values("r2", ascending=False).iloc[0]
                if str(top_fold["model"]) != str(top_pool["model"]):
                    warn(f"选模口径冲突（目标 {tname or ds.target_name}）：折均值第一 "
                         f"{top_fold['model']}（R²={float(top_fold['r2_mean']):.3f}），"
                         f"池化 OOF 第一 {top_pool['model']}（R²={float(top_pool['r2']):.3f}）。"
                         f"报告会同时标注两种口径；论文引用时不要混用。")
        summary.to_csv(out_dir / "metrics_summary.csv", index=False, encoding="utf-8")

    # 最终模型：用 CV 最优模型 + 最常用超参在全量数据上重训，供下游预测新样品。
    final_info: dict = {"saved": False, "reason": "no_summary"}
    if not summary.empty:
        try:
            from sml.final_model import save_final_model

            final_info = save_final_model(ctx, summary)
        except Exception as exc:  # noqa: BLE001 —— 保存失败不应毁掉 CV 结果
            warn(f"最终模型保存失败（不影响 CV 结果）：{type(exc).__name__}: {exc}")

    # 附：按 (target, model, prep) 的细分表，供"预处理到底选了什么"的追问
    prep_keys = tuple([k for k in ("target", "model") if k in ok_folds.columns] + ["prep"])
    by_prep = aggregate_repeats(ok_folds, metric_cols, group_keys=prep_keys)
    if not by_prep.empty:
        by_prep.to_csv(out_dir / "metrics_by_prep.csv", index=False, encoding="utf-8")

    ci = _bootstrap_all(ctx, oof)
    ci.to_csv(out_dir / "metrics_ci.csv", index=False, encoding="utf-8")

    comp = compare_models(
        oof, folds, ds.task,
        metrics=[m for m in metric_cols if m in ("r2", "rmse", "mae", "auc")],
        n_boot=int(cfg["evaluation"]["bootstrap"].get("n", 2000)),
        alpha=float(cfg["evaluation"]["bootstrap"].get("alpha", 0.05)),
        seed=int(cfg["runtime"]["seed"]),
        n_classes=n_classes,
    )
    comp.to_csv(out_dir / "comparison.csv", index=False, encoding="utf-8")

    result = {
        "n_models": len(specs),
        "models": [s.name for s in specs],
        "n_oof_rows": int(len(oof)),
        "n_folds": int(len(folds)),
        "n_failed": int(failed.sum()),
        "failed_models": fail_info,
        "prediction_range_flags": range_flags,
        "n_models_usable": int(len([s for s in specs if s.name not in
                                    {k for k, v in fail_info.items() if v["all_failed"]}])),
        "metric_cols": metric_cols,
        "aggregation": "按 (target, model) 聚合；pred 逐折内部选择，另见 metrics_by_prep.csv",
        "best_by_primary": _best_by_primary(summary, ds.task),
        "prep_choices": ({} if "prep" not in ok_folds else
                         {k: int(v) for k, v in ok_folds["prep"].value_counts().items()}),
        "total_fit_calls": int(cfg["runtime"].get("_total_fit_calls", 0)),
        "final_model": final_info,
    }
    write_json(out_dir / "cv_summary.json", result)
    msg = f"CV 完成：OOF {len(oof)} 行，逐折 {len(folds)} 行，失败 {result['n_failed']} 折"
    if fail_info:
        msg += f"（涉及 {len(fail_info)} 个模型，详见 cv_summary.json 的 failed_models）"
    info(msg)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = build_parser("嵌套分组重复交叉验证（最耗时的一步）")
    args = parser.parse_args(argv)
    ctx = prepare(args)
    run_cv(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
