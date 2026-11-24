# Copyright 2026 The research-ml-modeling Authors
# SPDX-License-Identifier: Apache-2.0
"""从已有的 OOF / 逐折结果重算汇总表，不重新拟合任何模型。

什么时候用
----------
* 聚合口径或指标定义改了，想验证影响，但不想再等一次完整 CV；
* 手工改过 `oof_predictions.csv`（不推荐）后想重新出一致的表；
* 只想换 bootstrap 次数 / alpha 重算置信区间。

重新拟合请用 `stage_cv.py`；本脚本只读 `cv/` 下的产物。

    python rebuild_tables.py --run-dir D:/runs/nir_v1
    python rebuild_tables.py --run-dir D:/runs/nir_v1 --bootstrap 5000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from sml.compare import compare_models, rank_models  # noqa: E402
from sml.cv import oof_group_metrics  # noqa: E402
from sml.metrics import aggregate_repeats, describe_pooled_rows, mark_metric_basis  # noqa: E402
from sml.utils import as_path, ensure_dir, info, read_json, setup_console, warn  # noqa: E402


def _prep_mix(folds: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
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


def rebuild(run_dir: Path, n_boot: int | None, alpha: float | None) -> dict:
    cv_dir = run_dir / "cv"
    oof_p, folds_p = cv_dir / "oof_predictions.csv", cv_dir / "fold_results.csv"
    if not oof_p.exists() or not folds_p.exists():
        raise FileNotFoundError(f"{cv_dir} 下缺少 oof_predictions.csv 或 fold_results.csv")
    oof = pd.read_csv(oof_p)
    folds = pd.read_csv(folds_p)
    cfg = read_json(run_dir / "config.resolved.json")
    task = str(cfg["task"]["type"])
    n_classes = 1
    if task == "classification":
        labels = pd.unique(oof["y_true"])
        n_classes = int(len(labels))
    boot = cfg["evaluation"]["bootstrap"]
    B = int(n_boot or boot.get("n", 2000))
    a = float(alpha or boot.get("alpha", 0.05))
    seed = int(cfg["runtime"]["seed"])

    metric_cols = [m for m in (cfg["evaluation"]["metrics"] or []) if m in folds.columns]
    if not metric_cols:
        metric_cols = [c for c in ("r2", "rmse", "mae", "auc") if c in folds.columns]

    if "status" in folds:
        status = folds["status"].astype(str)
        ok = folds[status.isin(["ok", "degenerate"])]
        n_failed = int((~status.isin(["ok", "degenerate"])).sum())
        if n_failed:
            warn(f"{n_failed} 折为失败折（status 形如 error:XXX），已排除在聚合之外")
    else:
        ok = folds
    model_keys = [k for k in ("target", "model") if k in ok.columns]

    summary = aggregate_repeats(ok, metric_cols, group_keys=tuple(model_keys))
    mix = _prep_mix(ok, model_keys)
    if not summary.empty and not mix.empty:
        summary = summary.merge(mix, on=model_keys, how="left")
    oof_metrics = oof_group_metrics(oof, task, n_classes, group_keys=tuple(model_keys))
    if not summary.empty:
        keep = [c for c in oof_metrics.columns if c not in summary.columns]
        summary = summary.merge(oof_metrics[model_keys + keep], on=model_keys, how="left")
        # 口径判定必须和 stage_cv 用同一份实现：重算表同样会碰上「有效折不足」的
        # 留一法/小样本场景。少了这一步，rebuild 会静默退回折均值口径，还不报警。
        summary = mark_metric_basis(summary, ok, group_keys=tuple(model_keys))
        pooled_msg = describe_pooled_rows(summary)
        if pooled_msg:
            warn(pooled_msg)
        summary = rank_models(summary, task)
        summary.to_csv(cv_dir / "metrics_summary.csv", index=False, encoding="utf-8")
        info(f"metrics_summary.csv 重算完成（{len(summary)} 行，按 {'+'.join(model_keys)} 聚合）")

    prep_keys = tuple(model_keys + (["prep"] if "prep" in ok.columns else []))
    by_prep = aggregate_repeats(ok, metric_cols, group_keys=prep_keys)
    if not by_prep.empty:
        by_prep.to_csv(cv_dir / "metrics_by_prep.csv", index=False, encoding="utf-8")
        info(f"metrics_by_prep.csv 重算完成（{len(by_prep)} 行）")

    # 置信区间
    from sml.metrics import bootstrap_classification, bootstrap_regression

    rows = []
    for name, sub in oof.groupby(model_keys, dropna=False):
        name_tuple = name if isinstance(name, tuple) else (name,)
        base = dict(zip(model_keys, name_tuple))
        if task == "regression":
            ci = bootstrap_regression(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy(),
                                      sub["group"].to_numpy(dtype=object),
                                      n_boot=B, alpha=a, seed=seed)
        else:
            use_proba = "proba" in sub and pd.to_numeric(sub["proba"], errors="coerce").notna().any()
            score = (pd.to_numeric(sub["proba"], errors="coerce").to_numpy(dtype=float)
                     if use_proba else sub["y_pred"].to_numpy(dtype=float))
            kind = "native"
            if "proba_kind" in sub:
                vals = sub["proba_kind"].dropna()
                kind = str(vals.iloc[0]) if len(vals) else "native"
            ci = bootstrap_classification(sub["y_true"].to_numpy(), score,
                                          sub["group"].to_numpy(dtype=object),
                                          n_classes=n_classes, n_boot=min(B, 1000),
                                          alpha=a, seed=seed, proba_kind=kind)
        for metric, (lo, hi) in ci.items():
            rows.append({**base, "metric": metric, "ci_low": lo, "ci_high": hi,
                         "method": "cluster_bootstrap_percentile", "n_boot": B, "alpha": a})
    ci_df = pd.DataFrame(rows)
    if not ci_df.empty:
        ci_df.to_csv(cv_dir / "metrics_ci.csv", index=False, encoding="utf-8")
        info(f"metrics_ci.csv 重算完成（B={B}, alpha={a}）")

    comp = compare_models(oof, folds, task,
                          metrics=[m for m in ("r2", "rmse", "mae", "auc") if m in metric_cols],
                          n_boot=B, alpha=a, seed=seed, n_classes=n_classes)
    if not comp.empty:
        comp.to_csv(cv_dir / "comparison.csv", index=False, encoding="utf-8")
        info(f"comparison.csv 重算完成（{len(comp)} 行）")
    return {"rows_summary": 0 if summary.empty else len(summary),
            "rows_by_prep": 0 if by_prep.empty else len(by_prep),
            "rows_ci": 0 if ci_df.empty else len(ci_df)}


def main(argv: list[str] | None = None) -> int:
    setup_console()
    ap = argparse.ArgumentParser(description="从已有 OOF/逐折结果重算汇总表（不重新拟合）")
    ap.add_argument("--run-dir", required=True, help="已有 run 目录（需含 cv/oof_predictions.csv）")
    ap.add_argument("--bootstrap", type=int, default=None, help="覆盖 bootstrap 次数")
    ap.add_argument("--alpha", type=float, default=None, help="覆盖显著性水平")
    args = ap.parse_args(argv)
    run_dir = ensure_dir(as_path(args.run_dir))
    res = rebuild(run_dir, args.bootstrap, args.alpha)
    info(f"完成：{res}")
    warn("报告(report/)与图(figures/)需要重跑 stage_report 才会同步更新")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
