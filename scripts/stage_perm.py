# Copyright 2026 The research-ml-modeling Authors
# SPDX-License-Identifier: Apache-2.0
"""stage 3：Y-randomization 置换检验。

回答"你的模型是不是在拟合噪声"。化学计量学审稿的标配自证。
产物：perm/permutation.json、perm/null_distribution.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sml.metrics import classification_metrics, oof_score_matrix, regression_metrics  # noqa: E402
from sml.models import resolve_models  # noqa: E402
from sml.permutation import y_randomization  # noqa: E402
from sml.selection import fixed_params_for, pick_best_model, spec_for_model  # noqa: E402
from sml.stage_common import build_parser, prepare  # noqa: E402
from sml.utils import info, warn, write_json  # noqa: E402


def _observed_from_oof(ctx, model_name: str, metric: str) -> float | None:
    """用与置换零分布完全相同的池化 OOF 口径重算观测值。

    旧实现直接取 ``metrics_summary`` 里的 ``r2_mean``/``auc_mean``（折均值），
    而零分布是全部折外预测拼起来算的池化值；两者不是同一个统计量，p 值会偏。
    """
    oof_path = ctx.run_dir / "cv" / "oof_predictions.csv"
    if not oof_path.exists():
        return None
    oof = pd.read_csv(oof_path)
    sub = oof[oof["model"].astype(str) == str(model_name)]
    if sub.empty:
        return None
    if ctx.ds.task == "regression":
        return float(regression_metrics(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy())[metric])
    n_classes = int(len(ctx.ds.classes or [2]))
    score, kind = oof_score_matrix(sub, n_classes)
    return float(classification_metrics(sub["y_true"].to_numpy(), score,
                                        n_classes=n_classes, proba_kind=kind)[metric])


def run_perm(ctx) -> dict:
    cfg = ctx.cfg
    ds = ctx.ds
    out_dir = ctx.sub("perm")
    if not cfg["permutation"].get("enabled", True):
        info("permutation.enabled=false，跳过置换检验")
        return {"skipped": True}

    model_name, best = pick_best_model(ctx.run_dir, ds.task)
    spec = spec_for_model(resolve_models(cfg, ds.task), model_name, ctx.run_dir)
    params = fixed_params_for(ctx.run_dir, model_name)
    n_perm = int(cfg["permutation"].get("n_perm", 200))
    metric = "r2" if ds.task == "regression" else "auc"
    observed = _observed_from_oof(ctx, model_name, metric)
    if observed is None or not np.isfinite(observed):
        observed = float(best.get(metric, float("nan")))

    n_jobs = int(cfg["runtime"].get("n_jobs", 1) or 1)
    info(f"Y-randomization：模型 {model_name}，n_perm={n_perm}"
         f"（样品级置换，分组结构保持不变，n_jobs={n_jobs}）")
    result = y_randomization(
        ctx, spec,
        obs_metric=observed,
        fixed_params=params,
        n_perm=n_perm,
        seed=int(cfg["runtime"]["seed"]),
        mode=str(cfg["permutation"].get("mode", "fast")),
        n_jobs=n_jobs,
    )
    write_json(out_dir / "permutation.json", result)
    null_values = result.pop("null_values", [])
    if null_values:
        pd.DataFrame({
            "perm_index": np.arange(len(null_values)),
            "null_metric": null_values,
            "metric": result.get("metric"),
            "observed": result.get("observed"),
        }).to_csv(out_dir / "null_distribution.csv", index=False, encoding="utf-8")
    if result.get("error"):
        warn(f"置换检验未完成：{result['error']}")
    else:
        info(f"{result.get('conclusion')}")
    if result.get("fixed_hyperparams"):
        info("注意：本次置换采用固定超参（保守估计），报告中需注明这一点")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = build_parser("Y-randomization 置换检验（较耗时）")
    args = parser.parse_args(argv)
    ctx = prepare(args)
    run_perm(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
