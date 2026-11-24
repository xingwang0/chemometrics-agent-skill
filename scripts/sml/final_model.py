# Copyright 2026 The research-ml-modeling Authors
# SPDX-License-Identifier: Apache-2.0
"""最终模型落盘与新样品预测。

为什么需要这一层
----------------
CV 里的 ``cv/models/*.joblib`` 是**逐折**模型（每折只看训练折数据），不能直接拿来
预测新样品：它们既不是全量重训，也没有统一的预处理/超参口径。下游要画工艺窗口、
预测新批次时，必须有一个「用 CV 选出的最优模型 + 最常用超参，在全部数据上重训」
的最终模型。

本模块在 ``stage_cv`` 末尾保存 ``model/final_model.joblib`` 与
``model/model_card.json``；``scripts/predict.py`` 直接加载它做新样品预测，
并可选地调用 ``ad/ad_model.joblib`` 先判适用域。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ConfigError
from .utils import info, read_json, warn, write_json

__all__ = ["save_final_model", "load_final_model", "predict_frame"]


def _target_valid(y: np.ndarray, task: str) -> np.ndarray:
    return np.isfinite(y) if task == "regression" else ~pd.isna(y)


def save_final_model(ctx, summary: pd.DataFrame | None = None) -> dict[str, Any]:
    """用 CV 最优模型 + 最常用超参在全部有效数据上重训并保存。

    多目标 run 不保存单一模型（不同目标的模型不能混在一个 joblib 里）；
    默认的按目标 fan-out 会让每个子 run 各自保存一份。
    """
    cfg = ctx.cfg
    ds = ctx.ds
    if len(ds.target_names) != 1:
        warn("当前 run 含多个目标，未保存单一最终模型；请用默认的按目标 fan-out，"
             "或对每个目标单独跑 stage_cv")
        return {"saved": False, "reason": "multi_target"}
    if not bool(cfg.get("runtime", {}).get("save_final_model", True)):
        info("runtime.save_final_model=false，跳过最终模型保存")
        return {"saved": False, "reason": "disabled"}

    from .models import build_pipeline, prep_label, resolve_models
    from .selection import fixed_params_for, pick_best_model, spec_for_model

    target = ds.target_name
    model_name, best = pick_best_model(ctx.run_dir, ds.task, target=target)
    spec = spec_for_model(resolve_models(cfg, ds.task), model_name, ctx.run_dir)
    params = fixed_params_for(ctx.run_dir, model_name)
    clean = {k: v for k, v in params.items()
             if str(k).startswith(("model__", "prep__", "scale__"))}

    pipe = build_pipeline(spec, cfg, ds.feature_names, seed=int(cfg["runtime"]["seed"]))
    if clean:
        try:
            pipe.set_params(**clean)
        except ValueError as exc:
            warn(f"最终模型无法应用 CV 超参（{exc}），将使用估计器默认值")

    y = np.asarray(ds.targets[target])
    valid = _target_valid(y, ds.task)
    X_fit = ds.X[valid]
    y_fit = y[valid]
    pipe.fit(X_fit, y_fit)

    # 新样品可能带缺失；保存训练集的列中位数，预测时按同一规则插补。
    med = np.nanmedian(np.asarray(X_fit, dtype=np.float64), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)

    out_dir = ctx.run_dir / "model"
    out_dir.mkdir(parents=True, exist_ok=True)
    prep_idx = int(clean.get("prep__which", 0) or 0)
    prep_choice = prep_label(cfg, prep_idx)
    card: dict[str, Any] = {
        "target": target,
        "task": ds.task,
        "model": model_name,
        "selection_basis": best.get("basis", "fold"),
        "params": clean,
        "prep": prep_choice,
        "n_train": int(valid.sum()),
        "n_features": int(ds.n_features),
        "feature_names": list(ds.feature_names),
        "classes": ds.classes,
        "positive_label": ds.positive_label,
        "label_map": {str(k): v for k, v in (ds.label_map or {}).items()},
        "data_hash": ds.data_hash,
        "seed": int(cfg["runtime"]["seed"]),
        "trained_on": "full_dataset_after_cv",
        "note": ("这是 CV 选模后的全量重训模型；性能数字请引用 cv/metrics_summary.csv，"
                 "不要用训练集拟合值重新报 R²。"),
    }
    payload = {
        "pipeline": pipe,
        "feature_names": list(ds.feature_names),
        "impute_median": med,
        "task": ds.task,
        "target": target,
        "model": model_name,
        "params": clean,
        "classes": ds.classes,
        "positive_label": ds.positive_label,
        "label_map": ds.label_map,
    }

    try:
        import joblib

        joblib.dump(payload, out_dir / "final_model.joblib", compress=3)
        write_json(out_dir / "model_card.json", card)
    except Exception as exc:  # noqa: BLE001
        warn(f"最终模型保存失败（不影响 CV 结果）：{exc}")
        return {"saved": False, "reason": str(exc)}

    info(f"最终模型已保存：{out_dir / 'final_model.joblib'}"
         f"（{model_name}，全量 {int(valid.sum())} 行重训）")
    return {"saved": True, "path": str(out_dir / "final_model.joblib"),
            "model": model_name, "target": target, "n_train": int(valid.sum())}


def load_final_model(run_dir: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    run = Path(run_dir)
    model_path = run / "model" / "final_model.joblib"
    card_path = run / "model" / "model_card.json"
    if not model_path.exists():
        raise ConfigError(
            "E-PRED-001",
            f"缺少 {model_path}。请先跑 stage_cv（会自动保存），"
            f"或运行 scripts/predict.py --run-dir {run} --retrain 从 CV 产物重建。",
        )
    import joblib

    payload = joblib.load(model_path)
    card = read_json(card_path) if card_path.exists() else {}
    return payload, card


def predict_frame(run_dir: str | Path, df: pd.DataFrame, use_ad: bool = True) -> pd.DataFrame:
    """对一张新样品表做预测，并按需附加适用域判据。"""
    payload, _card = load_final_model(run_dir)
    feats = list(payload["feature_names"])
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise ConfigError(
            "E-PRED-002",
            f"新样品表缺少 {len(missing)} 个训练特征（前 10 个：{missing[:10]}）。"
            f"特征必须与训练时完全一致；列名/顺序不要手工改。",
        )
    X = df[feats].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    med = np.asarray(payload.get("impute_median", []), dtype=np.float64)
    if med.size == X.shape[1]:
        inds = np.where(~np.isfinite(X))
        if inds[0].size:
            X[inds] = med[inds[1]]
    X = np.nan_to_num(X, nan=0.0)

    pipe = payload["pipeline"]
    pred = np.asarray(pipe.predict(X))
    out = df.copy()
    out["prediction"] = pred
    if payload.get("task") == "classification" and hasattr(pipe, "predict_proba"):
        proba = np.asarray(pipe.predict_proba(X), dtype=np.float64)
        if proba.ndim == 2 and proba.shape[1] == 2:
            out["proba_positive"] = proba[:, 1]
        elif proba.ndim == 2:
            for i in range(proba.shape[1]):
                out[f"p{i}"] = proba[:, i]
            out["proba_max"] = proba.max(axis=1)

    if use_ad:
        ad_path = Path(run_dir) / "ad" / "ad_model.joblib"
        if ad_path.exists():
            try:
                import joblib

                ad = joblib.load(ad_path)
                sc = ad.score(X)
                out["ad_flag"] = sc["ad_flag"].to_numpy()
                out["ad_votes"] = sc["ad_votes"].to_numpy()
                out["ad_majority"] = sc["ad_majority"].to_numpy()
            except Exception as exc:  # noqa: BLE001
                warn(f"适用域打分失败（预测结果仍可用，但不要宣称域内）：{exc}")
        else:
            warn("缺少 ad/ad_model.joblib，未做适用域判定")
    return out
