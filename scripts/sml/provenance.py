# Copyright 2026 The research-ml-modeling Authors
# SPDX-License-Identifier: Apache-2.0
"""复现锁定：版本矩阵、种子派生、数据/配置/代码三哈希。

审稿人问"怎么复现"时，答案应该是 ``provenance.json`` 一个文件。
"""

from __future__ import annotations

import platform
import socket
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .utils import as_path, sha256_file, sha256_text

__all__ = ["package_versions", "code_hash", "seed_table", "build_provenance", "config_hash"]

CORE_PACKAGES = [
    "numpy", "pandas", "scikit-learn", "scipy", "xgboost", "lightgbm", "catboost",
    "shap", "matplotlib", "seaborn", "joblib", "openpyxl", "xlsxwriter", "plotly", "tqdm",
]


def package_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in CORE_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "MISSING"
    return out


def code_hash(scripts_dir: str | Path | None = None) -> str:
    """对 scripts/ 下所有 .py 按路径排序后拼接求哈希：代码一改就失效。"""
    root = as_path(scripts_dir or Path(__file__).resolve().parent.parent)
    files = sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)
    payload = []
    for f in files:
        try:
            payload.append(f"{f.relative_to(root).as_posix()}:{sha256_file(f)}")
        except OSError:
            continue
    return sha256_text("\n".join(payload))


def config_hash(cfg: dict) -> str:
    import json

    clean = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
    return sha256_text(json.dumps(clean, ensure_ascii=False, sort_keys=True, default=str))


def seed_table(master: int, repeats: int, outer_folds: int,
               scheme: str = "legacy", n_models: int = 0,
               model_names: list[str] | None = None) -> dict[str, Any]:
    """种子派生规则：repeat 级=r*1000+master，fold 级=repeat_seed*100+fold。

    全项目只用 numpy 的 ``default_rng``；sklearn/xgb/lgb 需要整数时直接取这里的值，
    避免 ``np.random.seed`` 与 ``default_rng`` 混用导致的不可复现。

    ``scheme="per_model"`` 时额外给出 ``per_fold_model``（fold 级种子 ×100 + 模型序），
    与 `cv.run_nested_cv` 的 per_model 派生一致；默认 ``legacy`` 输出与旧版逐字相同，
    不污染已有 provenance。
    """
    per_repeat = [int(master + r * 1000) for r in range(repeats)]
    per_fold = {
        f"r{r}f{k}": int((master + r * 1000) * 100 + k)
        for r in range(repeats) for k in range(outer_folds)
    }
    out: dict[str, Any] = {"master": int(master), "per_repeat": per_repeat,
                           "per_fold": per_fold}
    if str(scheme) == "per_model":
        names = list(model_names or [])
        out["scheme"] = "per_model"
        out["n_models"] = int(n_models)
        out["per_fold_model"] = {
            f"r{r}f{k}m{(names[i] if i < len(names) else i)}": int(
                ((master + r * 1000) * 100 + k) * 100 + i)
            for r in range(repeats) for k in range(outer_folds)
            for i in range(max(1, int(n_models)))
        }
    return out


def build_provenance(
    cfg: dict,
    ds,
    scripts_dir: str | Path | None = None,
    timing: dict[str, float] | None = None,
    fit_calls: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scheme = str(cfg.get("runtime", {}).get("seed_scheme", "legacy")).lower()
    names: list[str] = []
    if scheme == "per_model":
        try:
            from .models import resolve_models
            names = [s.name for s in resolve_models(cfg, str(cfg["task"]["type"]))]
        except Exception:  # noqa: BLE001 —— 记录用，取不到就退回配置里写明的名字
            names = [str(m.get("name")) for m in (cfg.get("models") or [])]
    seeds = seed_table(
        int(cfg["runtime"]["seed"]),
        int(cfg["split"]["repeats"]),
        int(cfg["split"]["outer"]["n_splits"]),
        scheme=scheme, n_models=len(names), model_names=names,
    )
    prov: dict[str, Any] = {
        "sml_version": __version__,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.version.split()[0],
        "python_full": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "machine": socket.node() if extra and extra.get("include_machine") else "redacted",
        "executable": sys.executable,
        "packages": package_versions(),
        "seeds": seeds,
        "hashes": {
            "data_sha256": getattr(ds, "data_hash", None),
            "config_sha256": config_hash(cfg),
            "code_sha256": code_hash(scripts_dir),
        },
        "budget": cfg.get("runtime", {}).get("budget", "custom"),
        "data": {
            "path": getattr(ds, "source_path", None),
            "n_rows": getattr(ds, "n_samples", None),
            "n_features": getattr(ds, "n_features", None),
            "n_groups": getattr(ds, "n_groups", None),
            "task": getattr(ds, "task", None),
            "target": getattr(ds, "target_name", None),
        },
        "timing_seconds": timing or {},
        "total_fit_calls": fit_calls,
    }
    return prov


def resume_key(prov: dict) -> str:
    h = prov["hashes"]
    return sha256_text(f"{h['data_sha256']}|{h['config_sha256']}|{h['code_sha256']}")


def resume_diff(old: dict, new: dict) -> list[str]:
    """比较两次 provenance，指出失效原因（给 --resume 用）。"""
    reasons = []
    oh, nh = old.get("hashes", {}), new.get("hashes", {})
    labels = {"data_sha256": "数据文件", "config_sha256": "配置", "code_sha256": "脚本代码"}
    for key, label in labels.items():
        if oh.get(key) != nh.get(key):
            reasons.append(f"{label}已变化（{key}）")
    return reasons
