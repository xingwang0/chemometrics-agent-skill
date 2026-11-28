# Copyright 2026 The research-ml-modeling Authors
# SPDX-License-Identifier: Apache-2.0
"""数据审计：样本量、分组结构、缺失、近零方差、共线性、目标泄漏、EPV、CV 方案建议。

审计的定位是"在花四小时训练之前，先把会让结论作废的问题摆到桌面上"。
所有结论都写成「问题 + 依据 + 建议动作」，能自动判定的直接给配置字段。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .io_data import Dataset

__all__ = ["audit_dataset", "render_audit_markdown"]

MAX_PAIRWISE_FEATURES = 1500
LEAK_SPEARMAN = 0.98
LEAK_SPEARMAN_NEAR = 0.90   # 0.90–0.98：单特征已能解释大部分方差，先按「近克隆列」告警
LEAK_AUC = 0.99
LEAK_AUC_NEAR = 0.95
COLLINEAR_R = 0.99
NEAR_ZERO_CV = 1e-8


@dataclass
class Finding:
    level: str            # ok | info | warn | critical
    title: str
    detail: str
    action: str = ""
    key: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"level": self.level, "key": self.key, "title": self.title,
                "detail": self.detail, "action": self.action}


@dataclass
class AuditReport:
    summary: dict[str, Any] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    group_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    feature_issues: pd.DataFrame = field(default_factory=pd.DataFrame)
    leakage_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    recommendations: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "findings": [f.as_dict() for f in self.findings],
            "group_stats": self.group_stats.to_dict(orient="records"),
            "feature_issues": self.feature_issues.to_dict(orient="records"),
            "leakage_top": self.leakage_table.head(20).to_dict(orient="records"),
            "recommendations": self.recommendations,
        }

    def count(self, level: str) -> int:
        return sum(1 for f in self.findings if f.level == level)


# --------------------------------------------------------------------------- #
def _fast_auc_binary(y: np.ndarray, score: np.ndarray) -> float:
    """单变量 AUC（Mann-Whitney U 的归一化形式），无 sklearn 依赖、O(n log n)。"""
    y = np.asarray(y).astype(int)
    n_pos = int((y == 1).sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = stats.rankdata(score, method="average")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _group_structure(ds: Dataset) -> tuple[pd.DataFrame, dict[str, Any]]:
    g = pd.Series(ds.groups, name="group")
    sizes = g.value_counts()
    info: dict[str, Any] = {
        "n_rows": int(len(g)),
        "n_groups": int(sizes.size),
        "rows_per_group_min": int(sizes.min()),
        "rows_per_group_max": int(sizes.max()),
        "rows_per_group_median": float(sizes.median()),
        "has_replicates": bool(sizes.max() > 1),
        "n_groups_single_row": int((sizes == 1).sum()),
    }
    df = sizes.rename_axis("group").reset_index(name="n_rows")
    return df.sort_values("n_rows", ascending=False), info


def _feature_issues(ds: Dataset) -> pd.DataFrame:
    X = ds.X
    n = X.shape[0]
    std = X.std(axis=0, ddof=1) if n > 1 else np.zeros(X.shape[1])
    mean = X.mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where(np.abs(mean) > 0, std / np.abs(mean), np.inf)
    n_unique = np.array([len(np.unique(X[:, j])) for j in range(X.shape[1])])
    nan_frac = np.isnan(X).mean(axis=0)
    df = pd.DataFrame({
        "feature": ds.feature_names,
        "mean": mean, "std": std, "cv": cv,
        "n_unique": n_unique, "nan_frac": nan_frac,
        "constant": std == 0,
        "near_zero_var": (std < NEAR_ZERO_CV) | (n_unique <= 1),
    })
    return df


def _collinearity(ds: Dataset) -> dict[str, Any]:
    """用奇异值分解估条件数与有效秩；成对共线在特征数可控时才逐对算。"""
    X = np.asarray(ds.X, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    scale = X.std(axis=0, ddof=1)
    scale[scale == 0] = 1.0
    Z = X / scale
    out: dict[str, Any] = {"n_pairs_above": None, "max_abs_r": None, "pairs": []}
    try:
        sv = np.linalg.svd(Z, compute_uv=False)
    except np.linalg.LinAlgError:
        out["cond"] = None
        out["effective_rank"] = None
        return out
    sv = sv[sv > 0]
    if sv.size:
        out["cond"] = float(sv[0] / sv[-1])
        tol = sv[0] * max(Z.shape) * np.finfo(float).eps
        out["effective_rank"] = int((sv > tol).sum())
    else:
        out["cond"] = None
        out["effective_rank"] = 0

    p = ds.n_features
    if p <= MAX_PAIRWISE_FEATURES:
        # 常数列会让 np.corrcoef 出 0/0，先剔除再算（常数列已在 feature_issues 里单独报）
        keep = scale > 0
        if keep.sum() >= 2:
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = np.corrcoef(Z[:, keep], rowvar=False)
            np.fill_diagonal(corr, 0.0)
            abs_corr = np.nan_to_num(np.abs(corr), nan=0.0)
            out["max_abs_r"] = float(abs_corr.max())
            ii, jj = np.where(np.triu(abs_corr > COLLINEAR_R, k=1))
            out["n_pairs_above"] = int(ii.size)
            names = [ds.feature_names[i] for i in np.flatnonzero(keep)]
            out["pairs"] = [
                {"a": names[i], "b": names[j], "r": float(abs_corr[i, j])}
                for i, j in list(zip(ii, jj))[:20]
            ]
    else:
        rng = np.random.default_rng(0)
        idx = rng.choice(p, size=MAX_PAIRWISE_FEATURES, replace=False)
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(Z[:, idx], rowvar=False)
        np.fill_diagonal(corr, 0.0)
        abs_corr = np.nan_to_num(np.abs(corr), nan=0.0)
        out["max_abs_r"] = float(abs_corr.max())
        out["n_pairs_above"] = int((np.triu(abs_corr > COLLINEAR_R, k=1)).sum())
        out["sampled"] = True
    return out


def _leakage_scan(ds: Dataset, top_k: int = 15) -> pd.DataFrame:
    """单特征与目标的关联强度。过强即疑似目标泄漏或标签编码错误。"""
    X = np.asarray(ds.X, dtype=np.float64)
    y = np.asarray(ds.y, dtype=np.float64)
    p = ds.n_features
    idx = np.arange(p)
    if p > 20000:
        idx = np.random.default_rng(0).choice(p, size=20000, replace=False)

    if ds.task == "regression":
        rows = []
        for j in idx:
            col = X[:, j]
            if np.nanstd(col) == 0:
                continue
            rho = stats.spearmanr(col, y, nan_policy="omit").statistic
            rows.append({"feature": ds.feature_names[j], "stat": "spearman", "value": float(rho)})
        df = pd.DataFrame(rows)
        df["abs_value"] = df["value"].abs()
    else:
        yb = (y == (len(ds.classes) - 1)).astype(int)
        rows = []
        for j in idx:
            col = X[:, j]
            if np.nanstd(col) == 0:
                continue
            auc = _fast_auc_binary(yb, col)
            rows.append({"feature": ds.feature_names[j], "stat": "univar_auc", "value": float(auc)})
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df["abs_value"] = (df["value"] - 0.5).abs() + 0.5
    if df.empty:
        return df
    return df.sort_values("abs_value", ascending=False).head(top_k).reset_index(drop=True)


def _looks_like_id(name: str) -> bool:
    """按词元判断列名是否像编号，避免 ``solid``/``amide``/``width`` 这类
    仅因为包含子串 ``id`` 就被误判为泄漏（旧实现会直接给 critical 并建议删列）。
    """
    low = str(name).strip().lower()
    if any(tok in low for tok in ("编号", "序号", "样号", "样本号")):
        return True
    tokens = [t for t in re.split(r"[^0-9a-z]+", low) if t]
    # ``idx`` 是 idx 列最常见的名字（pandas 导出的行号就叫 idx），漏掉它是漏报，
    # 而漏报一个行号列比误报一列更危险：模型会直接记住行序。
    if any(t in {"id", "index", "idx", "no", "serial"} for t in tokens):
        return True
    return bool(re.search(r"(^|[_\-])(id|index|idx|no|serial)([_\-]|$)", low))


def _pseudo_id_scan(ds: Dataset) -> list[str]:
    """列名像 ID 且取值几乎唯一 -> 疑似把编号当特征。"""
    suspects = []
    n = ds.n_samples
    for j, name in enumerate(ds.feature_names):
        if _looks_like_id(name) and len(np.unique(ds.X[:, j])) >= max(0.95 * n, 3):
            suspects.append(str(name))
    return suspects


def _epv(ds: Dataset) -> dict[str, Any]:
    n = ds.n_samples
    p = ds.n_features
    out = {"n_samples": n, "n_features": p, "ratio_n_p": float(n / p) if p else None}
    if ds.task == "classification":
        counts = pd.Series(ds.y).value_counts()
        out["min_class_count"] = int(counts.min())
        out["epv"] = float(counts.min() / p) if p else None
        out["imbalance_ratio"] = float(counts.max() / counts.min()) if counts.min() else None
        out["class_counts"] = {str(ds.label_map.get(k, k)): int(v) for k, v in counts.items()}
    if p > 0:
        # 光谱数据的经验判据：n >= 3*sqrt(p) 才算"极勉强"
        out["n_vs_sqrt_p"] = float(n / np.sqrt(p))
    return out


def _recommend(ds: Dataset, ginfo: dict[str, Any], cfg: dict) -> dict[str, Any]:
    rec: dict[str, Any] = {}
    n_groups = ginfo["n_groups"]
    k_out = int(cfg["split"]["outer"]["n_splits"])
    # 建议值必须反映实际生效的分组键：split.group_col 优先，其次 data.id_col。
    # 只写 data.id_col 会让「按配比 mixture_id 分组」的 run 在报告里被写成 sample_id。
    rec["group_col"] = (cfg["split"].get("group_col")
                        or cfg["data"].get("id_col") or "auto_fingerprint")

    if n_groups < 5:
        rec["outer_n_splits"] = max(2, n_groups)
        rec["note_outer"] = f"仅 {n_groups} 个独立样品，外层折数降为 {rec['outer_n_splits']}，性能估计极不稳定"
    elif n_groups < 10:
        rec["outer_n_splits"] = 3
        rec["note_outer"] = f"{n_groups} 个 group，建议外层 3 折（5 折会有折只含 1–2 个样品）"
    elif n_groups < 30:
        # 15–30 个独立样品时 5 折只剩 3–6 个测试样品，单折指标不可解释；
        # SKILL.md 的样品量—折数对照表也把这一段定为 3 折，这里必须与之一致。
        rec["outer_n_splits"] = 3
        rec["note_outer"] = (f"{n_groups} 个 group：外层取 3 折（每折约 {n_groups // 3} 个测试样品）；"
                             f"5 折会把测试折压到 {n_groups // 5} 个以内，单折 R² 更不可靠")
    else:
        rec["outer_n_splits"] = min(5, n_groups)

    min_test_rows = ginfo["rows_per_group_median"] * n_groups / max(rec["outer_n_splits"], 2)
    if min_test_rows < 10:
        rec["warn_small_test"] = "外层测试折平均不到 10 个样品，单折 R² 不可解释，只能看 repeat 级聚合"

    if ds.task == "classification":
        counts = pd.Series(ds.y).value_counts()
        n_min = int(counts.min())
        rec["inner_n_splits"] = max(2, min(int(cfg["split"]["inner"]["n_splits"]), n_min))
        rec["stratify"] = True
        if n_min < rec["inner_n_splits"]:
            rec["warn_stratify"] = f"最小类只有 {n_min} 个样本，内层分层折数降到 {rec['inner_n_splits']}"
    else:
        rec["inner_n_splits"] = max(2, min(3, n_groups - 1))
        rec["stratify"] = False

    if not ginfo["has_replicates"]:
        rec["warn_no_replicate"] = (
            "每个 group 只有 1 行：分组 CV 与随机 CV 等价，无法演示抗泄漏能力；"
            "若同一材料确实重复测量过，请补 sample_id"
        )
    if ds.batch is not None:
        n_batch = len(pd.unique(ds.batch))
        rec["n_batch"] = n_batch
        if n_batch >= 3:
            rec["suggest_loo_batch"] = "批次 >=3，建议额外跑 split.strategy=loo_batch 作为真外推验证"
        elif n_batch == 2:
            rec["suggest_loo_batch"] = "仅 2 个批次，留一批次外验证只剩单折，只能作参考"
    else:
        rec["warn_no_batch"] = "未提供 batch 列，无法做留一批次外部验证；报告中需在 limitations 说明"
    return rec


def _nn_distances(X_te: np.ndarray, X_tr: np.ndarray, chunk: int = 256) -> np.ndarray:
    """每个测试行到训练集的最小欧氏距离（分块算，不建 n×m×p 的整块数组）。"""
    out = np.empty(len(X_te), dtype=float)
    for i in range(0, len(X_te), chunk):
        block = X_te[i:i + chunk]
        d2 = ((block[:, None, :] - X_tr[None, :, :]) ** 2).sum(-1)
        out[i:i + chunk] = np.sqrt(d2.min(1))
    return out


def _duplicate_structure(ds: Dataset) -> dict:
    """完全重复行落在几个 group 上。

    只报「N 行完全重复」是不够的（UCI wine 就有 240 行重复）：如果每行各自一个
    sample_id，分组 CV 也拦不住——同一个特征向量会同时出现在训练折和测试折，
    OOF accuracy / R² 会被抬高，而读者从「未检测到重复测量」看不出来。
    """
    out = {"duplicate_clusters": 0, "duplicate_rows_cross_group": 0}
    try:
        X = np.asarray(ds.X, dtype=np.float64)
        _, inv, counts = np.unique(X, axis=0, return_inverse=True, return_counts=True)
        clusters = np.flatnonzero(counts > 1)
        if clusters.size == 0:
            return out
        groups = np.asarray(ds.groups, dtype=object)
        n_cross = 0
        for c in clusters:
            idx = np.flatnonzero(inv == c)
            if len(pd.unique(groups[idx])) > 1:
                # 只有"副本"（簇内除首行外的行）才算重复行，否则会把簇的原始行也计入，
                # 报出 cross > duplicate_rows 的自相矛盾数（240 行重复却写 460 行跨组）。
                n_cross += int(idx.size - 1)
        out = {"duplicate_clusters": int(clusters.size), "duplicate_rows_cross_group": n_cross}
    except Exception:  # noqa: BLE001 —— 纯诊断，算不出来不阻断审计
        return out
    return out


def _batch_overlap(ds: Dataset, cfg: dict, max_rows: int = 3000) -> dict | None:
    """批次之间的样品重叠：完全重复行 + 最近邻距离。

    为什么必须在 audit 里做：`holdout_batch` / `loo_batch` 的全部说服力都来自
    「测试批没见过」。但公开数据集里常见同一样品在两批里各测一次（Tecator 的
    240 行里有 22 对重复光谱），此时所谓外部测试其实是内插，RPD 能报到 20+。
    全表「完全重复行」只说明有重复，不说明重复**跨进了测试批**——这两件事
    对结论的影响完全不同。
    """
    batch = getattr(ds, "batch", None)
    if batch is None:
        return None
    batch = np.asarray(batch, dtype=object).astype(str)
    uniq = list(pd.unique(batch))
    if len(uniq) < 2:
        return None
    X = np.asarray(ds.X, dtype=np.float64)
    if X.shape[0] > max_rows:
        return {"skipped": f"行数 {X.shape[0]} > {max_rows}，未做跨批次近邻检查"}
    split = cfg.get("split") or {}
    strategy = str(split.get("strategy", "nested_group"))
    focus = None
    if strategy == "holdout_batch" and split.get("holdout_batch") is not None:
        focus = str(split.get("holdout_batch"))

    per: dict[str, dict[str, Any]] = {}
    for b in uniq:
        te = np.flatnonzero(batch == b)
        tr = np.flatnonzero(batch != b)
        if te.size == 0 or tr.size < 2:
            continue
        dmin = _nn_distances(X[te], X[tr])
        exact = int(np.isclose(dmin, 0.0, atol=1e-9).sum())
        # 训练集内部的最近邻距离 = 「正常间距」的基准；测试批比它还近，就是内插。
        rng = np.random.default_rng(0)
        sub = tr if tr.size <= 300 else rng.choice(tr, size=300, replace=False)
        base = []
        for j, r in enumerate(sub):
            d = np.sqrt(((X[sub] - X[r]) ** 2).sum(1))
            d[j] = np.inf
            base.append(float(np.min(d)))
        base_med = float(np.median(base))
        med = float(np.median(dmin))
        per[b] = {
            "n_rows": int(te.size),
            "exact_duplicates": exact,
            "exact_rate": round(exact / max(1, int(te.size)), 4),
            "nn_median": round(med, 4),
            "train_nn_median": round(base_med, 4),
            "nn_ratio": round(med / base_med, 3) if base_med > 0 else None,
        }
    if not per:
        return None
    inside = {b: v for b, v in per.items() if focus is None or b == focus}
    return {
        "strategy": strategy,
        "focus_batch": focus,
        "per_batch": per,
        "n_exact_duplicates_in_focus": int(sum(v["exact_duplicates"] for v in inside.values())),
        "min_nn_ratio_in_focus": (min([v["nn_ratio"] for v in inside.values()
                                       if v.get("nn_ratio") is not None], default=None)),
        "note": ("最近邻距离为原始特征空间的欧氏距离；nn_ratio = 测试批到训练批的中位最近邻距离 / "
                 "训练集内部的中位最近邻距离，<1 说明测试批落在训练云内部"),
    }


def audit_dataset(ds: Dataset, cfg: dict, raw_df: pd.DataFrame | None = None) -> AuditReport:
    rep = AuditReport()
    gstats, ginfo = _group_structure(ds)
    rep.group_stats = gstats
    fissues = _feature_issues(ds)
    rep.feature_issues = fissues
    coll = _collinearity(ds)
    leak = _leakage_scan(ds)
    rep.leakage_table = leak
    pseudo = _pseudo_id_scan(ds)
    epv = _epv(ds)
    rec = _recommend(ds, ginfo, cfg)
    # 建议文案必须写实际生效的分组键：按 mixture_id 分组的 run 若被建议成
    # 「确保 sample_id 相同」，作者会照着改错列（与报告方法学段落同源，见 1.2.11）。
    gkey = cfg["split"].get("group_col") or cfg["data"].get("id_col") or "分组键"

    miss_frac = float(np.isnan(ds.X).mean()) if ds.n_features else 0.0
    dup_rows = int(ds.X.shape[0] - len(np.unique(np.asarray(ds.X, dtype=np.float64), axis=0)))
    dup_info = _duplicate_structure(ds)
    overlap = _batch_overlap(ds, cfg)

    summary = {
        "n_rows": ds.n_samples,
        "n_features": ds.n_features,
        "n_groups": ginfo["n_groups"],
        "task": ds.task,
        "target": ds.target_name,
        "data_hash": ds.data_hash,
        "source": ds.source_path,
        "missing_fraction": miss_frac,
        "duplicate_rows": dup_rows,
        **dup_info,
        "n_constant_features": int(fissues["constant"].sum()),
        "n_near_zero_var": int(fissues["near_zero_var"].sum()),
        "cond_number": coll.get("cond"),
        "effective_rank": coll.get("effective_rank"),
        "max_abs_pairwise_r": coll.get("max_abs_r"),
        "n_collinear_pairs": coll.get("n_pairs_above"),
        "collinear_sampled": bool(coll.get("sampled", False)),
        "collinear_pairs": coll.get("pairs", []),
        "group_structure": ginfo,
        "epv": epv,
        "batch_overlap": overlap,
    }
    if ds.task == "regression":
        yv = np.asarray(ds.y, dtype=np.float64)
        summary["target_stats"] = {
            "min": float(np.nanmin(yv)), "max": float(np.nanmax(yv)),
            "mean": float(np.nanmean(yv)), "sd": float(np.nanstd(yv, ddof=1)),
            "skew": float(stats.skew(yv, nan_policy="omit")),
            "n_unique": int(len(np.unique(yv))),
        }
    else:
        summary["target_stats"] = epv.get("class_counts", {})

    # ---------------- findings ----------------
    F = rep.findings

    if ginfo["has_replicates"]:
        F.append(Finding(
            "info", "重复测量结构", key="replicates",
            detail=(f"{ginfo['n_groups']} 个独立样品、{ginfo['n_rows']} 行测量，"
                    f"每样品 {ginfo['rows_per_group_min']}–{ginfo['rows_per_group_max']} 行"),
            action=f"必须使用 GroupKFold(groups={gkey})，随机 KFold 会用同一样品的另一次测量预测自己",
        ))
    else:
        F.append(Finding(
            "warn", "未检测到重复测量", key="no_replicate",
            detail="每个 group 只有一行，分组 CV 与普通 K 折完全等价",
            action="确认是否漏了 sample_id；若确实无重复测量，报告中不要声称「采用分组交叉验证可避免重复测量泄漏」",
        ))

    if pseudo:
        F.append(Finding(
            "critical", "疑似把 ID 当特征", key="pseudo_id",
            detail=f"{len(pseudo)} 列列名像编号且取值近乎唯一：{pseudo[:6]}",
            action="把这几列加入 data.drop_cols，否则模型会学到编号与顺序的伪关系",
        ))

    if not leak.empty:
        if ds.task == "regression":
            top = leak.iloc[0]
            if top["abs_value"] >= LEAK_SPEARMAN:
                F.append(Finding(
                    "critical", "疑似目标泄漏（回归）", key="leak_reg",
                    detail=f"单特征 {top['feature']} 与目标的 |Spearman| = {top['abs_value']:.3f} ≥ {LEAK_SPEARMAN}",
                    action="该特征很可能是目标的派生量（如换算后的浓度、标签的中间量），必须剔除后再建模",
                ))
            elif top["abs_value"] >= LEAK_SPEARMAN_NEAR:
                F.append(Finding(
                    "warn", "疑似近克隆特征（目标泄漏嫌疑）", key="leak_reg_near",
                    detail=(f"最强单特征 {top['feature']} 与目标的 |Spearman| = {top['abs_value']:.3f}"
                            f"（{LEAK_SPEARMAN_NEAR} ≤ |r| < {LEAK_SPEARMAN}）："
                            f"单列已能解释 ≥{top['abs_value'] ** 2:.0%} 的方差，量级上更像目标的派生量"),
                    action=(f"先核对 {top['feature']} 是否由目标换算而来（效率/收率/换算浓度/中间量）；"
                            "确认后加入 data.drop_cols 重跑，不要因为它没到 0.98 就放着"),
                ))
            else:
                F.append(Finding(
                    "ok", "无显著目标泄漏", key="leak_reg_ok",
                    detail=f"最强单特征 |Spearman| = {top['abs_value']:.3f}（阈值 {LEAK_SPEARMAN}）",
                ))
        else:
            top = leak.iloc[0]
            if top["abs_value"] >= LEAK_AUC:
                F.append(Finding(
                    "critical", "疑似目标泄漏（分类）", key="leak_clf",
                    detail=f"单特征 {top['feature']} 的独立 AUC = {top['value']:.4f} ≥ {LEAK_AUC}",
                    action="单变量就能完美分类，通常是泄漏或标签编码错误；先做该特征的分布检查",
                ))
            elif top["abs_value"] >= LEAK_AUC_NEAR:
                F.append(Finding(
                    "warn", "疑似近克隆变量（分类）", key="leak_clf_near",
                    detail=(f"最强单变量 {top['feature']} 的独立 AUC = {top['value']:.4f}"
                            f"（{LEAK_AUC_NEAR} ≤ AUC < {LEAK_AUC}）：单变量已接近完美区分"),
                    action=(f"核对 {top['feature']} 是否与标签同源（如按标签分组后的派生统计量、"
                            "事后计算的比率），确认后加入 data.drop_cols"),
                ))
            else:
                F.append(Finding(
                    "ok", "无显著目标泄漏", key="leak_clf_ok",
                    detail=f"最强单变量 AUC = {top['value']:.4f}（阈值 {LEAK_AUC}）",
                ))

    n_const = int(fissues["constant"].sum())
    if n_const:
        F.append(Finding(
            "warn", "常数特征", key="const",
            detail=f"{n_const} 列方差为 0（如某波段全为 0 或超出仪器量程）",
            action="自动剔除；若这些列本应有信号，检查数据导出时的列对齐",
        ))
    n_nzv = int((fissues["near_zero_var"] & ~fissues["constant"]).sum())
    if n_nzv:
        F.append(Finding(
            "info", "近零方差特征", key="nzv",
            detail=f"{n_nzv} 列近零方差（std<{NEAR_ZERO_CV}）",
            action="保留但标准化时会放大噪声，建议在预处理里做波段裁剪",
        ))

    if coll.get("cond") is not None and coll["cond"] > 1e8:
        F.append(Finding(
            "warn", "严重共线性", key="cond",
            detail=(f"标准化设计矩阵条件数 {coll['cond']:.2e}，有效秩 {coll['effective_rank']} / {ds.n_features}"
                    + (f"，|r|>{COLLINEAR_R} 的成对特征 {coll['n_pairs_above']} 对" if coll.get("n_pairs_above") else "")),
            action=("这是光谱数据的常态信号：不要用 OLS 或未加正则的线性模型出结论，"
                    "优先 PLS / Ridge / ElasticNet；VIP 与 permutation 重要性比单系数可靠"),
        ))

    if epv["ratio_n_p"] is not None and epv["ratio_n_p"] < 0.1:
        F.append(Finding(
            "info", "p >> n 结构", key="p_gt_n",
            detail=f"样本/特征 = {epv['ratio_n_p']:.3f}，n/√p = {epv.get('n_vs_sqrt_p', float('nan')):.2f}",
            action="降维必须在交叉验证内部完成（Pipeline），严禁先对全量数据做 PCA 再切分",
        ))

    if ds.task == "classification":
        ratio = epv.get("imbalance_ratio")
        if ratio and ratio >= 3:
            F.append(Finding(
                "warn", "类别不平衡", key="imbalance",
                detail=f"类别计数 {epv.get('class_counts')}，不平衡比 {ratio:.2f}",
                action=("主指标用 AUPRC / balanced accuracy / MCC，不要只看 accuracy；"
                        "管线硬标签指标固定 0.5 阈值（未做内层 Youden 选择），"
                        "阈值敏感性见报告表 T8；若论文要声称阈值为 Youden，须自行在内层折选择并记录每折阈值"),
            ))

    if dup_rows:
        cross = int(dup_info.get("duplicate_rows_cross_group") or 0)
        detail = f"{dup_rows} 行特征向量完全重复（{dup_info.get('duplicate_clusters', 0)} 组）"
        if cross:
            detail += f"，其中 {cross} 行的副本落在不同 group"
            action = ("这 {cross} 行会被分组 CV 分到不同折：训练折里已有同一行，测试折等于抄答案，"
                      "OOF 指标偏高。先去重（保留每组一行）再重跑，把两次结果并列报告"
                      ).format(cross=cross)
        else:
            action = (f"检查是否同一测量被重复录入；若确为独立重复测量，确保 {gkey} 相同"
                      f"（分组会处理）")
        F.append(Finding("warn", "完全重复行", key="dup_rows", detail=detail, action=action))

    # 跨批次重叠：决定 holdout_batch / loo_batch 的结论还算不算「外部验证」
    if overlap and overlap.get("per_batch"):
        focus = overlap.get("focus_batch")
        scope = f"（本次作为测试批的 batch={focus}）" if focus else ""
        n_exact = overlap.get("n_exact_duplicates_in_focus") or 0
        ratio = overlap.get("min_nn_ratio_in_focus")
        if n_exact:
            worst = (int(overlap["per_batch"][focus]["n_rows"]) if focus else
                     sum(v["n_rows"] for v in overlap["per_batch"].values()))
            F.append(Finding(
                "warn", "测试批与训练批存在完全重复样品", key="batch_dup",
                detail=(f"{scope} 有 {n_exact} 行与训练批的光谱逐点相同"
                        f"（占该批 {n_exact / max(1, worst):.1%}）"),
                action=("该批不能当作独立外部测试：报告须改写为「批次内一致性」，"
                        "或剔除重复样品后重跑，否则 RPD/RMSEP 会被近邻重叠抬高"),
            ))
        elif ratio is not None and ratio < 0.8:
            F.append(Finding(
                "warn", "测试批落在训练云内部（近邻重叠）", key="batch_overlap",
                detail=(f"{scope} 测试批到训练批的中位最近邻距离只有训练集内部间距的 {ratio:.2f} 倍"
                        f"（<0.8 说明测试点几乎被训练点夹住）"),
                action=("外部测试接近内插：R²/RPD 会明显偏高，写论文时须说明测试批与训练批的"
                        "谱空间重叠，不能把该数字当作外推精度"),
            ))

    if miss_frac > 0:
        F.append(Finding(
            "info", "缺失值", key="missing",
            detail=f"缺失比例 {miss_frac:.2%}（保留到 Pipeline 内按训练折中位数插补）",
            action="中位数插补会低估方差；若缺失 >5%，改用迭代插补并做敏感性分析",
        ))

    if ds.batch is None:
        F.append(Finding(
            "info", "无批次信息", key="no_batch",
            detail="未提供 batch 列，无法做留一批次外部验证",
            action="报告中 limitations 必须写明：模型仅在当前批次/仪器内验证，跨批次外推未验证",
        ))

    tgt_all = [str(t) for t in ((cfg.get("task") or {}).get("target") or [])]
    excl = [str(c) for c in ((cfg.get("data") or {}).get("exclude_from_features") or [])]
    if len(tgt_all) > 1 or excl:
        names = list(dict.fromkeys(tgt_all + excl))
        F.append(Finding(
            "info", "目标列未进入特征矩阵", key="targets_excluded",
            detail=(f"已从特征中剔除：{', '.join(names)}"
                    + ("（多目标兄弟列）" if len(tgt_all) > 1 else "")),
            action=("这是必要处理：用「蛋白」列预测「水分」会白拿高分，而通用泄漏检查"
                    "（单特征完美预测）识别不出来。核对被剔除的列是否确实只含目标与显式排除项"),
        ))

    if len(ds.target_names) > 1:
        counts = {t: int(ds.target_available.get(t, ds.n_samples)) for t in ds.target_names}
        thin = [t for t, n in counts.items() if n < ds.n_samples]
        if thin:
            detail = "；".join(f"{t} {n}/{ds.n_samples} 行" for t, n in counts.items())
            F.append(Finding(
                "warn", "多目标各自的有效行数不同", key="multi_target_rows",
                detail=(f"已按目标各自的 y 有效值建模，不再因兄弟目标缺行整行剔除：{detail}。"
                        + (f"所有目标均缺失而剔除 {ds.dropped_all_targets} 行；"
                           if ds.dropped_all_targets else "")),
                action=("报告里每个目标的 n 要分别写（例如厚度 n=14、其余 n=15），"
                        "不要把不同目标的性能放在同一张表里按同一 n 比较；"
                        "若某目标样本量过小，可单独跑：--set task.target=[\"该目标\"]"),
            ))
    elif ds.dropped_all_targets:
        F.append(Finding(
            "info", "目标缺失行已剔除", key="target_missing_rows",
            detail=f"{ds.dropped_all_targets} 行目标值为缺失，已从建模数据中剔除",
        ))

    # 适用域判据在小样本下会先验失效：杠杆阈值 h*=3(p+1)/n 在 n 接近 p 时趋近 1，
    # 几乎必然越界，Williams 图与 AD 有效性检验都没有解释力。必须在 audit 阶段就说，
    # 而不是等 ad 阶段跑完再补一句「判据未显示判别力」。
    ad_cfg = cfg.get("ad") or {}
    if ad_cfg.get("enabled", True):
        n = int(ds.n_groups)
        p_eff = int(min(int(ad_cfg.get("pca_max_comp", 10)), max(1, n - 1), ds.n_features))
        h_star = 3.0 * (p_eff + 1) / max(1, n)
        if n < 5 * (p_eff + 1) or h_star >= 1.0:
            n_deg = 0
            detail = (f"独立样品 n={n}，PCA 成分上限 p={p_eff}：杠杆阈值 "
                      f"h*=3(p+1)/n={h_star:.3f}"
                      + ("（≥1：杠杆值恒 ≤1，该法永远不越界，等于失效）" if h_star >= 1.0 else "")
                      + f"；n < 5(p+1)={5 * (p_eff + 1)} 时经验分位阈值不可靠")
            if p_eff >= int(ds.n_features):
                n_deg += 1
                detail += ("；PCA 保留全部 p 个主成分时解释方差 = 100%，"
                           "Q 残差恒为 0、阈值退化为数值零（任何浮点噪声都判域外）")
            if h_star >= 1.0:
                n_deg += 1
            if n <= p_eff + 1:
                detail += "，T² 的 F 理论阈值也无法计算"
            F.append(Finding(
                "warn", "适用域判据在小样本下退化", key="ad_degenerate",
                detail=detail,
                action=(f"四法里有 {n_deg} 法在本次数据上退化：把 ad.methods 里的 leverage"
                        + ("、pca_q" if p_eff >= int(ds.n_features) else "")
                        + " 去掉（ad 阶段会自动剔除退化方法并在 T5 标注），"
                        "或只保留 AD 的定义与阈值描述、不宣称判别力；"
                        "n<5(p+1) 时 AD 结论只能视为探索性，不得写「模型可安全外推」"),
            ))

    # 非数值列被静默丢弃：German Credit 20 列里 13 列是类别（A11/A12…），
    # 旧版只留一条 info「载入说明」，报告正文照样写「1000 行 × 7 特征」，
    # 读者不会意识到 65% 的列根本没进模型，AUC 从文献的 0.78 掉到 0.67 也无从解释。
    dropped_cat = list(getattr(ds, "dropped_non_numeric", []) or [])
    if dropped_cat:
        n_used = int(ds.X.shape[1])
        frac = len(dropped_cat) / max(1, n_used + len(dropped_cat))
        F.append(Finding(
            "warn" if len(dropped_cat) >= 3 else "info",
            f"类别/非数值列被丢弃：{len(dropped_cat)} 列（占原始特征 {frac:.0%}），"
            f"模型实际只用 {n_used} 列",
            key="non_numeric_dropped",
            detail=("这些列 dtype 不是数值、又没有写进 feature_cols，所以没有进入特征矩阵："
                    f"{dropped_cat[:12]}" + ("…" if len(dropped_cat) > 12 else "")
                    + "。若它们本应是类别特征（如 checking_status / purpose / housing），"
                    "当前模型对它们完全不可见，任何「用了全部变量」的说法都不成立。"),
            action=("先用 pd.get_dummies(..., drop_first=False) 把类别列展开成 0/1 列、"
                    "或写 data.feature_cols 指向展开后的列，再重跑；"
                    "整数编码的标称列（0/1/2…）不会被这里识别，同样必须显式 one-hot，"
                    "否则会被当成有序数值、给出并不存在的「每升一级风险+Δ」结论"),
        ))

    for note in ds.notes:
        F.append(Finding("info", "载入说明", key="io_note", detail=note))

    rep.summary = summary
    rep.recommendations = rec
    return rep


# --------------------------------------------------------------------------- #
def render_audit_markdown(rep: AuditReport) -> str:
    s = rep.summary
    icon = {"ok": "OK ", "info": "·  ", "warn": "!  ", "critical": "XX "}
    lines: list[str] = []
    lines.append("# 数据审计报告\n")
    lines.append(f"- 数据文件：`{s.get('source')}`")
    lines.append(f"- 数据指纹：`{str(s.get('data_hash'))[:16]}…`")
    lines.append(f"- 规模：{s['n_rows']} 行 × {s['n_features']} 特征 × {s['n_groups']} 个独立样品")
    lines.append(f"- 任务：{s['task']}｜目标：`{s['target']}`")
    lines.append("")
    lines.append("## 一、关键发现\n")
    order = {"critical": 0, "warn": 1, "info": 2, "ok": 3}
    for f in sorted(rep.findings, key=lambda x: order.get(x.level, 9)):
        lines.append(f"### {icon.get(f.level, '')} {f.title}\n")
        lines.append(f"{f.detail}\n")
        if f.action:
            lines.append(f"> 建议：{f.action}\n")
    lines.append("## 二、样本量与统计效力\n")
    e = s["epv"]
    lines.append(f"- 样本/特征比：{float(e.get('ratio_n_p') or 0):.4f}")
    lines.append(f"- n/√p：{float(e.get('n_vs_sqrt_p') or 0):.2f}")
    if s["task"] == "classification":
        lines.append(f"- EPV（最小类样本/特征数）：{e.get('epv')}")
        lines.append(f"- 类别计数：{e.get('class_counts')}")
    else:
        ts = s.get("target_stats", {})
        lines.append(f"- 目标：min={ts.get('min'):.4g} max={ts.get('max'):.4g} "
                     f"mean={ts.get('mean'):.4g} sd={ts.get('sd'):.4g}")
        lines.append(f"- 偏度：{ts.get('skew'):.3f}" + ("（右偏明显，考虑对数变换后重跑）" if abs(ts.get("skew") or 0) > 1 else ""))
    lines.append("")
    lines.append("## 三、共线性与有效自由度\n")
    cond = s.get("cond_number")
    lines.append(f"- 条件数：{cond:.3e}" if cond else "- 条件数：无法计算")
    lines.append(f"- 有效秩（数值秩）：{s.get('effective_rank')} / {s['n_features']}")
    mar = s.get("max_abs_pairwise_r")
    if mar is None:
        lines.append("- 最大成对 |r|：无法计算（常数特征过多）")
    else:
        suffix = "（抽样估计）" if s.get("collinear_sampled") else ""
        lines.append(f"- 最大成对 |r|：{float(mar):.4f}{suffix}")
    lines.append(f"- |r|>0.99 的成对特征数：{s.get('n_collinear_pairs')}")
    pairs = s.get("collinear_pairs") or []
    if pairs:
        lines.append("")
        lines.append("| 特征 A | 特征 B | \\|r\\| |")
        lines.append("|---|---|---|")
        for pr in pairs[:8]:
            lines.append(f"| {pr['a']} | {pr['b']} | {pr['r']:.4f} |")
    lines.append("")
    lines.append("## 四、分组结构（防泄漏依据）\n")
    gs = s["group_structure"]
    lines.append(f"- group 数：{gs['n_groups']}，每 group 行数 {gs['rows_per_group_min']}–{gs['rows_per_group_max']}"
                 f"（中位 {gs['rows_per_group_median']}）")
    lines.append(f"- 单行 group 数：{gs['n_groups_single_row']}")
    lines.append("")
    if not rep.group_stats.empty:
        lines.append("| group | 行数 |")
        lines.append("|---|---|")
        for _, row in rep.group_stats.head(10).iterrows():
            lines.append(f"| {row['group']} | {row['n_rows']} |")
        if len(rep.group_stats) > 10:
            lines.append(f"| …（共 {len(rep.group_stats)} 组） | |")
        lines.append("")
    lines.append("## 五、与目标关联最强的单特征（泄漏排查）\n")
    if rep.leakage_table.empty:
        lines.append("（无法计算：存在常数特征或样本不足）\n")
    else:
        lines.append("| 特征 | 统计量 | 值 |")
        lines.append("|---|---|---|")
        for _, row in rep.leakage_table.iterrows():
            lines.append(f"| {row['feature']} | {row['stat']} | {row['value']:.4f} |")
        lines.append("")
    lines.append("## 六、建议的验证方案\n")
    for k, v in rep.recommendations.items():
        lines.append(f"- **{k}**：{v}")
    lines.append("")
    lines.append("## 七、下一步配置建议\n")
    rec = rep.recommendations
    lines.append("```json")
    lines.append('"split": {')
    lines.append(f'  "outer": {{"n_splits": {rec.get("outer_n_splits", 5)}}},')
    lines.append(f'  "inner": {{"n_splits": {rec.get("inner_n_splits", 3)}}},')
    lines.append(f'  "stratify": {str(bool(rec.get("stratify", False))).lower()}')
    lines.append("}")
    lines.append("```")
    return "\n".join(lines) + "\n"
