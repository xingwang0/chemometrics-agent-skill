"""可解释性：波段级置换重要性、PLS VIP、跨折系数稳定性、SHAP（补充）。

为什么顺序是这样
----------------
高维共线光谱上，单波长的重要性**本质上不可识别**——相邻波长的信息几乎重复，
你把其中一列打乱，模型立刻用邻居补上，Δ指标≈0，于是"重要波段"被稀释成一堆噪声。
所以本模块的第一选择是**按波段聚合后置换**（band_size 个连续波长视为一个变量）。

SHAP 在这里只作补充：它把贡献分摊到共线的各列上，结果是"一组同样重要的相邻峰"，
在数学上不唯一。把它当主图会被懂行的审稿人反打，因此默认只在树模型且已降维时输出，
并在图上强制加注解。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .cv import effective_repeats, make_outer_splits
from .models import ModelSpec, build_pipeline
from .utils import info, warn

__all__ = [
    "build_band_groups", "band_permutation_importance", "fit_fold_models",
    "vip_scores", "coef_stability", "shap_values_safe", "top_features",
]

MAX_BANDS = 200


def _is_numeric_name(name: str) -> bool:
    try:
        float(str(name))
        return True
    except ValueError:
        return False


def build_band_groups(feature_names: list[str], band_size: int = 25) -> list[dict[str, Any]]:
    """把特征分组：数值型列名（波数）按连续块聚成波段，其它列各自成组。"""
    if not feature_names:
        return []
    band_size = max(1, int(band_size))
    groups: list[dict[str, Any]] = []
    spectral_idx = [i for i, n in enumerate(feature_names) if _is_numeric_name(n)]
    other_idx = [i for i, n in enumerate(feature_names) if not _is_numeric_name(n)]

    for i in other_idx:
        groups.append({"label": str(feature_names[i]), "indices": [i], "kind": "meta"})

    if spectral_idx:
        contiguous: list[list[int]] = []
        run = [spectral_idx[0]]
        for prev, cur in zip(spectral_idx, spectral_idx[1:]):
            if cur == prev + 1:
                run.append(cur)
            else:
                contiguous.append(run)
                run = [cur]
        contiguous.append(run)
        for run in contiguous:
            for start in range(0, len(run), band_size):
                chunk = run[start:start + band_size]
                try:
                    lo = float(feature_names[chunk[0]])
                    hi = float(feature_names[chunk[-1]])
                    label = f"{lo:.0f}–{hi:.0f}"
                except ValueError:
                    label = f"{feature_names[chunk[0]]}~{feature_names[chunk[-1]]}"
                groups.append({"label": label, "indices": chunk, "kind": "band",
                               "center": float(np.mean([float(feature_names[i]) for i in chunk]))
                               if _is_numeric_name(feature_names[chunk[0]]) else float("nan")})
    if len(groups) > MAX_BANDS:
        warn(f"波段数 {len(groups)} > {MAX_BANDS}，请增大 interp.band_size，否则置换检验会很慢")
    return groups


def fit_fold_models(
    ctx,
    spec: ModelSpec,
    folds: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """按 cv 阶段选出的逐折最优超参重新拟合模型，供置换重要性与系数稳定性使用。

    （默认 save_models="last" 只留最后一个折的模型，所以这里按需重拟合。）
    """
    cfg = ctx.cfg
    ds = ctx.ds
    y = np.asarray(ds.targets[ctx.target])
    out: list[dict[str, Any]] = []
    repeats = effective_repeats(cfg)
    import json

    best_by_fold: dict[tuple[int, int], dict[str, Any]] = {}
    if folds is not None and not folds.empty:
        status = folds["status"].astype(str) if "status" in folds.columns else pd.Series("ok", index=folds.index)
        sub = folds[(folds["model"] == spec.name) & status.isin(["ok", "degenerate"])]
        for _, row in sub.iterrows():
            try:
                best_by_fold[(int(row["repeat"]), int(row["fold"]))] = json.loads(row["best_params"])
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

    for r in range(repeats):
        outer = make_outer_splits(ds, cfg, int(cfg["runtime"]["seed"]) + r * 1000)
        for k, (tr, te) in enumerate(outer):
            params = best_by_fold.get((r, k), {})
            pipe = build_pipeline(spec, cfg, ds.feature_names,
                                  seed=(int(cfg["runtime"]["seed"]) + r * 1000) * 100 + k)
            clean = {kk: vv for kk, vv in params.items()
                     if kk.startswith(("model__", "prep__", "scale__"))}
            if clean:
                try:
                    pipe.set_params(**clean)
                except ValueError:
                    pass
            pipe.fit(ds.X[tr], y[tr])
            out.append({"repeat": r, "fold": k, "tr": tr, "te": te,
                        "model": pipe, "params": params})
    info(f"为可解释性重拟合 {len(out)} 个折模型（{spec.name}）")
    return out


def _predict_score(pipe, X: np.ndarray, task: str) -> tuple[np.ndarray, str]:
    """返回用于计算指标的分数；分类优先用概率，绝不拿硬标签算 AUC。"""
    if task == "regression":
        return np.asarray(pipe.predict(X)), "none"
    if hasattr(pipe, "predict_proba"):
        p = np.asarray(pipe.predict_proba(X), dtype=np.float64)
        if p.ndim == 2 and p.shape[1] == 2:
            return p[:, 1], "native"
        return p, "native"
    if hasattr(pipe, "decision_function"):
        d = np.asarray(pipe.decision_function(X), dtype=np.float64)
        if d.ndim == 1:
            return 1.0 / (1.0 + np.exp(-d)), "decision"
        z = d - d.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True), "decision"
    return np.asarray(pipe.predict(X)), "none"


def band_permutation_importance(
    ctx,
    fold_models: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    n_repeats: int = 10,
    seed: int = 0,
    max_bands: int = MAX_BANDS,
) -> pd.DataFrame:
    """波段级置换重要性：把整段波长同时打乱，衡量 OOF 指标的下降。"""
    from .metrics import classification_metrics, regression_metrics

    ds = ctx.ds
    task = ds.task
    y = np.asarray(ds.targets[ctx.target])
    X = ds.X
    n_classes = 1 if task == "regression" else int(len(ds.classes or [2]))
    pairs = [g for g in groups if len(g["indices"]) > 0][:max_bands]
    if not pairs:
        return pd.DataFrame()

    def score(preds: dict[tuple[int, int], np.ndarray],
              kinds: dict[tuple[int, int], str]) -> float:
        truth, est, ks = [], [], []
        for fm in fold_models:
            key = (fm["repeat"], fm["fold"])
            if key not in preds:
                continue
            truth.append(y[fm["te"]])
            est.append(preds[key])
            ks.append(kinds.get(key, "native"))
        if not truth:
            return float("nan")
        yt, yp = np.concatenate(truth), np.concatenate(est)
        if task == "regression":
            return float(regression_metrics(yt, yp)["r2"])
        proba_kind = "native" if all(k == "native" for k in ks) else (
            "decision" if all(k in ("native", "decision") for k in ks) else "none")
        return float(classification_metrics(yt, yp, n_classes=n_classes,
                                            proba_kind=proba_kind)["auc"])

    base_preds, base_kinds = {}, {}
    for fm in fold_models:
        key = (fm["repeat"], fm["fold"])
        base_preds[key], base_kinds[key] = _predict_score(fm["model"], X[fm["te"]], task)
    base_score = score(base_preds, base_kinds)
    rng = np.random.default_rng(seed)
    rows = []

    # 留一法（每折只留 1 个测试样品）下，对单行做列置换是恒等操作：ΔR² 必然全为 0。
    # 旧实现不判这一点，直接把「全零」当结果写进报告（「贡献最大的波段 ΔR²=0.0000」），
    # 读者会以为没有重要特征。这里改为全样本置换：把该列在全部样品上打乱，
    # 用各折模型对全量 X 打分后按行平均再算指标（含内样本偏差，只用于排序）。
    min_test = min((len(fm["te"]) for fm in fold_models), default=0)
    all_rows = min_test < 2

    def _avg_pred(Xmat: np.ndarray) -> np.ndarray:
        cols = []
        for fm in fold_models:
            p, _kind = _predict_score(fm["model"], Xmat, task)
            p = np.asarray(p, dtype=float)
            if p.ndim == 2:
                p = p[:, -1]
            cols.append(p)
        return np.mean(np.vstack(cols), axis=0)

    def _score_all(pred: np.ndarray) -> float:
        if task == "regression":
            return float(regression_metrics(y, pred)["r2"])
        return float(classification_metrics(y, pred, n_classes=n_classes,
                                            proba_kind="native")["auc"])

    if all_rows:
        warn("每折测试样品 < 2（如留一法），列置换在单行上是恒等操作："
             "改用全样本置换，ΔR² 含内样本偏差，仅用于特征排序")
        base_score = _score_all(_avg_pred(X))
    for gi, grp in enumerate(pairs):
        deltas: list[float] = []
        idx = np.asarray(grp["indices"], dtype=int)
        for _ in range(int(n_repeats)):
            if all_rows:
                Xp = X.copy()
                perm = rng.permutation(Xp.shape[0])
                Xp[:, idx] = Xp[perm][:, idx]
                s = _score_all(_avg_pred(Xp))
                if np.isfinite(s):
                    deltas.append(base_score - s)
                continue
            preds, kinds = {}, {}
            for fm in fold_models:
                Xte = X[fm["te"]].copy()
                perm = rng.permutation(Xte.shape[0])
                Xte[:, idx] = Xte[perm][:, idx]
                key = (fm["repeat"], fm["fold"])
                preds[key], kinds[key] = _predict_score(fm["model"], Xte, task)
            s = score(preds, kinds)
            if np.isfinite(s):
                deltas.append(base_score - s)
        if not deltas:
            continue
        rows.append({
            "band_index": gi,
            "label": grp["label"],
            "kind": grp["kind"],
            "center": grp.get("center", np.nan),
            "n_features": len(grp["indices"]),
            "delta_mean": float(np.mean(deltas)),
            "delta_sd": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
            "delta_min": float(np.min(deltas)),
            "delta_max": float(np.max(deltas)),
            "n_repeats": len(deltas),
        })
        if (gi + 1) % 25 == 0:
            info(f"  波段置换 {gi + 1}/{len(pairs)}")

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("delta_mean", ascending=False).reset_index(drop=True)
    total = df["delta_mean"].clip(lower=0).sum()
    df["share"] = df["delta_mean"].clip(lower=0) / total if total > 0 else 0.0
    df["base_score"] = base_score
    df["rank"] = np.arange(1, len(df) + 1)
    df.attrs["mode"] = "all_rows" if all_rows else "test_fold"
    df.attrs["min_test_rows"] = int(min_test)
    # 全零意味着「置换没有产生任何信息」，不是「没有重要特征」：
    # 必须在 json 里标出来，报告不能照抄 top_bands。
    if float(np.nanmax(np.abs(df["delta_mean"].to_numpy(dtype=float)))) < 1e-9:
        df.attrs["degenerate"] = True
        df.attrs["note"] = "置换前后指标无变化（ΔR² 全为 0），本次重要性不可用"
    return df


def vip_scores(X: np.ndarray, y: np.ndarray, n_components: int, scale: bool = False) -> np.ndarray:
    """PLS 的 VIP 分数（Variable Importance in Projection）。"""
    from sklearn.cross_decomposition import PLSRegression

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n = X.shape[0]
    k = int(min(max(2, n_components), n - 1, X.shape[1]))
    pls = PLSRegression(n_components=k, scale=scale).fit(X, y)
    t = pls.x_scores_          # (n, A)
    w = pls.x_weights_         # (p, A)
    q = pls.y_loadings_        # (1, A)
    p = X.shape[1]
    ss = np.sum(t ** 2, axis=0) * np.sum(q ** 2, axis=0)   # (A,)
    if np.sum(ss) <= 0:
        return np.zeros(p)
    wn = w / np.maximum(np.linalg.norm(w, axis=0, keepdims=True), 1e-12)
    vip = np.sqrt(p * ((wn ** 2) @ ss) / np.sum(ss))
    return np.asarray(vip, dtype=float)


def coef_stability(
    fold_models: list[dict[str, Any]],
    feature_names: list[str],
) -> pd.DataFrame:
    """跨折系数符号一致率：回答"选出的波段是否稳定"。"""
    mats = []
    for fm in fold_models:
        est = fm["model"].named_steps.get("model")
        coef = getattr(est, "coef_", None)
        if coef is None:
            continue
        coef = np.asarray(coef).ravel()
        # 经过预处理的模型特征数可能已变（波段裁剪），长度不一致则跳过
        if coef.size != len(feature_names):
            continue
        mats.append(coef)
    if not mats:
        return pd.DataFrame()
    M = np.vstack(mats)
    mean_abs = np.abs(M).mean(axis=0)
    pos_frac = (M > 0).mean(axis=0)
    sign_consistency = np.maximum(pos_frac, 1 - pos_frac)
    return pd.DataFrame({
        "feature": feature_names,
        "coef_mean": M.mean(axis=0),
        "coef_sd": M.std(axis=0, ddof=1) if M.shape[0] > 1 else np.zeros(M.shape[1]),
        "abs_coef_mean": mean_abs,
        "sign_consistency": sign_consistency,
        "n_folds": M.shape[0],
    }).sort_values("abs_coef_mean", ascending=False).reset_index(drop=True)


def shap_values_safe(
    pipe,
    X: np.ndarray,
    max_rows: int = 200,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """在预处理后的空间上算 SHAP；失败返回 None（不中断主流程）。"""
    try:
        import shap
    except ImportError:
        return None
    est = pipe.named_steps.get("model")
    if est is None:
        return None
    try:
        pre = pipe[:-1]
        Xt = pre.transform(X)
        Xt = np.asarray(Xt, dtype=np.float64)
        if Xt.shape[0] > max_rows:
            idx = np.random.default_rng(seed).choice(Xt.shape[0], size=max_rows, replace=False)
            Xt = Xt[np.sort(idx)]
        explainer = shap.TreeExplainer(est)
        vals = np.asarray(explainer.shap_values(Xt))
        if vals.ndim == 3:
            vals = vals[..., -1]
        return vals, Xt
    except Exception as exc:  # noqa: BLE001
        warn(f"SHAP 计算失败，已跳过（不影响主结果）：{type(exc).__name__}: {exc}")
        return None


def top_features(df: pd.DataFrame, k: int = 20, by: str = "delta_mean") -> pd.DataFrame:
    if df.empty or by not in df:
        return df
    return df.sort_values(by, ascending=False).head(k).reset_index(drop=True)
