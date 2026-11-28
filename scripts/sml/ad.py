# Copyright 2026 The research-ml-modeling Authors
# SPDX-License-Identifier: Apache-2.0
"""适用域（Applicability Domain, AD）：判断一个新样本是否在模型可信范围内。

四种方法并行，各有盲区
----------------------
* **Hotelling T²**：PCA 得分空间的马氏距离，捕捉"在已有变化方向上走得太远"。
* **Q 残差（SPE）**：PCA 重构误差，捕捉"出现了训练集里没有的新变化方向"。
  T² 与 Q 是一对：只做 T² 会漏掉全新的谱形，只做 Q 会漏掉极端值。
* **kNN 距离**：到第 k 近邻的距离。**必须在 PCA 得分空间算**——在 2000 维原始谱上算
  欧氏距离会遇到"距离集中"现象（所有点彼此几乎等距），阈值失去意义。
* **Leverage（Williams plot）**：h_i 与学生化残差的组合图，化学计量学审稿人最熟悉的图。

必须验证 AD 的有效性
--------------------
"我们做了适用域"不是结论，**"域外样本的误差确实显著更大"才是**。若域内外性能无差异，
必须如实写进 limitations，而不是宣称模型可外推。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .utils import warn

__all__ = ["ADModel", "AD_SCHEMA_VERSION", "ad_model_version", "fit_ad", "evaluate_ad",
           "METHOD_LABELS"]

#: ``ad_model.joblib`` 的结构版本。2 = 带训练得分矩阵（杠杆的参考集）。
#: 旧的 1 会让 ``score()`` 把待判样品自己当参考集，判域结果不可信；
#: ``predict.py`` 读到 <2 的模型会直接报 E-PRED-006，而不是静默降级。
AD_SCHEMA_VERSION = 2


def ad_model_version(model: Any) -> int:
    """读 ``ad_model`` 的结构版本；0 表示没有版本字段的旧 pickle。

    1.2.4 生成的模型有 ``T_train_`` 但还没写版本号（该字段 1.2.5 才加），
    它已经满足 v2 的**实际要求**（有训练参考集），按 v2 处理；
    只有真正缺参考集的模型（≤1.2.3）才会被判成低于 v2。
    """
    v = int(getattr(model, "schema_version_", 0) or 0)
    if v == 0 and getattr(model, "T_train_", None) is not None:
        return AD_SCHEMA_VERSION
    return v

METHOD_LABELS = {
    "pca_t2": "Hotelling T²",
    "pca_q": "Q 残差(SPE)",
    "knn": "kNN 距离(PCA空间)",
    "leverage": "Leverage(h)",
}


def _leverage(T: np.ndarray, T_ref: np.ndarray | None = None) -> np.ndarray:
    """帽子矩阵对角线：``h_i = t_iᵀ(T_refᵀT_ref)⁻¹t_i + 1/n_ref``。

    ``+1/n`` 是截距项对应的那一份，不能省：Williams 的经验警戒线 ``3(k+1)/n`` 正是按
    「含截距、平均杠杆 (k+1)/n」推出来的。``fit`` 与 ``score`` 必须调同一个函数——
    曾经 ``fit`` 带 ``+1/n`` 而 ``score`` 不带，等于拿「带截距」的分布估计阈值，
    再去卡「不带截距」的分数，只在经验分位主导时才会露馅，且方向是系统性偏松。

    ``T_ref`` 是**参考集**（训练集的得分矩阵），``T`` 是被打分的点。二者必须分开：
    旧实现 ``score()`` 把待判样品自己的 ``T`` 同时当参考集，``(TᵀT)⁻¹`` 于是变成
    「新样品自己张成的空间」，``n`` 也变成新表行数。m 行批量时 ``Σh ≈ min(k,m) + 1``，
    单行打分恒得 ``h = 2.0``（远高于阈值 0.1–0.3），**所有新样品都被判域外**。
    这个错误在报告里查不出来——``stage_ad`` 对全量自打分时 ``T_new ≡ T_train``，
    两条路径天然一致。
    """
    T = np.asarray(T, dtype=np.float64)
    if T.size == 0:
        return np.zeros(0)
    ref = T if T_ref is None else np.asarray(T_ref, dtype=np.float64)
    if ref.size == 0 or ref.shape[1] != T.shape[1]:
        raise ValueError(
            f"杠杆参考集维度不匹配：T 有 {T.shape[1]} 列，参考集有 "
            f"{ref.shape[1] if ref.size else 0} 列"
        )
    h = np.einsum("ij,jk,ik->i", T, np.linalg.pinv(ref.T @ ref), T)
    return h + 1.0 / ref.shape[0]


@dataclass
class ADModel:
    methods: list[str] = field(default_factory=lambda: ["pca_t2", "pca_q", "knn", "leverage"])
    alpha: float = 0.05
    quantile: float = 0.95
    pca_variance: float = 0.95
    pca_max_comp: int = 10
    knn_k: int = 5

    # 训练后填充
    scaler: Any = None
    pca: Any = None
    nn: Any = None
    n_comp_: int = 0
    thr_: dict[str, float] = field(default_factory=dict)
    n_train_: int = 0
    p_train_: int = 0
    #: 结构版本（见 AD_SCHEMA_VERSION）。旧 joblib 反序列化后没有这个属性，
    #: 用 getattr(...,1) 读会把它们一律当成 v1 并拒绝打分。
    schema_version_: int = 0
    # 退化方法（阈值在本次数据上没有判别力）：方法名 -> 原因。
    # 退化方法不参与投票，也不计入「任一法越界」的并集——否则一个恒不越界的
    # leverage 和一个阈值恒为 0 的 Q 残差会把 AD 结论悄悄带偏。
    degenerate_: dict[str, str] = field(default_factory=dict)
    active_: list[str] = field(default_factory=list)
    #: 杠杆阈值的口径自述：h* 用行数还是样品数、本数据上杠杆法是否会触发。
    diag_: dict[str, Any] = field(default_factory=dict)

    # ---------------- 拟合 ----------------
    def fit(self, X: np.ndarray, groups: np.ndarray | None = None) -> "ADModel":
        X = np.asarray(X, dtype=np.float64)
        n, p = X.shape
        self.n_train_, self.p_train_ = n, p
        self.scaler = StandardScaler(with_mean=True).fit(X)
        Z = self.scaler.transform(X)

        max_comp = int(min(self.pca_max_comp, n - 1, p))
        cand = PCA(n_components=min(max_comp, p), random_state=0).fit(Z)
        cum = np.cumsum(cand.explained_variance_ratio_)
        k = int(np.searchsorted(cum, self.pca_variance) + 1)
        k = max(1, min(k, max_comp))
        self.pca = PCA(n_components=k, random_state=0).fit(Z)
        self.n_comp_ = k
        T = self.pca.transform(Z)
        Zhat = self.pca.inverse_transform(T)
        q = np.sum((Z - Zhat) ** 2, axis=1)
        t2 = np.sum((T / np.sqrt(np.maximum(self.pca.explained_variance_, 1e-12))) ** 2, axis=1)

        # kNN 在 PCA 得分空间（避免高维距离集中）
        kk = int(min(max(self.knn_k, 1), max(1, n - 1)))
        self.nn = NearestNeighbors(n_neighbors=min(kk + 1, n)).fit(T)
        dist, _ = self.nn.kneighbors(T)
        knn_dist = dist[:, -1] if dist.shape[1] > 1 else dist[:, 0]
        self.self_knn_ = knn_dist

        lev = _leverage(T)
        self.self_lev_ = lev
        # 判域时的杠杆必须相对「训练集张成的空间」算，不能相对待判样品自己。
        # 这里把训练得分矩阵（PCA 空间）存下来，score() 用它作参考集。
        self.T_train_ = T
        self.schema_version_ = AD_SCHEMA_VERSION

        self.thr_ = {
            "pca_t2": float(np.quantile(t2, self.quantile)),
            "pca_q": float(np.quantile(q, self.quantile)),
            "knn": float(np.quantile(knn_dist, self.quantile)),
            "leverage": float(max(3.0 * (k + 1) / n, np.quantile(lev, self.quantile))),
        }
        # 经验分位与理论阈值都存一份，报告里说明用了哪个
        self.thr_f_ = float(
            (k * (n - 1) * (n + 1)) / (n * (n - k)) * stats.f.ppf(1 - self.alpha, k, n - k)
        ) if n > k + 1 else float("inf")

        # ---- 退化判定：阈值在本数据上不可能被越过时，该法必须退出投票 ----
        # 否则「任一法越界即域外」的并集规则会被一个恒不越界的方法稀释，
        # 或被一个阈值≈0 的方法（浮点噪声即可越界）放大。
        deg: dict[str, str] = {}
        if n <= k + 1:
            deg["pca_t2"] = f"n={n} ≤ 主成分数+1={k + 1}，T² 的 F 理论阈值不可算"
        if k >= p:
            deg["pca_q"] = (f"PCA 保留全部 {p} 个主成分（解释方差=100%），"
                            "Q 残差恒为 0、阈值退化为数值零")
        else:
            q_scale = float(np.mean(Z ** 2)) or 1.0
            if float(np.quantile(q, self.quantile)) <= 1e-10 * q_scale:
                deg["pca_q"] = "Q 残差经验分位≈0（低于浮点相对精度），阈值无判别力"
        if self.thr_["leverage"] >= 1.0:
            deg["leverage"] = (f"杠杆阈值 h*=3(k+1)/n={self.thr_['leverage']:.3f} ≥ 1，"
                               "而杠杆值恒 ≤1，该法永远不越界")
        self.degenerate_ = deg
        self.active_ = [m for m in self.methods if m not in deg]
        if not self.active_:            # 全退化时保留原方法集合，避免 votes 恒为 0
            self.active_ = [m for m in self.methods]
        for m, why in deg.items():
            warn(f"AD 方法 {METHOD_LABELS.get(m, m)} 退化，已退出投票：{why}")

        # 「h* 该用哪个 n」必须在数据里讲清楚。逐行打分用的是拟合行数（每个谱图一个点，
        # 帽子矩阵建立在 n_rows 上），而「独立样品够不够撑起经验阈值」用的是样品数。
        # 两个数不是同一个量；同一个 run 里同时出现又不加区分，读者只会认为其中一个算错了。
        n_groups_fit = len(set(map(str, groups))) if groups is not None else n
        self.diag_ = {
            "n_rows": int(n),
            "n_groups": int(n_groups_fit),
            "n_components": int(k),
            "h_star_rows": float(3.0 * (k + 1) / max(1, n)),
            "h_star_groups": float(3.0 * (k + 1) / max(1, n_groups_fit)),
            "leverage_threshold": float(self.thr_["leverage"]),
            # 阈值取的是 max(理论界限, 经验分位)，两分支必须都留档：
            # 只给一个 0.15，读者无法判断它是 3(k+1)/n 还是 95% 分位算出来的。
            "leverage_q95": float(np.quantile(lev, self.quantile)),
            "leverage_rule": "h* = max(3(k+1)/n, 训练集经验分位)",
            "leverage_reference": "train_score_matrix",
            "leverage_max": float(np.max(lev)),
            # 阈值高于数据里最大的杠杆值 ⇒ 这一法本次一次也不会触发。它不属于"退化"
            # （h* < 1，公式本身没错），但会把 ad_majority 的分母（参与投票的方法数）抬高，
            # 让"过半"更难达到，所以必须在产物里说清楚，而不是让读者对着 0 越界自己猜。
            "leverage_inert": bool(float(np.max(lev)) <= float(self.thr_["leverage"])),
            # 本次是否真的把 leverage 放进投票集合：smoke 预设只跑 T²+kNN，
            # 这时 ad_scores.csv 里仍有 leverage 列与阈值，但它不参与判域，
            # 图与报告必须说清楚，否则读者会以为那条红线决定了域外样品。
            "leverage_active": bool("leverage" in self.active_),
        }
        return self

    # ---------------- 打分 ----------------
    def score(self, X: np.ndarray) -> pd.DataFrame:
        if self.pca is None:
            raise RuntimeError("ADModel 未 fit")
        X = np.asarray(X, dtype=np.float64)
        Z = self.scaler.transform(X)
        T = self.pca.transform(Z)
        Zhat = self.pca.inverse_transform(T)
        q = np.sum((Z - Zhat) ** 2, axis=1)
        t2 = np.sum((T / np.sqrt(np.maximum(self.pca.explained_variance_, 1e-12))) ** 2, axis=1)
        # kNN 距离的口径必须与 fit 一致：阈值是「到第 k 个**他人**的距离」的分位数。
        # 多取一列用于分辨「被评分的这一行本身是否就在参考集里」——自打分时最近邻
        # 是自己（距离 0），直接取 dist[:,-1] 会得到第 k-1 个他人，判据系统性偏松；
        # 新样品不在参考集里时最近邻已经是他人，应取第 k 个。
        k = int(min(max(self.knn_k, 1), self.n_train_ - 1)) if self.n_train_ > 1 else 1
        m = int(min(k + 1, self.nn.n_samples_fit_))
        dist, _ = self.nn.kneighbors(T, n_neighbors=m)
        if dist.shape[1] > k:
            self_hit = dist[:, 0] <= 1e-12
            knn_dist = np.where(self_hit, dist[:, -1], dist[:, k - 1])
        else:
            knn_dist = dist[:, -1]
        # 参考集是训练得分矩阵；旧实现误用待判样品自身（见 _leverage docstring）。
        # 旧的 joblib 里没有 T_train_：**必须显式失败**，不能退回自参考口径——
        # 退回后 ad_flag 会近乎恒为 1，而 predictions.csv 看起来完全正常。
        ref = getattr(self, "T_train_", None)
        if ref is None or ad_model_version(self) < AD_SCHEMA_VERSION:
            raise RuntimeError(
                "ad_model.joblib 由 ≤1.2.3 的 skill 生成（缺少训练得分矩阵 T_train_），"
                "杠杆判域只能退化为「待判样品自己当参考集」，会给出看似正常但判域全错的预测。"
                f"请对该 run 重跑 AD 阶段重建模型：--stages ad,report（期望 schema={AD_SCHEMA_VERSION}）")
        lev = _leverage(T, ref)

        flags = {
            "pca_t2": t2 > self.thr_["pca_t2"],
            "pca_q": q > self.thr_["pca_q"],
            "knn": knn_dist > self.thr_["knn"],
            "leverage": lev > self.thr_["leverage"],
        }
        out = pd.DataFrame({
            "t2": t2, "t2_thr": self.thr_["pca_t2"],
            "q": q, "q_thr": self.thr_["pca_q"],
            "knn_dist": knn_dist, "knn_thr": self.thr_["knn"],
            "leverage": lev, "leverage_thr": self.thr_["leverage"],
        })
        votes = np.zeros(X.shape[0], dtype=int)
        active = [m for m in self.methods if m in flags and m not in self.degenerate_]
        if not active:
            active = [m for m in self.methods if m in flags]
        for m, fl in flags.items():
            if m in self.methods:
                out[f"out_{m}"] = fl.astype(int)
                if m in active:
                    votes += fl.astype(int)
        out["ad_votes"] = votes
        n_active = max(1, len(active))
        out["ad_flag"] = (votes > 0).astype(int)          # 保守：任一越界即域外
        out["ad_majority"] = (votes > n_active / 2).astype(int)
        return out

    def describe(self) -> dict[str, Any]:
        return {
            "methods": self.methods,
            "active_methods": list(self.active_),
            "degenerate_methods": dict(self.degenerate_),
            "n_components": int(self.n_comp_),
            "explained_variance": float(np.sum(self.pca.explained_variance_ratio_)) if self.pca is not None else None,
            "thresholds": self.thr_,
            "quantile": self.quantile,
            "n_train": int(self.n_train_),
            "diagnostics": dict(self.diag_),
            "note": ("阈值取训练集经验分位（小样本下 F/χ² 近似偏乐观），另存 F 理论值 thr_f_=%s；"
                     "退化方法（阈值不可能被越过）已退出投票，见 degenerate_methods") % (
                f"{self.thr_f_:.3f}" if hasattr(self, "thr_f_") else "n/a"),
        }


def fit_ad(
    X: np.ndarray,
    cfg: dict,
    groups: np.ndarray | None = None,
) -> ADModel:
    ad_cfg = cfg.get("ad", {})
    model = ADModel(
        methods=list(ad_cfg.get("methods") or ["pca_t2", "pca_q", "knn", "leverage"]),
        alpha=float(ad_cfg.get("alpha", 0.05)),
        quantile=float(ad_cfg.get("quantile", 0.95)),
        pca_variance=float(ad_cfg.get("pca_variance", 0.95)),
        pca_max_comp=int(ad_cfg.get("pca_max_comp", 10)),
        knn_k=int(ad_cfg.get("knn_k", 5)),
    )
    return model.fit(X, groups)


def evaluate_ad_with_X(
    ad: ADModel,
    X: np.ndarray,
    oof: pd.DataFrame,
    row_ids: np.ndarray,
    groups: np.ndarray,
    task: str,
    metric: str = "rmse",
    n_boot: int = 500,
    seed: int = 0,
) -> dict[str, Any]:
    """把 AD 打分对齐到 OOF 行（按 row_id 一一对应），再比较域内/域外误差。"""
    from .metrics import regression_metrics

    sc = ad.score(X)
    sc["row_id"] = row_ids
    merged = oof.merge(sc, on="row_id", how="left", suffixes=("", "_ad"))
    if merged["ad_flag"].isna().all():
        return {"error": "AD 打分与 OOF 行无法对齐"}

    inside = merged[merged["ad_flag"] == 0]
    outside = merged[merged["ad_flag"] == 1]
    res: dict[str, Any] = {
        "n_inside": int(len(inside)),
        "n_outside": int(len(outside)),
        "flag_rate": float(merged["ad_flag"].mean()),
        "n_oof_rows": int(len(merged)),
        "thresholds": ad.thr_,
        "n_components": ad.n_comp_,
    }
    # 行口径 ≠ 论文口径。OOF 里同一样品会出现「重复测量数 × 重复次数」次
    # （本项目 60 样品 × 3 次 × 2 repeat = 360 行），行数会把域外样品数放大 3–6 倍。
    # 论文里只能报独立样品数，所以这里同时给出 group 口径，避免下游各写一个数字。
    if "group" in merged:
        g = merged["group"].astype(str)
        all_groups = set(g)
        flagged = set(g[merged["ad_flag"].to_numpy() == 1])
        maj = set(g[merged["ad_majority"].to_numpy() == 1]) if "ad_majority" in merged else set()
        res.update({
            "n_groups": int(len(all_groups)),
            "n_outside_groups": int(len(flagged)),
            "n_inside_groups": int(len(all_groups - flagged)),
            "flag_rate_groups": float(len(flagged) / max(1, len(all_groups))),
            "n_outside_groups_majority": int(len(maj)),
            "flag_rate_groups_majority": float(len(maj) / max(1, len(all_groups))),
            "group_rule": "样品内任一测量被判越界，即该样品计为域外（保守口径）",
            "rows_per_group": float(len(merged) / max(1, len(all_groups))),
        })
    if task == "regression":
        m_in = regression_metrics(inside["y_true"], inside["y_pred"]) if len(inside) > 2 else {}
        m_out = regression_metrics(outside["y_true"], outside["y_pred"]) if len(outside) > 2 else {}
        res["metrics_inside"] = {k: float(v) for k, v in m_in.items()}
        res["metrics_outside"] = {k: float(v) for k, v in m_out.items()}
        # ---- 有效性检验的单位必须是**独立样品**，不是 OOF 行 ----
        # 60 样品 × 3 次重复 × 2 repeat = 360 行；同一样品的残差高度相关，
        # 直接对行做 Mann-Whitney 会同时踩两个坑：
        #   1) 独立性假设被违反 → p 值偏小，"有效"偏容易成立；
        #   2) 「样本量 > 5」的门槛等于 6 行，也就是**1 个样品**——一个样品就能"证明"AD 有效。
        # 这里先按样品求平均绝对误差，再在样品层面做秩检验。行口径另存 `_rows` 字段，
        # 只作敏感性分析用（与 SKILL.md「不能把 OOF 行数当样品数」一致）。
        r_in_rows = np.abs(inside["y_true"] - inside["y_pred"]).to_numpy(dtype=float)
        r_out_rows = np.abs(outside["y_true"] - outside["y_pred"]).to_numpy(dtype=float)
        if "group" in merged:
            agg = (pd.DataFrame({
                "group": merged["group"].astype(str),
                "_ae": np.abs(merged["y_true"] - merged["y_pred"]).to_numpy(dtype=float),
                "_flag": merged["ad_flag"].to_numpy(dtype=int),
            }).groupby("group", as_index=False)
                .agg(_ae=("_ae", "mean"), _flag=("_flag", "max")))
            r_in = agg.loc[agg["_flag"] == 0, "_ae"].to_numpy(dtype=float)
            r_out = agg.loc[agg["_flag"] == 1, "_ae"].to_numpy(dtype=float)
            res["test_unit"] = "group"
            res["test_unit_note"] = "按样品平均绝对误差检验（同一样品的重复测量视为一个观测）"
        else:
            r_in, r_out = r_in_rows, r_out_rows
            res["test_unit"] = "row"
            res["test_unit_note"] = "数据无 group 列，退回行口径（重复测量未聚合，p 值偏乐观）"
        res["n_units_inside"] = int(r_in.size)
        res["n_units_outside"] = int(r_out.size)
        res["row_mannwhitney_p_greater"] = None
        if r_in.size >= 5 and r_out.size >= 5:
            # 方向：AD 要在「域外误差更大」这件事上被证明。
            # 旧写法 mannwhitneyu(r_in, r_out, alternative="greater") 的备择假设是
            # 「域内误差 > 域外误差」，与要证的方向相反：域外确实更差时 p→1，于是恒判
            # no_difference，effective 分支不可达（reversed 分支也永远拿不到 p<0.05）。
            #
            # 判定方向还必须与检验本身同口径：Mann-Whitney 比的是秩（≈中位数），
            # 旧代码却拿**均值**判有效/反向。均值会被少数极端残差拉动，于是出现
            # 「p 值显示域外更差，但域外均值反而更小」这种自相矛盾的判定（在真实数据上
            # 实测出现过：均值与中位数的方向相反，差值足以翻转结论）。
            # 现在：秩方向决定 effective / reversed；秩与均值方向相反时判 conflict，
            # 不硬给一个"有效"或"无差异"的结论。
            med_in, med_out = float(np.median(r_in)), float(np.median(r_out))
            mean_in, mean_out = float(r_in.mean()), float(r_out.mean())
            p_greater = float(stats.mannwhitneyu(r_out, r_in, alternative="greater").pvalue)
            p_less = float(stats.mannwhitneyu(r_out, r_in, alternative="less").pvalue)
            p_two = float(stats.mannwhitneyu(r_out, r_in, alternative="two-sided").pvalue)
            rank_worse = bool(med_out > med_in)
            mean_worse = bool(mean_out > mean_in)
            res["mannwhitney_p_greater"] = p_greater
            res["mannwhitney_p_less"] = p_less
            res["mannwhitney_p_two"] = p_two
            res["mean_abs_error_inside"] = mean_in
            res["mean_abs_error_outside"] = mean_out
            res["median_abs_error_inside"] = med_in
            res["median_abs_error_outside"] = med_out
            res["effect_direction"] = "outside_worse" if rank_worse else "outside_better"
            res["mean_rank_direction_conflict"] = bool(rank_worse != mean_worse)
            if res["test_unit"] == "group" and r_in_rows.size > 5 and r_out_rows.size > 5:
                res["row_mannwhitney_p_greater"] = float(
                    stats.mannwhitneyu(r_out_rows, r_in_rows, alternative="greater").pvalue)
            conflict = res["mean_rank_direction_conflict"] and (p_greater < 0.05 or p_less < 0.05)
            if conflict:
                res["ad_verdict"] = "conflict"
                res["mannwhitney_p"] = min(p_greater, p_less)
            elif p_greater < 0.05:
                res["ad_verdict"] = "effective"
                res["mannwhitney_p"] = p_greater
            elif p_less < 0.05:
                res["ad_verdict"] = "reversed"
                res["mannwhitney_p"] = p_less
            else:
                res["ad_verdict"] = "no_difference"
                res["mannwhitney_p"] = p_two
            # conflict 是"两种口径打架"，不是"已证明无效"：给 None，让下游写「无法判定」
            # 而不是把它当成第二个 no_difference 混进"AD 无效"的统计里。
            res["ad_effective"] = (None if res["ad_verdict"] == "conflict"
                                   else res["ad_verdict"] == "effective")
            if metric in ("rmse", "mae"):
                unit_txt = ("样品级平均绝对误差" if res.get("test_unit") == "group"
                            else "行级绝对误差")
                res["note"] = (
                    f"域外{unit_txt} 均值 {mean_out:.4f} / 中位数 {med_out:.4f}，"
                    f"域内 均值 {mean_in:.4f} / 中位数 {med_in:.4f}"
                    f"（Mann-Whitney 秩检验 单向 p={min(p_greater, p_less):.4g}，双向 p={p_two:.4g}"
                    f"；n={int(r_in.size)} vs {int(r_out.size)}）"
                    + ("；**秩方向与均值方向相反**：少数极端残差翻转了均值，结论存疑"
                       if res["mean_rank_direction_conflict"] else "")
                )
        else:
            res["ad_effective"] = None
            res["note"] = (f"域内/域外独立单位过少（{int(r_in.size)} vs {int(r_out.size)}，"
                           f"各需 ≥5），无法检验 AD 有效性")
    else:
        # 分类：用 2×2 列联表做 Fisher 精确检验（域外是否正确率更低）。
        # 只比较两个准确率就说"无显著差异"是不严谨的——那只是点估计。
        # 方向同样不能写反：旧实现用 alternative="less"（域内正确率更低），
        # 备择假设与要证的方向相反，域外确实更差时 p→1，只会恒判 no_difference。
        acc_in = float((inside["y_pred"].round() == inside["y_true"]).mean()) if len(inside) else float("nan")
        acc_out = float((outside["y_pred"].round() == outside["y_true"]).mean()) if len(outside) else float("nan")
        res["acc_inside"] = acc_in
        res["acc_outside"] = acc_out
        res["note"] = "分类任务的 AD 有效性以误分类率比较（Fisher 精确检验，双向为主判据）"
        res["test_unit"] = "row"
        res["test_unit_note"] = ("分类用行级 2×2 列联表；同一标签下重复测量使 p 值偏乐观，"
                                 "只作参考，不单独作为 AD 有效的证据")
        try:
            c_in = int((inside["y_pred"].round() == inside["y_true"]).sum())
            w_in = int(len(inside) - c_in)
            c_out = int((outside["y_pred"].round() == outside["y_true"]).sum())
            w_out = int(len(outside) - c_out)
            if min(len(inside), len(outside)) > 0 and (c_in + w_in) and (c_out + w_out):
                tbl = [[c_in, w_in], [c_out, w_out]]
                p_greater = float(stats.fisher_exact(tbl, alternative="greater").pvalue)
                p_less = float(stats.fisher_exact(tbl, alternative="less").pvalue)
                p_two = float(stats.fisher_exact(tbl, alternative="two-sided").pvalue)
                outside_worse = bool(np.isfinite(acc_in) and acc_out < acc_in)
                p_dir = p_greater if outside_worse else p_less
                res["fisher_p"] = p_dir
                res["fisher_p_greater"] = p_greater
                res["fisher_p_less"] = p_less
                res["fisher_p_two"] = p_two
                res["effect_direction"] = ("outside_worse" if outside_worse
                                           else "outside_better_or_equal")
                if p_dir < 0.05 and outside_worse:
                    res["ad_verdict"] = "effective"
                elif p_dir < 0.05:
                    res["ad_verdict"] = "reversed"
                else:
                    res["ad_verdict"] = "no_difference"
                res["ad_effective"] = res["ad_verdict"] == "effective"
            else:
                res["ad_effective"] = None
        except Exception:  # noqa: BLE001
            res["ad_effective"] = bool(np.isfinite(acc_in) and np.isfinite(acc_out) and acc_out < acc_in)

    res["conclusion"], res["author_note"] = _ad_conclusions(res, task)
    return res


def _ad_conclusions(res: dict[str, Any], task: str = "regression") -> tuple[str, str]:
    """返回 ``(conclusion, author_note)``：前者可直接进论文，后者只给作者看。

    这两类文字必须分开。旧版本把"应在 limitations 中如实说明"这类**写作指令**
    写进 conclusion，再被 Methods 与报告正文原样引用——审稿人会在正文里读到
    一条写给作者的待办事项。凡是"不得宣称/必须写入/不要……"一律归 author_note。
    """
    eff = res.get("ad_effective")
    verdict = res.get("ad_verdict") or (None if eff is None else ("effective" if eff else "no_difference"))
    clf = task == "classification"
    what = "误分类率" if clf else "预测误差"
    if eff is None and verdict != "conflict":
        return (f"域内或域外样品数量不足，未对适用域判据的有效性作统计检验。",
                "样本量不足以验证 AD：正文只描述 AD 的定义与阈值，不得出现「可外推」表述。")
    if verdict == "conflict":
        p = res.get("mannwhitney_p")
        p_txt = "—" if p is None else f"{float(p):.4g}"
        mi, mo = res.get("mean_abs_error_inside"), res.get("mean_abs_error_outside")
        di, do = res.get("median_abs_error_inside"), res.get("median_abs_error_outside")
        nums = ("" if None in (mi, mo, di, do) else
                f"（均值 域外 {mo:.4g} vs 域内 {mi:.4g}；中位数 域外 {do:.4g} vs 域内 {di:.4g}）")
        return (f"适用域有效性检验出现口径冲突（秩检验 p={p_txt}）{nums}：按中位数域外误差更高，"
                f"按均值域外反而更低。域外样品较少时，少数极端残差就足以翻转均值，"
                f"因此不能据此宣称该判据能筛出低可靠预测。",
                "AD 有效性存疑：正文不得写「AD 可识别不可靠预测」「域外预测不可靠」；"
                "必须写入 limitations 并说明口径冲突；要定论需扩大域外样本，"
                "或改用中位残差等稳健统计重做检验。")
    if verdict == "reversed":
        p = res.get("fisher_p") if clf else res.get("mannwhitney_p")
        p_txt = "—" if p is None else f"{float(p):.4g}"
        return (f"域外样品的{what}反而低于域内样品（p={p_txt}）：判据标出的「域外」样品"
                f"并不比域内样品更不可靠，适用域的方向与预期相反，不能用于筛选低可靠性预测。",
                "AD 方向异常：必须写入 limitations；不要用 AD 做样品筛选或外推判定；"
                "先查是否有样品标签/分组错配，或阈值取法（经验分位 vs 理论分布）是否合适。")
    if eff:
        return (f"域外样品的{what}显著高于域内样品，表明在当前特征空间下定义的适用域"
                f"能够识别出可靠性较低的预测；对判定为域外的样品，其预测结果应谨慎解释。",
                "AD 有效。可明确宣称「域外样品预测不可靠」，并给出域内外性能对比；"
                "注意不要剔除域外样品后重新计算性能。")
    return (f"域内与域外样品的{what}无显著差异，当前适用域判据未能区分出可靠性较低的预测。",
            "不得宣称「模型可安全外推」。必须写入 limitations：AD 判据在本数据上未显示判别力；"
            "也不要用「剔除域外样本后性能更优」来论证外推能力。")


def mixed_author_note(effective: list[str] | None = None, n_models: int | None = None) -> str:
    """AD 结论按模型分裂时的作者提示。

    1.2.8：旧实现只覆写了顶层 ``conclusion``，``author_note`` 仍原样保留第一个模型的
    「AD 有效。可明确宣称……」，于是 report.md 的「写作提示」与同一节的「不一致」
    结论自相矛盾（正是 1.2.5「Methods 取第一个模型」漏掉的另一半）。
    两者必须由此函数统一生成。
    """
    names = "、".join(str(m) for m in (effective or [])) or "部分模型"
    of = f"（{n_models} 个模型）" if n_models else ""
    return (f"AD 结论按模型分裂{of}：只有 {names} 判定「有效」，可以写「域外预测不可靠」；"
            f"其余模型必须把「AD 判据未显示判别力」写入 limitations。"
            f"正文与 Methods 都要逐模型说明，禁止用一句话概称「AD 有效」或「AD 无效」，"
            f"也不要用「剔除域外样品后性能更好」来论证外推能力。")


def ad_summary_table(scores: pd.DataFrame,
                     degenerate: dict[str, str] | None = None) -> pd.DataFrame:
    """各 AD 方法的越界统计。主口径是**独立样品**，行口径另存一列备查。

    ``degenerate`` 里的方法阈值在本数据上不可能被越过（如 h*≥1 的 leverage、
    保留全部主成分时的 Q 残差），它们的「0 个越界」不是好消息而是失效，
    所以单列 status / note，不能与真正有效的 0 越界混在一张表里。
    """
    degenerate = dict(degenerate or {})
    spec = [("pca_t2", "t2", "t2_thr"), ("pca_q", "q", "q_thr"),
            ("knn", "knn_dist", "knn_thr"), ("leverage", "leverage", "leverage_thr")]
    groups = scores["group"].astype(str).to_numpy() if "group" in scores else None
    n_groups = len(set(groups)) if groups is not None else 0
    rows = []
    for key, col, thr_col in spec:
        flag_col = f"out_{key}"
        if flag_col not in scores:
            continue
        ring = scores[flag_col].to_numpy() == 1
        n_row = int(ring.sum())
        n_grp = len(set(groups[ring])) if groups is not None else n_row
        rows.append({
            "method": METHOD_LABELS.get(key, key),
            "key": key,
            "status": "退化，已退出投票" if key in degenerate else "有效",
            "note": degenerate.get(key, ""),
            "flagged": n_grp,
            "rate": float(n_grp / n_groups) if n_groups else float("nan"),
            "flagged_rows": n_row,
            "threshold": float(scores[thr_col].iloc[0]) if thr_col in scores and len(scores) else float("nan"),
        })
    return pd.DataFrame(rows)
