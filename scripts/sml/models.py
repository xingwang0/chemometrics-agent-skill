# Copyright 2026 The research-ml-modeling Authors
# SPDX-License-Identifier: Apache-2.0
"""模型工厂：估计器、默认搜索空间、Pipeline 组装。

关键约定
--------
* **PLS 是强制基线**：无论配置怎么写，化学计量学读者期待的 PLS 一定出现在结果里
  （配置只能改它的搜索空间，删不掉）。这不是"多一个模型"，而是审稿的第一道门槛。
* 所有估计器都通过 ``SafePLS`` / 静默参数处理过已知坑：
  n_components 越界、catboost 写盘、lightgbm 刷屏、xgboost 早停位置。
* 同一个 ``kind`` 按任务类型自动落到 Regressor / Classifier，配置里不用写两套。
"""

from __future__ import annotations

import copy
import os
import warnings
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge, RidgeClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from .preprocess import PrepSwitcher, prep_name
from .utils import warn

__all__ = [
    "ModelSpec", "resolve_models", "build_pipeline", "build_estimator",
    "search_space", "default_space", "validate_spaces", "SafePLS", "SafePLSDA",
    "SafePCR", "SafePCRDA", "SUPPORTED_KINDS", "CLASSIFICATION_ONLY",
    "is_tree", "uses_shap",
]

SUPPORTED_KINDS = {
    "pls", "pcr", "ridge", "lasso", "enet", "svr", "knn", "rf", "et", "gbr",
    "xgb", "lgbm", "cat", "gp", "logreg", "svc", "ridge_clf", "gnb", "lda",
}

# 只有分类语义的 kind：回归任务下必须**明确拒绝**，而不是拿分类器去拟合连续 y。
# 反例（本地实测）：GaussianNB 在连续标签上抛 Unknown label type，而 CV 层会把这个
# 异常记成 error:ValueError 然后整模型从结果表消失——报错信息离现场太远，等于没报错。
CLASSIFICATION_ONLY = {"logreg", "svc", "ridge_clf", "gnb", "lda"}

TREE_KINDS = {"rf", "et", "gbr", "xgb", "lgbm", "cat"}

# sklearn 1.9 起 SVC(probability=True) 被标记弃用（1.11 移除）。本机仍可用且行为稳定，
# 而换成 CalibratedClassifierCV 会在嵌套 CV 里再嵌一层交叉验证、把小折彻底搞坏。
# 这里只静音这条提示，并在 references/environment.md 记录升级时的替换方案。
warnings.filterwarnings("ignore", message=".*The `probability` parameter was deprecated.*")
# 本包已不再传 penalty（L1 用 l1_ratio=1.0）；这条只兜用户配置里残留的旧写法
warnings.filterwarnings("ignore", message=".*'penalty' was deprecated.*")
# GP 在极小样本/常值目标上会刷几十屏 ConvergenceWarning（"Prediction is constant" /
# "fmin_l_bfgs_b terminated abnormally"）。这类提示对结果没有额外信息，且会把真正的
# 进度与告警淹掉；只压 sklearn.gaussian_process 的收敛警告，不影响其它模型。
warnings.filterwarnings("ignore", message=r".*Prediction is constant.*")
warnings.filterwarnings("ignore", message=r".*fmin_l_bfgs_b terminated abnormally.*")
warnings.filterwarnings(
    "ignore", category=UserWarning, module=r"sklearn\.gaussian_process\..*",
)
# 1.2.8：enet/lasso 在共线光谱上会逐折刷 sklearn 的 ConvergenceWarning
# （单次运行实测数百条，把 [sml] 进度与真正的告警全冲掉）。这条警告对结果没有
# 额外信息——坐标下降未收敛的后果已经体现在该折的指标里，且 search 会把 alpha
# 往更正则的方向走。统一静音，但留一个开关给排查用。
if os.environ.get("SML_SHOW_CONVERGENCE", "") not in {"1", "true", "TRUE"}:
    try:
        from sklearn.exceptions import ConvergenceWarning as _ConvWarn
    except Exception:  # noqa: BLE001
        _ConvWarn = UserWarning  # type: ignore[assignment,misc]
    warnings.filterwarnings("ignore", category=_ConvWarn)


# --------------------------------------------------------------------------- #
# PLS 安全封装
# --------------------------------------------------------------------------- #
class SafePLS(BaseEstimator, RegressorMixin):
    """PLS 回归，fit 时把 n_components 裁剪到合法范围。

    直接 ``PLSRegression(n_components=15)`` 在训练折只有 8 个样品时会
    ``ValueError: n_components upper bound is 7``，而内层搜索不该因折小就整折报废。
    """

    def __init__(self, n_components: int = 5, scale: bool = False):
        self.n_components = n_components
        self.scale = scale

    def _clipped(self, X) -> int:
        n = int(np.asarray(X).shape[0])
        p = int(np.asarray(X).shape[1])
        upper = max(1, min(n - 1, p))
        k = int(min(int(self.n_components), upper))
        if k < int(self.n_components):
            warn(f"PLS n_components 由 {self.n_components} 裁剪为 {k}（训练折仅 {n} 行 × {p} 列）")
        return k

    def fit(self, X, y):
        self.model_ = PLSRegression(n_components=self._clipped(X), scale=self.scale)
        self.model_.fit(np.asarray(X), np.asarray(y, dtype=np.float64).ravel())
        self.n_features_in_ = int(np.asarray(X).shape[1])
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X)).ravel()

    @property
    def coef_(self):
        return getattr(self.model_, "coef_", None)

    @property
    def x_weights_(self):
        return getattr(self.model_, "x_weights_", None)

    @property
    def x_scores_(self):
        return getattr(self.model_, "x_scores_", None)


class SafePLSDA(BaseEstimator, ClassifierMixin):
    """PLS-DA：用 PLS 回归拟合 one-hot 响应，再取 argmax 分类。

    直接用回归的 PLS 做分类会得到可以超出 [0,1] 的连续输出（例如 -0.31），
    既不能当概率，也会让 Brier/log-loss 直接报错。这里显式转成概率接口并裁剪到 (0,1)。
    """

    def __init__(self, n_components: int = 5, scale: bool = False):
        self.n_components = n_components
        self.scale = scale

    def fit(self, X, y):
        from sklearn.preprocessing import OneHotEncoder

        X = np.asarray(X)
        y = np.asarray(y).astype(int).ravel()
        self.classes_ = np.unique(y)
        self._enc = OneHotEncoder(sparse_output=False, categories=[self.classes_])
        Y = self._enc.fit_transform(y.reshape(-1, 1))
        upper = max(1, min(int(X.shape[0]) - 1, int(X.shape[1])))
        k = int(min(int(self.n_components), upper))
        self.model_ = PLSRegression(n_components=k, scale=self.scale)
        self.model_.fit(X, Y)
        self.n_features_in_ = int(X.shape[1])
        return self

    def predict_proba(self, X):
        raw = np.asarray(self.model_.predict(np.asarray(X)), dtype=np.float64)
        if raw.ndim == 1:
            raw = raw.reshape(-1, 1)
        raw = np.clip(raw, 1e-6, 1 - 1e-6)
        if raw.shape[1] == 1:            # 二分类且只拟合了一列
            return np.column_stack([1 - raw[:, 0], raw[:, 0]])
        return raw / raw.sum(axis=1, keepdims=True)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    @property
    def coef_(self):
        return getattr(self.model_, "coef_", None)

    @property
    def x_weights_(self):
        return getattr(self.model_, "x_weights_", None)


# --------------------------------------------------------------------------- #
# PCR / PCR-DA 安全封装
# --------------------------------------------------------------------------- #
def _pca_upper(X) -> int:
    """PCA 的 n_components 合法上界：min(n_samples, n_features)。"""
    shape = np.asarray(X).shape
    return int(max(1, min(shape[0], shape[1])))


class SafePCR(BaseEstimator, RegressorMixin):
    """主成分回归，fit 时把 n_components 裁到 PCA 的合法范围。

    为什么不做成 ``Pipeline([("pca",…),("ols",…)])``：那样超参路径会变成
    ``model__pca__n_components``，写成 ``model__n_components`` 会让每一折都
    ``Invalid parameter`` 而整模型静默消失。封装成单估计器后路径固定为
    ``model__n_components``，且能在折内裁剪上界（训练折只有 8 行时 PCA(20) 必崩）。
    """

    def __init__(self, n_components: int = 10):
        self.n_components = n_components

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        self.n_features_in_ = int(X.shape[1])
        k = int(min(int(self.n_components), _pca_upper(X)))
        if k < int(self.n_components):
            warn(f"PCR n_components 由 {self.n_components} 裁剪为 {k}（训练折 {X.shape[0]} 行 × {X.shape[1]} 列）")
        self.pca_ = PCA(n_components=k).fit(X)
        self.model_ = LinearRegression().fit(self.pca_.transform(X),
                                             np.asarray(y, dtype=np.float64).ravel())
        return self

    def predict(self, X):
        return self.model_.predict(self.pca_.transform(np.asarray(X, dtype=np.float64)))

    @property
    def coef_(self):
        """把 PCA 载荷折回去，供系数/重要性分析使用（原空间坐标）。"""
        if not hasattr(self, "pca_"):
            return None
        return np.asarray(self.model_.coef_, dtype=np.float64) @ np.asarray(self.pca_.components_)


class SafePCRDA(BaseEstimator, ClassifierMixin):
    """PCR-DA：PCA 降维后接（多项）逻辑回归，提供标准概率接口。

    直接复用 ``LinearRegression`` 做分类会输出连续值（可越界），既非概率也会让
    log-loss 报错——和 PLS-DA 是同一个坑。
    """

    def __init__(self, n_components: int = 10, C: float = 1.0):
        self.n_components = n_components
        self.C = C

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).astype(int).ravel()
        self.n_features_in_ = int(X.shape[1])
        self.classes_ = np.unique(y)
        k = int(min(int(self.n_components), _pca_upper(X)))
        if k < int(self.n_components):
            warn(f"PCR-DA n_components 由 {self.n_components} 裁剪为 {k}（训练折 {X.shape[0]} 行 × {X.shape[1]} 列）")
        self.pca_ = PCA(n_components=k).fit(X)
        self.model_ = LogisticRegression(C=float(self.C), max_iter=5000,
                                        class_weight="balanced", solver="lbfgs")
        self.model_.fit(self.pca_.transform(X), y)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(self.pca_.transform(np.asarray(X, dtype=np.float64)))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    @property
    def coef_(self):
        if not hasattr(self, "pca_"):
            return None
        return np.atleast_2d(self.model_.coef_) @ np.asarray(self.pca_.components_)


# --------------------------------------------------------------------------- #
# 模型规格
# --------------------------------------------------------------------------- #
class ModelSpec:
    def __init__(self, raw: dict, task: str):
        self.raw = copy.deepcopy(raw)
        self.task = task
        self.name = str(raw.get("name") or raw.get("kind") or "model")
        self.kind = str(raw.get("kind") or "pls")
        self.search_type = str(raw.get("search_type") or "grid").lower()
        self.search: dict[str, Any] = copy.deepcopy(raw.get("search") or {})
        self.fixed: dict[str, Any] = copy.deepcopy(raw.get("fixed") or {})
        self.n_iter = int(raw.get("n_iter") or 30)
        self.required = bool(raw.get("required", False))

    def __repr__(self) -> str:
        return f"ModelSpec({self.name}:{self.kind})"

    @property
    def is_tree(self) -> bool:
        return self.kind in TREE_KINDS

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "search_type": self.search_type,
                "n_iter": self.n_iter, "search": self.search, "fixed": self.fixed}


def resolve_models(cfg: dict, task: str) -> list[ModelSpec]:
    """解析配置里的 models，并保证 PLS 基线存在。"""
    specs = [ModelSpec(m, task) for m in (cfg.get("models") or [])]
    for spec in specs:
        if spec.kind not in SUPPORTED_KINDS:
            raise ValueError(f"未知模型 kind={spec.kind!r}，可选：{sorted(SUPPORTED_KINDS)}")
    if not any(s.kind == "pls" for s in specs):
        injected = ModelSpec({
            "name": "pls(自动注入)", "kind": "pls", "search_type": "grid", "required": True,
            "search": {"n_components": [3, 5, 8, 10, 12, 15, 20]},
        }, task)
        specs.insert(0, injected)
        warn("配置缺少 PLS 基线，已自动注入：化学计量学论文必须有 PLS 对照，否则审稿人第一刀就砍这里")
    # 名字去重（防止用户两个模型都叫 "m1"）
    seen: dict[str, int] = {}
    for spec in specs:
        if spec.name in seen:
            seen[spec.name] += 1
            spec.name = f"{spec.name}_{seen[spec.name]}"
        else:
            seen[spec.name] = 1
    for spec in specs:
        if spec.kind == "gp" and task == "classification":
            warn("GP 分类用 Laplace 近似（O(n³)），高维光谱下极慢，建议仅在小样本（n<500）时使用")
    if cfg.get("preprocess", {}).get("candidates"):
        validate_spaces(specs, cfg)
    return specs


# --------------------------------------------------------------------------- #
# 估计器
# --------------------------------------------------------------------------- #
def build_estimator(spec: ModelSpec, seed: int, apply_fixed: bool = True) -> BaseEstimator:
    est = _build_estimator_base(spec, seed)
    if spec.fixed and apply_fixed:
        # 容忍 `fixed` 里带 model__ 前缀（用户从 search 里复制过来时的常见写法）
        applied = {str(k).removeprefix("model__"): v for k, v in spec.fixed.items()}
        try:
            est.set_params(**applied)
        except ValueError as exc:
            warn(f"模型 {spec.name} 的 fixed 参数无法应用（{exc}），已忽略：{spec.fixed}")
    return est


def _build_estimator_base(spec: ModelSpec, seed: int) -> BaseEstimator:
    kind = spec.kind
    task = spec.task
    reg = task == "regression"

    if kind == "pls":
        # 分类任务用 PLS-DA（概率接口）；回归用 SafePLS
        return SafePLS(n_components=5) if reg else SafePLSDA(n_components=5)
    if kind == "pcr":
        return SafePCR(n_components=10) if reg else SafePCRDA(n_components=10)
    if kind == "ridge":
        return Ridge(alpha=1.0) if reg else RidgeClassifier(alpha=1.0)
    if kind == "lasso":
        if reg:
            return Lasso(alpha=0.01, max_iter=20000, random_state=seed)
        # L1 逻辑回归：sklearn 1.9 起 penalty 已弃用、1.11 移除，用 l1_ratio 表达罚项类型。
        # l1_ratio=1.0 即 L1；saga 是唯一支持 L1 的求解器。
        return LogisticRegression(l1_ratio=1.0, C=1.0, solver="saga", max_iter=5000,
                                  class_weight="balanced", random_state=seed)
    if kind == "enet":
        if reg:
            return ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=20000, random_state=seed)
        return LogisticRegression(l1_ratio=0.5, C=1.0, solver="saga",
                                  max_iter=5000, class_weight="balanced", random_state=seed)
    if kind == "svr":
        return SVR(C=10.0, epsilon=0.1, gamma="scale") if reg else SVC(
            C=10.0, gamma="scale", probability=True, class_weight="balanced", random_state=seed)
    if kind == "svc":
        return SVC(C=10.0, gamma="scale", probability=True, class_weight="balanced", random_state=seed)
    if kind == "knn":
        return KNeighborsRegressor(n_neighbors=5) if reg else KNeighborsClassifier(n_neighbors=5)
    if kind == "rf":
        base = dict(n_estimators=300, random_state=seed, n_jobs=1,
                    min_samples_leaf=1, max_features="sqrt")
        return RandomForestRegressor(**base) if reg else RandomForestClassifier(
            class_weight="balanced_subsample", **base)
    if kind == "et":
        base = dict(n_estimators=300, random_state=seed, n_jobs=1, max_features="sqrt")
        return ExtraTreesRegressor(**base) if reg else ExtraTreesClassifier(
            class_weight="balanced", **base)
    if kind == "gbr":
        if reg:
            return GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                             max_depth=3, random_state=seed)
        return GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                          max_depth=3, random_state=seed)
    if kind == "xgb":
        from xgboost import XGBClassifier, XGBRegressor

        base = dict(n_estimators=300, learning_rate=0.05, max_depth=4,
                    subsample=0.8, colsample_bytree=0.5, reg_lambda=1.0,
                    random_state=seed, n_jobs=1, tree_method="hist", verbosity=0)
        # 注意：xgboost 3.x 的 early_stopping_rounds 只能写在构造器，且需要 eval_set；
        # 内层搜索里不传 eval_set，因此这里固定 n_estimators，不用早停。
        if reg:
            return XGBRegressor(**base)
        return XGBClassifier(**base, eval_metric="logloss")
    if kind == "lgbm":
        import lightgbm as lgb

        base = dict(n_estimators=300, learning_rate=0.05, num_leaves=15,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                    reg_lambda=1.0, random_state=seed, n_jobs=1,
                    verbose=-1, min_child_samples=5)
        if reg:
            return lgb.LGBMRegressor(**base)
        return lgb.LGBMClassifier(**base, class_weight="balanced")
    if kind == "cat":
        from catboost import CatBoostClassifier, CatBoostRegressor

        base = dict(iterations=300, learning_rate=0.05, depth=4, random_seed=seed,
                    verbose=0, allow_writing_files=False)  # 不写 catboost_info/ 污染目录
        if reg:
            return CatBoostRegressor(**base)
        return CatBoostClassifier(**base, auto_class_weights="Balanced")
    if kind == "gp":
        kernel = ConstantKernel(1.0, (1e-2, 1e3)) * RBF(10.0, (1e-1, 1e3)) + WhiteKernel(1e-2)
        if reg:
            return GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                            random_state=seed, n_restarts_optimizer=0)
        # 分类用 Laplace 近似的 GP（有 predict_proba）；注意它没有 alpha 参数——
        # 噪声方差由 WhiteKernel 与优化器承担，因此分类的默认搜索空间为空。
        return GaussianProcessClassifier(kernel=kernel, random_state=seed,
                                         n_restarts_optimizer=0)
    if kind == "logreg":
        return LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced",
                                  random_state=seed, solver="lbfgs")
    if kind == "ridge_clf":
        return RidgeClassifier(alpha=1.0, class_weight="balanced")
    if kind == "gnb":
        return GaussianNB()
    if kind == "lda":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    raise ValueError(f"未实现的 kind={kind!r}")


def build_pipeline(
    spec: ModelSpec,
    cfg: dict,
    feature_names: list[str] | None = None,
    seed: int = 0,
    apply_fixed: bool = True,
) -> Pipeline:
    """预处理 → 标准化 → 模型，全部装在一个 Pipeline 里。

    预处理候选（``prep__which``）与模型超参一起在内层搜索，保证：
    * 任何 fit-time 统计量（MSC 参考谱、标准化均值方差）只看训练折；
    * "挑预处理"这个动作本身也在交叉验证内部完成。

    ``apply_fixed=False`` 只给配置期校验用：那时 fixed 还没校验过，先别应用，
    否则拼错的键会先打一句「已忽略」的 warn 再报错，读起来自相矛盾。
    """
    cands = cfg["preprocess"]["candidates"]
    steps: list[tuple[str, Any]] = [
        # 缺失值必须在 Pipeline 内插补：中位数只看当前训练折，测试折不参与。
        # 放在预处理之前，避免 SNV/MSC 等算子把 NaN 传播到整行。
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("prep", PrepSwitcher(candidates=cands, which=0, feature_names=feature_names)),
    ]
    if cfg["preprocess"].get("scale", True):
        steps.append(("scale", StandardScaler(with_mean=bool(cfg["preprocess"].get("scale_with_mean", False)))))
    steps.append(("model", build_estimator(spec, seed, apply_fixed=apply_fixed)))
    return Pipeline(steps)


def is_tree(spec: ModelSpec) -> bool:
    return spec.is_tree


def uses_shap(spec: ModelSpec) -> bool:
    return spec.is_tree


# --------------------------------------------------------------------------- #
# 搜索空间
# --------------------------------------------------------------------------- #
DEFAULT_SPACES: dict[str, dict[str, Any]] = {
    "pls": {"n_components": [3, 5, 8, 10, 12, 15, 20, 25]},
    "pcr": {"n_components": [5, 10, 15, 20, 30]},
    "ridge": {"alpha": {"dist": "loguniform", "low": 1e-3, "high": 1e4}},
    "lasso": {"alpha": {"dist": "loguniform", "low": 1e-4, "high": 1e1}},
    "enet": {"alpha": {"dist": "loguniform", "low": 1e-4, "high": 1e1},
             "l1_ratio": [0.1, 0.5, 0.9]},
    "svr": {"C": {"dist": "loguniform", "low": 1e-1, "high": 1e3},
            "gamma": {"dist": "loguniform", "low": 1e-4, "high": 1e0},
            "epsilon": [0.01, 0.05, 0.1, 0.2]},
    "svc": {"C": {"dist": "loguniform", "low": 1e-1, "high": 1e3},
            "gamma": {"dist": "loguniform", "low": 1e-4, "high": 1e0}},
    "knn": {"n_neighbors": [2, 3, 5, 7, 10, 15]},
    "rf": {"n_estimators": [200, 500], "max_depth": [None, 5, 10, 20],
           "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", 0.1, 0.3]},
    "et": {"n_estimators": [200, 500], "max_depth": [None, 10, 20],
           "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", 0.1, 0.3]},
    "gbr": {"n_estimators": [100, 200, 400], "learning_rate": {"dist": "loguniform", "low": 0.01, "high": 0.3},
            "max_depth": [2, 3, 4, 5]},
    "xgb": {"n_estimators": {"dist": "int", "low": 200, "high": 800},
            "max_depth": [3, 4, 5, 6],
            "learning_rate": {"dist": "loguniform", "low": 0.01, "high": 0.3},
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.2, 0.4, 0.6, 1.0],
            "reg_lambda": {"dist": "loguniform", "low": 1e-2, "high": 1e2}},
    "lgbm": {"n_estimators": {"dist": "int", "low": 200, "high": 800},
             "num_leaves": [7, 15, 31],
             "learning_rate": {"dist": "loguniform", "low": 0.01, "high": 0.3},
             "subsample": [0.6, 0.8, 1.0],
             "colsample_bytree": [0.2, 0.4, 0.6, 1.0],
             "reg_lambda": {"dist": "loguniform", "low": 1e-2, "high": 1e2}},
    "cat": {"iterations": {"dist": "int", "low": 200, "high": 800},
            "depth": [3, 4, 5, 6],
            "learning_rate": {"dist": "loguniform", "low": 0.01, "high": 0.3},
            "l2_leaf_reg": {"dist": "loguniform", "low": 1.0, "high": 30.0}},
    "gp": {"alpha": {"dist": "loguniform", "low": 1e-8, "high": 1e-1}},
    "logreg": {"C": {"dist": "loguniform", "low": 1e-3, "high": 1e3}},
    "ridge_clf": {"alpha": {"dist": "loguniform", "low": 1e-3, "high": 1e3}},
    "gnb": {"var_smoothing": {"dist": "loguniform", "low": 1e-11, "high": 1e-3}},
    "lda": {},
}

# 分类任务下必须替换的搜索空间：同一个 kind 在两个任务里落到的估计器不同，
# 参数名也就不同（Lasso 用 alpha / L1 逻辑回归用 C；SVR 有 epsilon / SVC 没有；
# GP 回归有 alpha / GP 分类没有）。写错名字的症状和 pcr 那次一模一样：
# 每一折 Invalid parameter → 整模型静默从结果里消失。
_CLF_SPACES: dict[str, dict[str, Any]] = {
    "lasso": {"C": {"dist": "loguniform", "low": 1e-3, "high": 1e2}},
    "enet": {"C": {"dist": "loguniform", "low": 1e-3, "high": 1e2},
             "l1_ratio": [0.1, 0.5, 0.9]},
    "svr": {"C": {"dist": "loguniform", "low": 1e-1, "high": 1e3},
            "gamma": {"dist": "loguniform", "low": 1e-4, "high": 1e0}},
    "gp": {},   # GaussianProcessClassifier 无 alpha，kernel 由优化器内部拟合
}


def default_space(kind: str, task: str) -> dict[str, Any]:
    """按 (kind, 任务) 返回默认搜索空间。分类有专门覆盖时优先用覆盖。"""
    if task == "classification" and kind in _CLF_SPACES:
        return copy.deepcopy(_CLF_SPACES[kind])
    return copy.deepcopy(DEFAULT_SPACES.get(kind, {}))


def search_space(spec: ModelSpec) -> dict[str, Any]:
    """返回带 ``model__`` 前缀的搜索空间（prep__which 由 tune 统一加）。"""
    raw = spec.search or default_space(spec.kind, spec.task)
    out: dict[str, Any] = {}
    for key, val in raw.items():
        full = key if "__" in key else f"model__{key}"
        if isinstance(val, dict) and "dist" in val:
            out[full] = dict(val)
        elif isinstance(val, (list, tuple)):
            vals = list(val)
            # 单值等价于固定，跳过
            out[full] = vals
        else:
            out[full] = [val]
    _drop_fixed(out, spec.fixed or {})
    return out


def _drop_fixed(space: dict[str, Any], fixed: dict[str, Any]) -> None:
    """fixed 优先于 search：删掉与之冲突的搜索键。

    必须按**末级参数名**比较，而不能只认 ``model__{key}``——否则
    ``fixed={{n_components: 10}}`` 压不住 ``model__pca__n_components`` 这类嵌套键。
    """
    for key in fixed:
        plain = str(key).removeprefix("model__")
        for full in list(space):
            if full == key or full == f"model__{plain}" or full.endswith(f"__{plain}"):
                space.pop(full, None)


def _space_error_message(spec: ModelSpec, bad: list[str], valid: set[str]) -> str:
    import difflib

    lines = []
    for key in bad:
        near = difflib.get_close_matches(key, sorted(valid), n=2, cutoff=0.5)
        hint = f"，是否想写 {' 或 '.join(near)}？" if near else ""
        lines.append(f"  · {key}{hint}")
    avail = sorted(k for k in valid if k.startswith("model__"))
    return (f"模型 {spec.name!r}（kind={spec.kind}）的搜索空间里有无法被估计器解析的参数：\n"
            + "\n".join(lines)
            + f"\n  该估计器可调的 model__ 参数共 {len(avail)} 个：{avail[:12]}"
            + ("…" if len(avail) > 12 else ""))


# get_params() 的已知盲区：这些参数能通过 set_params 生效，但不出现在 get_params 里。
# 只在**校验**处使用，不影响搜索（搜索键仍需带 model__ 前缀）。
_GET_PARAMS_BLIND_SPOTS: dict[str, set[str]] = {
    "cat": {"l2_leaf_reg"},   # catboost 只报告构造器显式收到的少量键
}


def _probe_value(val: Any) -> Any:
    """从搜索空间的一项里取一个可用于试探设置的值。"""
    if isinstance(val, (list, tuple, np.ndarray)) and len(val):
        return val[0]
    if isinstance(val, dict) and "dist" in val:
        if val.get("values"):
            return val["values"][0]
        return val.get("low", 1)
    return val


def validate_spaces(specs: list[ModelSpec], cfg: dict) -> None:
    """配置期契约校验：搜索键必须真的能被 ``set_params`` 接受。

    这项校验的价值不在于"更优雅地报错"，而在于把失败从"第 90 折静默消失"
    提前到"开跑前 0.1 秒"。没有它，用户自定义 ``search`` 里任何一个拼写错误
    都会让整个模型从结果表中蒸发，而日志里只有一行 warn。

    判定方式用"两把尺子"：

    * 以 ``get_params()`` 为主——它才是 ``GridSearchCV`` 认的白名单，能挡住
      ``model__n_estimator`` 这类笔误。单用 set_params 挡不住：xgboost/catboost
      会静默接受未知 kwarg，不报错、安静地用默认值训练，比崩掉更坏。
    * 再用 ``set_params`` 确认取值本身可接受（例如 ``C=-1``）。
    * ``get_params()`` 也有盲区（catboost 的 ``l2_leaf_reg`` 能设置但不在其中），
      故另有一张显式例外表 ``_GET_PARAMS_BLIND_SPOTS``。
    """
    from .config import ConfigError

    for spec in specs:
        if spec.task == "regression" and spec.kind in CLASSIFICATION_ONLY:
            raise ConfigError(
                "E-MOD-003",
                f"模型 {spec.name!r}（kind={spec.kind}）只提供分类器，无法用于回归任务。"
                f"回归请改用 {sorted(SUPPORTED_KINDS - CLASSIFICATION_ONLY)[:8]}… "
                f"（如 ridge / lasso / svr / knn / rf）。")
        space = search_space(spec)
        if not space and not spec.fixed:
            # 既没有搜索空间也没有 fixed：这个模型只按构造器默认值训练，没什么可校验的
            continue
        try:
            probe = build_pipeline(spec, cfg, feature_names=None, seed=0, apply_fixed=False)
            valid = set(probe.get_params().keys())
        except Exception as exc:  # noqa: BLE001 —— 估计器本身构造失败也要早报
            raise ConfigError(
                "E-MOD-001", f"模型 {spec.name!r}（kind={spec.kind}）无法在 {spec.task} 任务下构造："
                             f"{type(exc).__name__}: {exc}") from exc
        # 判定方式：以 get_params() 为主（它才是 GridSearchCV 认的白名单）。
        # 单用 set_params 不够——xgboost/catboost 会**静默接受**任意未知 kwarg
        # （直接塞进 booster 参数），所以 `model__n_estimator` 这种笔误不会报错，
        # 只会安静地用默认值训练，比崩掉更坏。
        # 反过来 get_params() 也有盲区：catboost 的 l2_leaf_reg 能通过 set_params
        # 却不出现在 get_params 里，硬用白名单会误杀，故给出显式例外表。
        blind = _GET_PARAMS_BLIND_SPOTS.get(spec.kind, set())
        bad: list[str] = []
        for key, val in space.items():
            leaf = key.split("__")[-1]
            if key not in valid and not (key.startswith("model__") and leaf in blind):
                bad.append(key)
                continue
            try:   # 名字对了还要确认取值能被接受（例如 C=-1 这类）
                probe.set_params(**{key: _probe_value(val)})
            except ValueError as exc:
                bad.append(f"{key}（{str(exc).replace(chr(10), ' ')[:80]}）")
        if bad:
            raise ConfigError("E-MOD-002", _space_error_message(spec, bad, valid))

        # fixed 里的键走同一套白名单。旧实现只在 build_estimator 里 try/except + warn，
        # 于是"fixed 里参数名拼错"等于安静地用默认值训练——比 search 拼错更危险，
        # 因为 fixed 是用户明确想钉死的值（如 subsample=0.8），静默失效根本看不出来。
        bad_fixed: list[str] = []
        for key in (spec.fixed or {}):
            k = str(key)
            if not k.startswith("model__"):
                k = f"model__{k}"
            leaf = k.split("__")[-1]
            if k not in valid and leaf not in blind:
                bad_fixed.append(k)
        if bad_fixed:
            raise ConfigError("E-MOD-004", _space_error_message(spec, bad_fixed, valid)
                              .replace("搜索空间里有无法被估计器解析的参数",
                                       "fixed 里有无法被估计器解析的参数（这些值本该被钉死，"
                                       "拼错会静默失效）"))


def model_summary(specs: list[ModelSpec]) -> list[dict[str, Any]]:
    return [s.as_dict() for s in specs]


def clone_pipeline(pipe: Pipeline) -> Pipeline:
    return clone(pipe)


def prep_label(cfg: dict, index: int) -> str:
    cands = cfg["preprocess"]["candidates"]
    if 0 <= int(index) < len(cands):
        return prep_name(cands[int(index)])
    return f"cand{index}"
