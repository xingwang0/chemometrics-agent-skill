"""光谱预处理算子与 sklearn 兼容封装。

三条硬规则（违反任一条就会让结果虚高）
------------------------------------
1. 有跨样本统计的算子（MSC 参考谱）必须实现为 Transformer，在 ``fit`` 里只吃训练折数据；
2. 预处理链整体作为一个超参参与内层搜索（``PrepSwitcher``），消除"人工挑预处理"的乐观偏置；
3. 改变维度（波段裁剪）时同步维护特征名，否则 permutation importance 的列映射会错位。

单行作用的算子（SNV / SG / 导数 / 归一化 / 基线）天然无泄漏，但仍统一走 ``SpectralPrep``，
以保证"预处理链"是一个可整体 fit 的对象。
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from scipy.linalg import solveh_banded
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = [
    "SpectralPrep",
    "PrepSwitcher",
    "snv",
    "msc",
    "airpls",
    "poly_baseline",
    "build_prep",
    "describe_steps",
    "prep_name",
]

EPS = 1e-12

# --------------------------------------------------------------------------- #
# 预处理折级缓存（L3）
# --------------------------------------------------------------------------- #
# 为什么可以缓存：本模块所有算子都是**无监督**的（只看 X，不看 y），同一条预处理链
# 在同一份输入上重复 fit/transform 的结果完全相同。而嵌套 CV 里，同一折、同一预处理
# 会被每个候选超参各算一遍（光谱 2000 列时 airPLS 一次就要几秒），这是纯重复计算。
#
# 安全边界：
# * 键 = (预处理链签名, fit 输入指纹, 本次输入指纹)，指纹是整块内容的 blake2b，
#   形状 + 全字节参与，不存在"看起来一样但内容不同"的误命中；
# * 只缓存 **无监督** 结果；算子白名单在 ``_apply_rowwise`` 里硬校验，未知算子在
#   fit 阶段就会抛错，不可能悄悄把依赖 y 的算子塞进来；
# * 命中时返回的是同一次计算产出的数组（同一份数值），不是重新近似。
import hashlib
import json
import os
from collections import OrderedDict

_CACHE_MB = float(os.environ.get("SML_PREP_CACHE_MB", "96") or 96)
_CACHE_ON = os.environ.get("SML_PREP_CACHE", "1") not in ("0", "off", "false")
_OutCache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_OutCacheBytes = 0
_FitCache: "OrderedDict[tuple, dict]" = OrderedDict()


def _fingerprint(A: np.ndarray) -> tuple:
    """整块内容的指纹：形状 + blake2b(全部字节)。用于判断"是不是同一份输入"。"""
    A = np.ascontiguousarray(A, dtype=np.float32)
    h = hashlib.blake2b(A.tobytes(), digest_size=16).hexdigest()
    return (A.shape, h)


def _steps_sig(steps: list[dict], feature_names) -> tuple:
    """预处理链签名：算子参数 + 特征名（只影响输出名，但要一起区分）。"""
    return (json.dumps(steps, sort_keys=True, ensure_ascii=False, default=str),
            None if feature_names is None else hash(tuple(str(n) for n in feature_names)))


def clear_prep_cache() -> None:
    """清空缓存（大 fold 之间可手动调用；测试里也用）。"""
    global _OutCacheBytes
    _OutCache.clear()
    _FitCache.clear()
    _OutCacheBytes = 0


def _cache_get_out(key: tuple):
    arr = _OutCache.get(key)
    if arr is None:
        return None
    _OutCache.move_to_end(key)
    return arr


def _cache_put_out(key: tuple, arr: np.ndarray) -> None:
    global _OutCacheBytes
    if not _CACHE_ON:
        return
    nbytes = int(arr.nbytes)
    if nbytes > _CACHE_MB * 1024 * 1024:
        return
    while _OutCache and (_OutCacheBytes + nbytes) > _CACHE_MB * 1024 * 1024:
        _, old = _OutCache.popitem(last=False)
        _OutCacheBytes -= int(old.nbytes)
    _OutCache[key] = arr
    _OutCacheBytes += nbytes
    while _OutCache and _OutCacheBytes > _CACHE_MB * 1024 * 1024:
        _, old = _OutCache.popitem(last=False)
        _OutCacheBytes -= int(old.nbytes)


def _cache_get_fit(key: tuple):
    st = _FitCache.get(key)
    if st is None:
        return None
    _FitCache.move_to_end(key)
    return st


def _cache_put_fit(key: tuple, state: dict) -> None:
    if not _CACHE_ON:
        return
    _FitCache[key] = state
    while len(_FitCache) > 64:
        _FitCache.popitem(last=False)


# --------------------------------------------------------------------------- #
# 单算子实现（全部对单行向量工作）
# --------------------------------------------------------------------------- #
def snv(row: np.ndarray) -> np.ndarray:
    """标准正态变量变换：(x - mean) / std，逐行消除散射与光程差异。"""
    row = np.asarray(row, dtype=np.float64)
    sd = row.std()
    if sd < EPS:
        return np.zeros_like(row)
    return (row - row.mean()) / sd


def msc(row: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """多元散射校正：把样品的谱回归到参考谱上，x = a*ref + b -> (x-b)/a。"""
    row = np.asarray(row, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    denom = float(np.dot(ref, ref))
    if denom < EPS:
        return row - row.mean()
    a = float(np.dot(row, ref)) / denom
    b = float(row.mean() - a * ref.mean())
    if abs(a) < EPS:
        return row - b
    return (row - b) / a


def poly_baseline(row: np.ndarray, order: int = 3) -> np.ndarray:
    """多项式基线：拟合低阶多项式趋势并扣除。"""
    n = row.size
    x = np.linspace(-1.0, 1.0, n)
    order = int(min(max(order, 0), 8))
    coef = np.polyfit(x, row, order)
    return row - np.polyval(coef, x)


def _diff_matrix(n: int, order: int) -> sparse.csc_matrix:
    """构造 (n-order) × n 的 order 阶差分矩阵，供 airPLS 的平滑惩罚项使用。"""
    order = int(max(1, min(order, max(1, n - 2))))
    D = sparse.eye(n, format="csc")
    for _ in range(order):
        D = (D[1:] - D[:-1]).tocsc()
    return D


def _airpls_bands(n: int, order: int, lam: float) -> tuple[np.ndarray, int]:
    """把 ``lam * D'D`` 压成对称带状形式，供 ``solveh_banded`` 直接求解。

    返回 ``(ab, kd)``：``ab`` 形状 ``(kd+1, n)``，按 scipy 的上带约定
    ``ab[kd + i - j, j] == A[i, j]`` 存放（主对角在第 ``kd`` 行）。
    每轮迭代只需把对角线上加上权重 ``w``——这是把 airPLS 从"每行每轮重建稀疏
    矩阵 + 稀疏 LU"降到"O(n) 带状 Cholesky"的关键，结果与原稀疏解一致（1e-12 量级）。
    """
    kd = int(max(1, min(order, max(1, n - 2))))
    D = _diff_matrix(n, kd)
    DT_D = (D.T @ D).tocsc()
    ab = np.zeros((kd + 1, n), dtype=np.float64)
    for d in range(kd + 1):
        vals = DT_D.diagonal(d)
        if vals.size:
            ab[kd - d, d:] = lam * vals
    return ab, kd


def _airpls_sparse(y: np.ndarray, lam: float, order: int, max_iter: int,
                   tol: float) -> np.ndarray:
    """原始实现（稀疏 LU），仅在 lam<=0 或带状求解失败时作为兜底，保证旧行为不变。"""
    n = y.size
    D = _diff_matrix(n, order)
    DT_D = (D.T @ D).tocsc()
    w = np.ones(n)
    z = y.copy()
    for it in range(1, max_iter + 1):
        W = sparse.diags(w, 0, shape=(n, n), format="csc")
        z = sparse.linalg.spsolve((W + lam * DT_D).tocsc(), w * y)
        d = y - z
        neg = d < 0
        neg_sum = float(np.abs(d[neg]).sum())
        if neg_sum < tol * float(np.abs(y).sum()) or neg_sum < EPS:
            break
        w = np.zeros(n)
        if neg.any():
            w[neg] = np.exp(it * np.abs(d[neg]) / neg_sum)
            w[0] = np.exp(it * float(np.abs(d[neg]).max()) / neg_sum)
            w[-1] = w[0]
    return y - z


def _airpls_core(y: np.ndarray, ab_base: np.ndarray, kd: int, lam: float, order: int,
                 max_iter: int, tol: float) -> np.ndarray:
    n = y.size
    w = np.ones(n)
    z = y.copy()
    for it in range(1, max_iter + 1):
        ab = ab_base.copy()
        ab[kd] += w
        try:
            z = solveh_banded(ab, w * y, lower=False, check_finite=False)
        except np.linalg.LinAlgError:
            # 理论上不会发生（lam>0 且 W 半正定 → 强正定）；真遇到就回退，不静默出错
            return _airpls_sparse(y, lam, order, max_iter, tol)
        d = y - z
        neg = d < 0
        neg_sum = float(np.abs(d[neg]).sum())
        if neg_sum < tol * float(np.abs(y).sum()) or neg_sum < EPS:
            break
        w = np.zeros(n)
        if neg.any():
            w[neg] = np.exp(it * np.abs(d[neg]) / neg_sum)
            w[0] = np.exp(it * float(np.abs(d[neg]).max()) / neg_sum)
            w[-1] = w[0]
    return y - z


def airpls(row: np.ndarray, lam: float = 1e5, order: int = 2, max_iter: int = 20,
           tol: float = 1e-3) -> np.ndarray:
    """自适应迭代重加权惩罚最小二乘基线校正（Zhang et al., 2010）。

    线性方程组 ``(W + lam·D'D) z = W·y`` 是**对称正定、带宽 = order 的带状矩阵**，
    用带状 Cholesky 求解；n=2000、max_iter=20 时单行约 1–2 ms（原稀疏 LU 约 15–20 ms）。
    """
    y = np.asarray(row, dtype=np.float64)
    n = y.size
    if n < 6:
        return y - y.min()
    if lam <= 0:
        # lam=0 时惩罚项消失，系统可能奇异；退回原来的稀疏求解以保持旧行为
        return _airpls_sparse(y, lam, order, max_iter, tol)
    ab_base, kd = _airpls_bands(n, order, lam)
    return _airpls_core(y, ab_base, kd, lam, order, max_iter, tol)


def _sg(row: np.ndarray, window: int, polyorder: int, deriv: int, delta: float) -> np.ndarray:
    window = int(window)
    if window % 2 == 0:
        raise ValueError(f"Savitzky-Golay window 必须为奇数，收到 {window}（scipy 1.18 会静默接受偶数）")
    if window > row.size:
        window = row.size if row.size % 2 == 1 else row.size - 1
        if window <= int(polyorder):
            return row.copy()
    return savgol_filter(row, window_length=window, polyorder=int(polyorder),
                         deriv=int(deriv), delta=float(delta), mode="interp")


# --------------------------------------------------------------------------- #
# 批量算子：与上面的逐行版本逐行等价，只是把 Python 循环换成一次矩阵运算。
# 光谱数据动辄 1000–2000 列 × 数百行，逐行 vstack 的 Python 开销会占掉相当比例，
# 而 SG/SNV/MSC/多项式基线本身都是沿轴的整体运算。
# --------------------------------------------------------------------------- #
def _snv_rows(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    if A.shape[0] == 0:
        return A.astype(np.float32)
    out = np.zeros_like(A)
    mean = A.mean(axis=1, keepdims=True)
    sd = A.std(axis=1, keepdims=True)
    ok = (sd[:, 0] >= EPS)
    if ok.any():
        out[ok] = (A[ok] - mean[ok]) / sd[ok]
    return out.astype(np.float32)


def _msc_rows(A: np.ndarray, ref: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if A.shape[0] == 0:
        return A.astype(np.float32)
    denom = float(np.dot(ref, ref))
    if denom < EPS:
        return (A - A.mean(axis=1, keepdims=True)).astype(np.float32)
    a = (A @ ref) / denom
    b = A.mean(axis=1) - a * ref.mean()
    out = A.copy()
    small = np.abs(a) < EPS
    big = ~small
    if small.any():
        out[small] = A[small] - b[small, None]
    if big.any():
        out[big] = (A[big] - b[big, None]) / a[big, None]
    return out.astype(np.float32)


def _poly_rows(A: np.ndarray, order: int) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    if A.shape[0] == 0:
        return A.astype(np.float32)
    n = A.shape[1]
    x = np.linspace(-1.0, 1.0, n)
    order = int(min(max(order, 0), 8))
    V = np.vander(x, order + 1)              # 降幂，与 np.polyfit / np.polyval 一致
    coef = np.linalg.pinv(V) @ A.T
    return (A - (V @ coef).T).astype(np.float32)


def _sg_rows(A: np.ndarray, window: int, polyorder: int, deriv: int, delta: float) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    window = int(window)
    if window % 2 == 0:
        raise ValueError(f"Savitzky-Golay window 必须为奇数，收到 {window}（scipy 1.18 会静默接受偶数）")
    if A.shape[0] == 0:
        return A.astype(np.float32)
    if window > A.shape[1]:
        # 窗口比谱段还长：逐行回退，保持与原实现一致的边界语义
        return np.vstack([_sg(r, window, polyorder, deriv, delta) for r in A]).astype(np.float32)
    return savgol_filter(A, window_length=window, polyorder=int(polyorder),
                         deriv=int(deriv), delta=float(delta), axis=1,
                         mode="interp").astype(np.float32)


def _airpls_rows(A: np.ndarray, lam: float, order: int, iters: int) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    if A.shape[0] == 0:
        return A.astype(np.float32)
    n = A.shape[1]
    if n < 6 or lam <= 0:
        return np.vstack([airpls(r, lam, order, iters) for r in A]).astype(np.float32)
    ab_base, kd = _airpls_bands(n, order, lam)     # 每批只构造一次惩罚带
    out = np.empty_like(A)
    for i in range(A.shape[0]):
        out[i] = _airpls_core(A[i], ab_base, kd, lam, order, iters, 1e-3)
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# SpectralPrep：把算子链做成一个可 fit 的 Transformer
# --------------------------------------------------------------------------- #
class SpectralPrep(BaseEstimator, TransformerMixin):
    """按顺序应用预处理链；只在 fit 时用训练折数据估计 MSC 参考谱。

    Parameters
    ----------
    steps : list[dict]
        每个元素形如 ``{"op": "sg", "window": 11, "polyorder": 2, "deriv": 1}``。
    feature_names : list[str] | None
        输入特征名，用于波段裁剪后同步输出名。
    """

    def __init__(self, steps: list[dict] | None = None, feature_names: list[str] | None = None):
        # 注意：sklearn.clone 会校验构造器前后的参数同一性，因此这里绝不可做
        # ``steps if steps is not None else []`` 之类的替换，只能在用的时候兜底。
        self.steps = steps
        self.feature_names = feature_names

    @property
    def _steps(self) -> list[dict]:
        return list(self.steps or [])

    # -- 内部 ------------------------------------------------------------- #
    def _slice_index(self, step: dict, n_features: int) -> np.ndarray:
        lo = step.get("from_index", 0)
        hi = step.get("to_index", n_features)
        if lo is None:
            lo = 0
        if hi is None:
            hi = n_features
        lo, hi = int(lo), int(hi)
        if not (0 <= lo < hi <= n_features):
            raise ValueError(f"波段裁剪区间 [{lo}, {hi}) 超出特征数 {n_features}")
        return np.arange(lo, hi)

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        sig = _steps_sig(self._steps, self.feature_names)
        fp_fit = _fingerprint(X)
        state = _cache_get_fit((sig, fp_fit))
        if state is not None:
            self._restore(state, sig, fp_fit)
            return self

        self.n_features_in_ = int(X.shape[1])
        names = list(self.feature_names) if self.feature_names is not None else None
        work = X
        self.ref_ = None
        self._col_index: np.ndarray | None = None
        # 多个 band / msc step 必须各自保存自己的索引与参考谱。旧实现只有一个
        # _col_index 和一个 ref_：transform 时每个 band 都用最终组合索引重复切片，
        # 每个 msc 都套用最后一个参考谱，结果静默错误且不会报错。
        self._band_indices_: list[np.ndarray] = []
        self._msc_refs_: list[np.ndarray] = []
        for step in self._steps:
            op = step.get("op")
            if op == "band":
                idx = self._slice_index(step, work.shape[1])
                self._band_indices_.append(idx)
                work = work[:, idx]
                prev = self._col_index
                self._col_index = idx if prev is None else prev[idx]
                if names is not None:
                    names = [names[i] for i in idx]
            elif op == "msc":
                ref = np.asarray(work, dtype=np.float64).mean(axis=0)
                self._msc_refs_.append(ref)
                self.ref_ = ref
                work = _msc_rows(np.asarray(work, dtype=np.float64), ref)
            else:
                work = self._apply_rowwise(np.asarray(work, dtype=np.float32), step)
        self.output_feature_names_ = names
        self._sig_ = sig
        self._fit_fp_ = fp_fit
        _cache_put_fit((sig, fp_fit), {
            "n_features_in_": self.n_features_in_,
            "ref_": self.ref_,
            "col_index": self._col_index,
            "bands": self._band_indices_,
            "msc": self._msc_refs_,
            "names": names,
        })
        # fit 的结果就是 transform(训练折) 的结果，同一个键直接复用，省掉一次整链计算
        _cache_put_out((sig, fp_fit, fp_fit), work)
        return self

    def _restore(self, state: dict, sig: tuple, fp_fit: tuple) -> None:
        self.n_features_in_ = int(state["n_features_in_"])
        self.ref_ = state["ref_"]
        self._col_index = state["col_index"]
        self._band_indices_ = list(state["bands"])
        self._msc_refs_ = list(state["msc"])
        self.output_feature_names_ = state["names"]
        self._sig_ = sig
        self._fit_fp_ = fp_fit

    def _apply_rowwise(self, X: np.ndarray, step: dict) -> np.ndarray:
        op = step.get("op")
        if op == "snv":
            return _snv_rows(np.asarray(X, dtype=np.float64))
        if op == "normalize":
            mode = step.get("mode", "l2")
            A = np.asarray(X, dtype=np.float64)
            if mode == "l2":
                scale = np.linalg.norm(A, axis=1, keepdims=True)
            elif mode == "max":
                scale = np.abs(A).max(axis=1, keepdims=True)
            elif mode == "area":
                scale = np.abs(A).sum(axis=1, keepdims=True)
            else:
                raise ValueError(f"未知归一化模式 {mode!r}")
            scale[scale < EPS] = 1.0
            return (A / scale).astype(np.float32)
        if op == "sg":
            win = int(step.get("window", 11))
            poly = int(step.get("polyorder", 2))
            deriv = int(step.get("deriv", 0))
            delta = float(step.get("delta", 1.0))
            return _sg_rows(np.asarray(X, dtype=np.float64), win, poly, deriv, delta)
        if op == "baseline_poly":
            order = int(step.get("order", 3))
            return _poly_rows(np.asarray(X, dtype=np.float64), order)
        if op == "airpls":
            lam = float(step.get("lam", 1e5))
            order = int(step.get("order", 2))
            iters = int(step.get("max_iter", 20))
            return _airpls_rows(np.asarray(X, dtype=np.float64), lam, order, iters)
        if op == "diff":
            n = int(step.get("n", 1))
            return np.diff(np.asarray(X, dtype=np.float32), n=n, axis=1).astype(np.float32)
        raise ValueError(f"未知预处理算子 {op!r}")

    def transform(self, X):
        out_in = np.asarray(X, dtype=np.float32)
        sig = getattr(self, "_sig_", None)
        fp_in = _fingerprint(out_in) if sig is not None else None
        if sig is not None:
            hit = _cache_get_out((sig, getattr(self, "_fit_fp_", None), fp_in))
            if hit is not None:
                return hit
        out = out_in
        band_i = msc_i = 0
        for step in self._steps:
            op = step.get("op")
            if op == "band":
                if band_i >= len(getattr(self, "_band_indices_", [])):
                    raise RuntimeError("波段裁剪未 fit，无法 transform")
                out = out[:, self._band_indices_[band_i]]
                band_i += 1
            elif op == "msc":
                if msc_i >= len(getattr(self, "_msc_refs_", [])):
                    raise RuntimeError("MSC 未 fit，无法 transform")
                out = _msc_rows(np.asarray(out, dtype=np.float64), self._msc_refs_[msc_i])
                msc_i += 1
            else:
                out = self._apply_rowwise(out, step)
        if sig is not None:
            _cache_put_out((sig, getattr(self, "_fit_fp_", None), fp_in), out)
        return out

    def get_feature_names_out(self, input_features=None):
        if self.output_feature_names_ is not None:
            return np.asarray(self.output_feature_names_, dtype=object)
        if input_features is None:
            return np.asarray([f"f{i}" for i in range(self.n_features_in_)], dtype=object)
        names = list(input_features)
        if self._col_index is not None:
            names = [names[i] for i in self._col_index]
        return np.asarray(names, dtype=object)


class PrepSwitcher(BaseEstimator, TransformerMixin):
    """把「预处理候选链」也变成超参：``which`` 由内层搜索决定。

    这样"预处理是挑过的"这一审稿质疑在方法学上被消掉——它是搜索空间的一维。
    """

    def __init__(self, candidates: list[dict] | None = None, which: int = 0,
                 feature_names: list[str] | None = None):
        self.candidates = candidates
        self.which = which
        self.feature_names = feature_names

    @property
    def _candidates(self) -> list[dict]:
        return list(self.candidates or [{"name": "raw", "steps": []}])

    def fit(self, X, y=None):
        cands = self._candidates
        idx = int(self.which) % len(cands)
        cand = cands[idx]
        self.prep_ = SpectralPrep(steps=list(cand.get("steps") or []),
                                  feature_names=self.feature_names).fit(X, y)
        self.active_name_ = str(cand.get("name") or f"cand{idx}")
        self.active_index_ = idx
        self.n_features_in_ = int(np.asarray(X).shape[1])
        return self

    def transform(self, X):
        return self.prep_.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.prep_.get_feature_names_out(input_features)


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #
def build_prep(candidate: dict, feature_names: list[str] | None = None) -> SpectralPrep:
    return SpectralPrep(steps=list(candidate.get("steps") or []), feature_names=feature_names)


def prep_name(candidate: dict) -> str:
    name = candidate.get("name")
    if name:
        return str(name)
    return describe_steps(candidate.get("steps") or [])


def describe_steps(steps: list[dict]) -> str:
    if not steps:
        return "raw"
    parts = []
    for s in steps:
        op = s.get("op")
        if op == "sg":
            tag = f"sg{s.get('window', 11)}-p{s.get('polyorder', 2)}"
            if int(s.get("deriv", 0)):
                tag += f"-d{s['deriv']}"
            parts.append(tag)
        elif op == "normalize":
            parts.append(f"norm_{s.get('mode', 'l2')}")
        elif op == "band":
            parts.append(f"band[{s.get('from_index')}:{s.get('to_index')}]")
        elif op == "baseline_poly":
            parts.append(f"poly{s.get('order', 3)}")
        elif op == "airpls":
            parts.append(f"airpls{int(np.log10(float(s.get('lam', 1e5))))}")
        else:
            parts.append(str(op))
    return "_".join(parts)


def describe_candidates(candidates: list[dict]) -> list[dict[str, Any]]:
    return [{"index": i, "name": prep_name(c), "steps": c.get("steps") or []}
            for i, c in enumerate(candidates)]
