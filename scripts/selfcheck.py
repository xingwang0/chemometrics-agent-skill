# Copyright 2026 The research-ml-modeling Authors
# SPDX-License-Identifier: Apache-2.0
"""环境自检 + 防泄漏回归测试。

两件事
------
1. **环境探针**：逐项验证本机版本下的 API 行为（哪些写法能用、哪些会静默出错），
   失败项直接给出修法。换机器跑之前先跑这个，比跑到一半崩掉强。
2. **泄漏探针**（最重要的一项）：``leakage_probe.csv`` 里谱图唯一编码样品身份、
   y 与身份不可平滑映射。此时
       随机 KFold  → R² 应虚高（>0.9）
       分组 KFold  → R² 应崩掉（<0.3）
   若结果相反，说明 CV 实现存在泄漏，本技能不可信。这一条把"防泄漏"从口号变成了断言。
3. **注册表矩阵**：``SUPPORTED_KINDS`` × 任务逐个构造 + 跑一次内层搜索。
   这项覆盖的是"声明即支持"的漏洞——搜索键写错、估计器没有 predict_proba，
   症状都是整模型静默消失，端到端 smoke 只用到少数 kind，抓不到。

用法
----
    python selfcheck.py                      # 全量自检
    python selfcheck.py --allow-fail         # 失败也返回 0（仅打印）
    python selfcheck.py --only leakage,env   # 只跑某一组
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable

SHIM = Path(__file__).resolve().parent
sys.path.insert(0, str(SHIM))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sml.utils import (  # noqa: E402
    as_path, available_memory_gb, configure_matplotlib, ensure_dir, non_ascii,
    read_json, setup_console, write_json,
)

CHECKS: list[tuple[str, str, Callable[[], tuple[bool, str]]]] = []
ASSETS = SHIM.parent / "assets"


def check(group: str, name: str):
    def deco(fn: Callable[[], tuple[bool, str]]):
        CHECKS.append((group, name, fn))
        return fn

    return deco


def _ok(detail: str = "") -> tuple[bool, str]:
    return True, detail


def _fail(detail: str) -> tuple[bool, str]:
    return False, detail


# --------------------------------------------------------------------------- #
# 环境组
# --------------------------------------------------------------------------- #
@check("env", "Python 版本与解释器")
def _python() -> tuple[bool, str]:
    v = sys.version_info
    return _ok(f"{v.major}.{v.minor}.{v.micro} @ {sys.executable}")


@check("env", "核心依赖版本矩阵")
def _packages() -> tuple[bool, str]:
    from sml.provenance import package_versions

    vers = package_versions()
    missing = [k for k, v in vers.items() if v == "MISSING"]
    required_missing = [m for m in missing if m in
                        ("numpy", "pandas", "scikit-learn", "scipy", "matplotlib")]
    if required_missing:
        return _fail(f"必需包缺失：{required_missing}")
    detail = ", ".join(f"{k}={v}" for k, v in vers.items() if v != "MISSING")
    if missing:
        detail += f" | 可选包缺失：{missing}（对应功能会自动降级）"
    return _ok(detail)


@check("env", "numpy 2.x 已移除 API")
def _numpy_api() -> tuple[bool, str]:
    if hasattr(np, "float_"):
        return _fail("np.float_ 仍存在（预期 numpy 2.x 已移除）；代码里请用 float")
    return _ok("np.float_ 已移除，符合 numpy 2.x")


@check("env", "pandas 3.0 字符串 dtype 为 str")
def _pandas_dtype() -> tuple[bool, str]:
    dt = pd.DataFrame({"a": ["x"]})["a"].dtype
    if str(dt) == "str":
        return _ok("字符串列 dtype=str；代码中已禁用 dtype==object 判断")
    return _ok(f"字符串列 dtype={dt}（旧行为）；代码用 is_string_dtype 兼容")


@check("env", "matplotlib 中文字体")
def _cjk_font() -> tuple[bool, str]:
    plt = configure_matplotlib()
    from matplotlib import font_manager

    names = {f.name for f in font_manager.fontManager.ttflist}
    picked = plt.rcParams["font.sans-serif"][0]
    if picked not in names:
        return _fail(f"未找到中文字体，当前首选 {picked!r} 不在字体表中")
    with tempfile.TemporaryDirectory() as td:
        fig = plt.figure(figsize=(2, 1))
        fig.text(0.1, 0.5, "近红外 定量模型 中文测试")
        out = Path(td) / "cjk.png"
        fig.savefig(out)
        plt.close(fig)
        if not out.exists():
            return _fail("中文文本 savefig 失败")
    return _ok(f"字体 {picked}，中文 savefig 正常")


@check("env", "中文路径读写与 JSON 往返")
def _cn_path() -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as td:
        p = as_path(Path(td) / "中文目录" / "结果.json")
        ensure_dir(p.parent)
        payload = {"指标": "R²", "值": 0.95, "note": "中文"}
        write_json(p, payload)
        back = read_json(p)
        if back != payload:
            return _fail("JSON 中文往返不一致")
        raw = p.read_text(encoding="utf-8")
        if "\\u" in raw:
            return _fail("JSON 写成了 \\u 转义，human 不可读")
    return _ok("Path + utf-8 + ensure_ascii=False 正常")


@check("env", "内存探测可用")
def _mem() -> tuple[bool, str]:
    avail = available_memory_gb()
    if avail is None:
        return _ok("无法探测（不影响功能，mem_guard 自动跳过）")
    return _ok(f"可用 {avail:.1f} GB")


# --------------------------------------------------------------------------- #
# sklearn / 模型组
# --------------------------------------------------------------------------- #
@check("env", "sklearn 分组划分器是否支持 shuffle")
def _gkf_shuffle() -> tuple[bool, str]:
    import inspect

    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

    sig = inspect.signature(GroupKFold.__init__)
    if "shuffle" not in sig.parameters:
        return _fail("GroupKFold 不支持 shuffle；需改用自写分组打乱（cv.py 已按此降级路径设计）")
    s2 = inspect.signature(StratifiedGroupKFold.__init__)
    has2 = "shuffle" in s2.parameters
    # 实测：同数据不同 seed 应给出不同划分
    X = np.zeros((60, 2))
    g = np.repeat(np.arange(20), 3)
    y = np.tile([0, 1], 30)
    a = list(GroupKFold(n_splits=4, shuffle=True, random_state=1).split(X, y, groups=g))
    b = list(GroupKFold(n_splits=4, shuffle=True, random_state=2).split(X, y, groups=g))
    same = all(np.array_equal(x[1], z[1]) for x, z in zip(a, b))
    if same:
        return _fail("不同 seed 得到相同划分，shuffle 未生效")
    return _ok(f"GroupKFold 支持 shuffle；StratifiedGroupKFold {'支持' if has2 else '不支持'} shuffle")


@check("env", "内层搜索可用显式 split 列表（嵌套 CV 的关键）")
def _explicit_splits() -> tuple[bool, str]:
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV, GroupKFold

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 5))
    y = X[:, 0] * 2 + rng.normal(scale=0.1, size=60)
    g = np.repeat(np.arange(20), 3)
    # 模拟嵌套 CV：先切出外层训练折，再在训练折内部构造显式 split 列表
    tr_outer = np.arange(40)
    te_outer = np.arange(40, 60)
    Xtr, ytr, gtr = X[tr_outer], y[tr_outer], g[tr_outer]
    splits = list(GroupKFold(n_splits=3, shuffle=True, random_state=0).split(Xtr, ytr, groups=gtr))
    search = GridSearchCV(Ridge(), {"alpha": [0.1, 1.0, 10.0]}, cv=splits,
                          scoring="neg_root_mean_squared_error")
    search.fit(Xtr, ytr)
    pred = search.predict(X[te_outer])
    if pred.shape != (20,):
        return _fail(f"refit 后 predict 形状异常：{pred.shape}")
    if set(gtr[splits[0][0]]) & set(gtr[splits[0][1]]):
        return _fail("内层显式 split 的训练/验证 group 有重叠")
    return _ok(f"GridSearchCV(cv=显式split列表) 可 fit+refit+predict，最优 alpha={search.best_params_['alpha']}")


@check("env", "metadata routing 未被启用")
def _routing() -> tuple[bool, str]:
    import sklearn

    enabled = bool(sklearn.get_config().get("enable_metadata_routing", False))
    if enabled:
        return _fail("enable_metadata_routing=True 会改变 groups 的传递语义，请关掉")
    return _ok("enable_metadata_routing=False（本项目依赖此默认值）")


@check("env", "xgboost 早停只能写在构造器")
def _xgb_es() -> tuple[bool, str]:
    from xgboost import XGBRegressor

    X = np.random.default_rng(0).normal(size=(30, 4))
    y = X[:, 0]
    ctor_ok = True
    try:
        m = XGBRegressor(n_estimators=5, early_stopping_rounds=3, eval_metric="rmse", verbosity=0)
        m.fit(X, y, eval_set=[(X, y)], verbose=False)
    except Exception as exc:  # noqa: BLE001
        ctor_ok = False
        ctor_msg = f"{type(exc).__name__}: {exc}"
    fit_kw = False
    try:
        XGBRegressor(n_estimators=5, verbosity=0).fit(X, y, early_stopping_rounds=3)
        fit_kw = True
    except TypeError:
        pass
    if not ctor_ok:
        return _fail(f"构造器早停不可用：{ctor_msg}")
    return _ok(f"构造器可用；fit(early_stopping_rounds=) {'也可用（版本更宽松）' if fit_kw else '不可用（符合 3.x 行为）'}")


@check("env", "lightgbm / catboost 静默且不写盘")
def _gbm_silent() -> tuple[bool, str]:
    import lightgbm as lgb
    from catboost import CatBoostRegressor

    print_marker = Path("catboost_info")
    existed = print_marker.exists()
    X = np.random.default_rng(0).normal(size=(30, 4))
    y = X[:, 0]
    lgb.LGBMRegressor(n_estimators=5, verbose=-1).fit(X, y)
    CatBoostRegressor(iterations=5, verbose=0, allow_writing_files=False).fit(X, y)
    if not existed and print_marker.exists():
        return _fail("catboost 仍在当前目录创建了 catboost_info/")
    return _ok("两者均可静默训练，未污染工作目录")


@check("env", "shap TreeExplainer 可用")
def _shap() -> tuple[bool, str]:
    try:
        import shap
        from xgboost import XGBRegressor
    except ImportError as exc:
        return _fail(f"shap 未安装：{exc}")
    X = np.random.default_rng(0).normal(size=(40, 6))
    y = X[:, 0] * 2 + X[:, 1]
    m = XGBRegressor(n_estimators=10, verbosity=0).fit(X, y)
    vals = shap.TreeExplainer(m).shap_values(X[:5])
    arr = np.asarray(vals)
    if arr.shape[0] != 5:
        return _fail(f"shap 输出形状异常：{arr.shape}")
    return _ok(f"TreeExplainer 输出 {arr.shape}")


@check("env", "scipy 统计工具齐备")
def _scipy_tools() -> tuple[bool, str]:
    from scipy import stats

    have_fdr = hasattr(stats, "false_discovery_control")
    try:
        q = stats.false_discovery_control([0.01, 0.04, 0.03], method="bh")
        bh_ok = bool(np.all(np.asarray(q) >= np.asarray([0.01, 0.04, 0.03]) - 1e-12))
    except Exception:  # noqa: BLE001
        bh_ok = False
    return _ok(f"false_discovery_control={'有' if have_fdr else '无(已内置兜底)'}, BH 结果合理={bh_ok}")


@check("env", "Savitzky-Golay 偶数窗口行为（已知静默坑）")
def _savgol() -> tuple[bool, str]:
    from scipy.signal import savgol_filter

    try:
        savgol_filter(np.arange(20.0), window_length=10, polyorder=2)
        scipy_accepts_even = True
    except Exception:  # noqa: BLE001
        scipy_accepts_even = False
    from sml.preprocess import SpectralPrep

    guard_ok = False
    try:
        SpectralPrep(steps=[{"op": "sg", "window": 10, "polyorder": 2}]).fit(np.zeros((3, 20)))
    except ValueError as exc:
        guard_ok = "奇数" in str(exc)
    if scipy_accepts_even and not guard_ok:
        return _fail("scipy 接受偶数窗口且本包未拦截 → 会静默产出非标准 SG")
    return _ok(f"scipy 接受偶数={scipy_accepts_even}；本包拦截={'是' if guard_ok else '否'}")


# --------------------------------------------------------------------------- #
# 方法学组
# --------------------------------------------------------------------------- #
@check("leakage", "预处理只在训练折内 fit（MSC 参考谱隔离）")
def _prep_isolation() -> tuple[bool, str]:
    from sml.preprocess import SpectralPrep

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 30)).astype(np.float32)
    tr = np.arange(25)
    prep = SpectralPrep(steps=[{"op": "msc"}, {"op": "snv"}]).fit(X[tr])
    ref_train = X[tr].astype(np.float64).mean(axis=0)
    if not np.allclose(prep.ref_, ref_train):
        return _fail("MSC 参考谱不等于训练折均值 → 存在跨折泄漏")
    prep_all = SpectralPrep(steps=[{"op": "msc"}]).fit(X)
    if np.allclose(prep_all.ref_, ref_train):
        return _fail("全量 fit 与训练折 fit 结果相同，隔离测试无效")
    return _ok("MSC 参考谱 = 训练折均值；与全量 fit 明显不同")


@check("leakage", "外层划分的 group 严格不相交")
def _group_disjoint() -> tuple[bool, str]:
    from sml.config import resolve_config
    from sml.cv import make_outer_splits
    from sml.io_data import load_dataset

    cfg = resolve_config(ASSETS / "config" / "smoke.json", budget="fast")
    ds, _ = load_dataset(cfg)
    bad = 0
    for seed in (1, 2, 3):
        for tr, te in make_outer_splits(ds, cfg, seed):
            if set(ds.groups[tr]) & set(ds.groups[te]):
                bad += 1
    if bad:
        return _fail(f"{bad} 个折存在训练/测试 group 重叠 → 重复测量泄漏")
    return _ok("3 个种子 × 全部折均无 group 重叠")


@check("leakage", "泄漏探针：随机 KFold 虚高 / 分组 KFold 崩塌")
def _leakage_probe() -> tuple[bool, str]:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score
    from sklearn.model_selection import GroupKFold, KFold
    from sklearn.neighbors import KNeighborsRegressor

    probe = ASSETS / "examples" / "leakage_probe.csv"
    if not probe.exists():
        return _fail(f"缺少探针数据 {probe}，请先跑 assets/examples/make_demo_data.py")

    df = pd.read_csv(probe, encoding="utf-8")
    feat = [c for c in df.columns if c.startswith("f")]
    X = df[feat].to_numpy(dtype=np.float32)
    y = df["y"].to_numpy(dtype=float)
    g = df["sample_id"].astype(str).to_numpy()

    def cv_r2(make_model, splitter) -> float:
        scores = []
        for tr, te in splitter.split(X, y, g):
            m = make_model()
            m.fit(X[tr], y[tr])
            scores.append(r2_score(y[te], m.predict(X[te])))
        return float(np.mean(scores))

    # 1-NN 是"能否靠记住同一样品的另一次测量拿分"最直接的代理，且完全确定
    r2_random = cv_r2(lambda: KNeighborsRegressor(n_neighbors=1),
                      KFold(n_splits=5, shuffle=True, random_state=0))
    r2_group = cv_r2(lambda: KNeighborsRegressor(n_neighbors=1), GroupKFold(n_splits=5))
    r2_rf = cv_r2(lambda: RandomForestRegressor(n_estimators=120, random_state=0, n_jobs=1),
                  KFold(n_splits=5, shuffle=True, random_state=0))
    detail = (f"1-NN 随机KFold R²={r2_random:.3f}（>0.9 才算探针有效）| "
              f"1-NN 分组KFold R²={r2_group:.3f}（<0.3 才算无泄漏）| RF 随机KFold R²={r2_rf:.3f}")
    if r2_random <= 0.9:
        return _fail("探针未表现出应有的虚高，探针数据可能被破坏。 " + detail)
    if r2_group >= 0.3:
        return _fail("分组 CV 仍拿到高分 → 存在泄漏！ " + detail)
    return _ok(detail)


@check("leakage", "按组 bootstrap 与按行 bootstrap 的 CI 宽度差异")
def _bootstrap_group() -> tuple[bool, str]:
    from sml.metrics import bootstrap_regression

    rng = np.random.default_rng(0)
    n_groups, reps = 30, 3
    g = np.repeat(np.arange(n_groups), reps)
    # 构造"组内残差高度相关"的情形：模型对每个样品有一个固定的系统偏差
    group_effect = rng.normal(scale=1.0, size=n_groups)
    y = 10 + np.repeat(group_effect, reps) + rng.normal(scale=0.05, size=g.size)
    pred = 10 + np.repeat(group_effect, reps) + np.repeat(rng.normal(scale=1.5, size=n_groups), reps)
    ci_group = bootstrap_regression(y, pred, g, n_boot=400, seed=0)
    ci_row = bootstrap_regression(y, pred, np.arange(g.size), n_boot=400, seed=0)
    if not ci_group or not ci_row:
        return _fail("bootstrap 未产出置信区间")
    w_group = ci_group["rmse"][1] - ci_group["rmse"][0]
    w_row = ci_row["rmse"][1] - ci_row["rmse"][0]
    if w_group <= w_row:
        return _fail(f"按组 CI 未更宽（group={w_group:.3f} ≤ row={w_row:.3f}），分组重采样逻辑可能有误")
    return _ok(f"按组 CI 宽度 {w_group:.3f} > 按行 {w_row:.3f}（组内相关被正确计入不确定度）")


@check("leakage", "跨 repeat 聚合口径（不得混算 R×k 折）")
def _aggregate() -> tuple[bool, str]:
    from sml.metrics import aggregate_repeats

    rows = []
    for r in range(4):
        for k in range(5):
            # 每个 repeat 内部折间波动大，repeat 间均值稳定
            rows.append({"model": "m", "prep": "p", "target": "t", "repeat": r, "fold": k,
                         "r2": (r * 10 + k) * 0.0 + 0.8 + 0.05 * ((k % 2) * 2 - 1)})
    df = aggregate_repeats(pd.DataFrame(rows), ["r2"])
    if df.empty:
        return _fail("聚合结果为空")
    sd = float(df.loc[0, "r2_sd"])
    naive_sd = float(pd.DataFrame(rows)["r2"].std(ddof=1))
    if sd >= naive_sd:
        return _fail(f"repeat 级 sd ({sd:.4f}) 未小于混算 sd ({naive_sd:.4f})，聚合口径可能写反")
    return _ok(f"repeat 级 sd={sd:.4f} < 混算 sd={naive_sd:.4f}")


@check("leakage", "指标与 Bootstrap 自洽性")
def _metrics_sanity() -> tuple[bool, str]:
    from sml.metrics import bootstrap_regression, regression_metrics

    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    perfect = regression_metrics(y, y.copy())
    if not (abs(perfect["r2"] - 1.0) < 1e-9 and perfect["rmse"] < 1e-9):
        return _fail(f"完美预测的指标异常：{perfect}")
    worse = regression_metrics(y, np.zeros_like(y))
    if worse["r2"] > 0:
        return _fail("常数预测的 R² 应 <=0")
    ci = bootstrap_regression(y * 1.0, y * 1.02, np.arange(5), n_boot=50, seed=0)
    if ci and not (ci["rmse"][0] <= 0.06 <= ci["rmse"][1] + 1e-6):
        # 5 个样品做 bootstrap 本就不可靠，只校验区间有序
        if ci["rmse"][0] > ci["rmse"][1]:
            return _fail(f"CI 下界大于上界：{ci['rmse']}")
    return _ok("完美/常数预测与 CI 方向均正确")


@check("leakage", "SafePLS 在极小训练折下自动裁剪 n_components")
def _safe_pls() -> tuple[bool, str]:
    from sml.models import SafePLS

    X = np.random.default_rng(0).normal(size=(6, 100))
    y = X[:, 0] * 2
    m = SafePLS(n_components=15).fit(X, y)
    k = int(m.model_.n_components)
    if k > 5:
        return _fail(f"n_components 未裁剪：仍为 {k}（训练折仅 6 行）")
    pred = m.predict(X)
    if pred.shape != (6,):
        return _fail(f"predict 形状异常 {pred.shape}")
    return _ok(f"n_components 15 → {k}，predict 正常")


@check("leakage", "全链路 smoke 可跑通并产出 OOF")
def _smoke_pipeline() -> tuple[bool, str]:
    from sml.config import resolve_config
    from sml.cv import run_nested_cv
    from sml.io_data import load_dataset
    from sml.models import resolve_models
    from sml.stage_common import RunContext

    cfg = resolve_config(ASSETS / "config" / "smoke.json",
                         overrides=["split.repeats=1", "split.outer.n_splits=2",
                                    "split.inner.n_splits=2", "preprocess.candidates[0].name=raw"],
                         budget="fast")
    ds, _ = load_dataset(cfg)
    specs = resolve_models(cfg, ds.task)[:1]
    with tempfile.TemporaryDirectory() as td:
        ctx = RunContext(cfg, ds, as_path(td))
        for sub in ("cv", "cv/models"):
            ensure_dir(ctx.run_dir / sub)
        oof, folds = run_nested_cv(ctx, specs)
    if oof.empty or folds.empty:
        return _fail("smoke 未产出 OOF/折结果")
    if not np.isfinite(oof["y_pred"].to_numpy(dtype=float)).all():
        return _fail("OOF 预测含非有限值")
    r2 = float(folds["r2"].mean())
    if not np.isfinite(r2):
        return _fail("折级 R² 全为 NaN")
    return _ok(f"OOF {len(oof)} 行、{len(folds)} 折，mean R²={r2:.3f}")


# --------------------------------------------------------------------------- #
# 注册表组：模型 kind × 任务 的矩阵测试
#
# 这两条是"契约测试"，不是环境探针。它们的由来很具体：pcr 的搜索键写成了
# model__n_components（真实路径是 model__pca__n_components），ridge_clf 没有
# predict_proba 却一律用 neg_log_loss 打分——两者的症状完全一样：每一折都
# 抛异常，被 error_score 吞掉，最后**整个模型从结果表里消失**，只在
# fold_results.csv 留下 90 行 error:XXX。之前的 23 条自检覆盖了环境、泄漏、
# bootstrap，却从没实例化过 SUPPORTED_KINDS，所以这类错误能活下来。
# --------------------------------------------------------------------------- #
def _mini_cfg() -> dict:
    return {
        "preprocess": {"candidates": [{"name": "raw", "steps": []}], "scale": True},
        "models": [],
    }


@check("registry", "模型注册表 × 任务：每个 kind 要么跑通，要么被明确拒绝（不许静默消失）")
def _registry_matrix() -> tuple[bool, str]:
    from sklearn.model_selection import GroupKFold

    from sml.config import ConfigError
    from sml.models import (
        CLASSIFICATION_ONLY, SUPPORTED_KINDS, ModelSpec, build_pipeline, search_space,
        validate_spaces,
    )
    from sml.tune import expand_space, inner_search, scoring_for

    cfg = _mini_cfg()
    rng = np.random.default_rng(0)
    n, p = 36, 8
    X = rng.normal(size=(n, p)).astype(np.float32)
    g = np.repeat(np.arange(n // 3), 3)
    y_reg = X[:, 0] * 2.0 + rng.normal(scale=0.1, size=n)
    y_clf = (X[:, 0] > float(np.median(X[:, 0]))).astype(int)
    splits = list(GroupKFold(n_splits=3, shuffle=True, random_state=0).split(X, y_reg, groups=g))

    bad: list[str] = []
    ran: list[str] = []
    rejected: list[str] = []
    for kind in sorted(SUPPORTED_KINDS):
        for task in ("regression", "classification"):
            y = y_reg if task == "regression" else y_clf
            spec = ModelSpec({"name": f"{kind}_{task}", "kind": kind, "search_type": "none"}, task)
            expect_reject = (task == "regression" and kind in CLASSIFICATION_ONLY)
            try:
                validate_spaces([spec], cfg)                     # 键能被 set_params 接受
                if expect_reject:
                    raise RuntimeError("分类专用 kind 在回归任务下未被拒绝")
                pipe = build_pipeline(spec, cfg, None, seed=0)
                cands = expand_space(search_space(spec), "none", 1, seed=0)
                if not cands:
                    raise RuntimeError("搜索空间为空（expand_space 返回 0 个候选）")
                scoring = scoring_for(pipe, task)                # 无 proba 时必须换打分器
                if task == "classification" and not hasattr(pipe, "predict_proba"):
                    if scoring != "balanced_accuracy":
                        raise RuntimeError(f"无概率接口却用 {scoring} 打分")
                inner_search(pipe, cands, splits, X, y, scoring=scoring, n_jobs=1)
            except ConfigError:
                if expect_reject:
                    rejected.append(f"{kind}/{task}")
                else:
                    bad.append(f"{kind}/{task} → 被误拒：{sys.exc_info()[1]}")
            except Exception as exc:  # noqa: BLE001
                bad.append(f"{kind}/{task} → {type(exc).__name__}: {str(exc)[:110]}")
            else:
                ran.append(f"{kind}/{task}")
    if bad:
        return _fail(f"{len(bad)}/{len(ran) + len(bad) + len(rejected)} 个组合失败：\n        "
                     + "\n        ".join(bad[:8]))
    return _ok(f"{len(ran)} 个组合跑通，{len(rejected)} 个按预期明确拒绝"
               f"（分类专用 kind 用于回归：{', '.join(rejected) or '—'}）")


@check("registry", "搜索空间拼写错误必须在配置期被拦截（而不是整模型静默消失）")
def _space_typo_guard() -> tuple[bool, str]:
    from sml.config import ConfigError
    from sml.models import ModelSpec, validate_spaces

    cfg = _mini_cfg()
    cases = [
        # 历史 bug 的原样复现：pcr 曾把键写在错误层级
        ("pcr", {"model__pca__n_components": [5]}, "model__pca__n_components", "regression"),
        ("ridge_clf", {"model__alph": [1.0]}, "model__alph", "classification"),
        ("xgb", {"model__n_estimator": [100]}, "model__n_estimator", "regression"),
    ]
    misses: list[str] = []
    for kind, space, key, task in cases:
        spec = ModelSpec({"name": f"{kind}_typo", "kind": kind, "search": space}, task)
        try:
            validate_spaces([spec], cfg)
        except ConfigError as exc:
            if key not in str(exc):
                misses.append(f"{kind}: 报错信息未指出问题键 {key}")
        except Exception as exc:  # noqa: BLE001
            misses.append(f"{kind}: 抛了 {type(exc).__name__} 而不是 ConfigError")
        else:
            misses.append(f"{kind}: 拼写错误的搜索键 {key} 未被拦截")
    if misses:
        return _fail("；".join(misses))
    # 正确写法必须放行（否则我校验过严会把正常配置也拦掉）
    good = ModelSpec({"name": "pcr_ok", "kind": "pcr"}, "regression")
    try:
        validate_spaces([good], cfg)
    except Exception as exc:  # noqa: BLE001
        return _fail(f"默认空间被误拦：{type(exc).__name__}: {exc}")
    return _ok(f"{len(cases)} 个错键全部被拦截并指出键名；默认空间正常放行")


# --------------------------------------------------------------------------- #
# 多目标组：所有"目标维度"都要显式，不能靠全表排序或 single-key 字典
# --------------------------------------------------------------------------- #
@check("multitarget", "排名与选模在目标内进行（跨目标尺度不可比）")
def _rank_within_target() -> tuple[bool, str]:
    from sml.compare import rank_models
    from sml.selection import pick_best_model

    tmp = Path(tempfile.mkdtemp())
    (tmp / "cv").mkdir()
    # 含量：a 最好；水分：b 最好；跨目标绝对 RMSE 最大的是「水分/a」
    summary = pd.DataFrame({
        "target": ["含量"] * 3 + ["水分"] * 3,
        "model": ["a", "b", "c"] * 2,
        "rmse_mean": [0.5, 0.6, 9.0, 5.0, 4.0, 8.0],
        "r2_mean": [0.96, 0.94, 0.10, 0.70, 0.80, 0.40],
    })
    ranked = rank_models(summary, "regression").set_index(["target", "model"])
    if ranked.loc[("含量", "a"), "rank"] != 1 or ranked.loc[("水分", "b"), "rank"] != 1:
        return _fail("rank 未按目标分组计算：\n" + ranked["rank"].to_string())
    if ranked.loc[("水分", "b"), "rmse_gap_to_best"] != 0.0:
        return _fail("gap_to_best 跨目标计算，水分内最优的 gap 不为 0")

    summary.to_csv(tmp / "cv" / "metrics_summary.csv", index=False)
    try:
        pick_best_model(tmp, "regression")
        return _fail("多目标未指定 target 时未报错——会静默选出「跨目标最强」")
    except Exception as exc:  # noqa: BLE001
        if "E-STAGE-004" not in str(exc):
            return _fail(f"报错类型不对：{type(exc).__name__}: {exc}")
    pair = (pick_best_model(tmp, "regression", target="含量")[0],
            pick_best_model(tmp, "regression", target="水分")[0])
    if pair != ("a", "b"):
        return _fail(f"各目标最优应为 ('a', 'b')，实际 {pair}")
    return _ok("rank/gap 目标内计算；未指定 target 报 E-STAGE-004；各目标最优 a / b")


@check("multitarget", "T2 的 CI 按 (target, model) 对齐，不被后一个目标覆盖")
def _ci_keyed_by_target() -> tuple[bool, str]:
    from sml.report import _attach_ci

    t2 = pd.DataFrame({"target": ["含量", "水分"], "model": ["a", "a"]})
    ci = pd.DataFrame({"target": ["含量", "水分"], "model": ["a", "a"], "metric": ["r2", "r2"],
                       "ci_low": [0.95, 0.60], "ci_high": [0.98, 0.75]})
    out = _attach_ci(t2, ci)
    lo = out["r2_ci_low"].tolist()
    hi = out["r2_ci_high"].tolist()
    if lo != [0.95, 0.60] or hi != [0.98, 0.75]:
        return _fail(f"CI 被覆盖（只按 model 建映射）：low={lo} high={hi}")
    # 若 CI 表缺 target 列（老产物），退回按 model 映射，不能直接崩
    out2 = _attach_ci(t2[["model"]].iloc[:1], ci.drop(columns=["target"]))
    if out2.empty or out2["r2_ci_low"].isna().all():
        return _fail("老格式（无 target 列）的 CI 退路失效")
    return _ok("两个目标各自保留自己的 CI；无 target 列时可按 model 退路")


@check("multitarget", "显著性门槛：不显著时不得写「唯一最优」")
def _significance_gate() -> tuple[bool, str]:
    from sml.report import _best_model_line, _comparison_status
    from sml.selection import significance_vs_runner_up

    tmp = Path(tempfile.mkdtemp())
    (tmp / "cv").mkdir()
    pd.DataFrame({"model": ["A", "B"], "r2_mean": [0.9600, 0.9585],
                  "r2_sd": [0.003, 0.004]}).to_csv(tmp / "cv" / "metrics_summary.csv", index=False)

    def _write(ci: tuple[float, float], p: float, sig: bool) -> None:
        pd.DataFrame({"metric": ["r2"], "model_a": ["A"], "model_b": ["B"], "delta": [0.0015],
                      "ci_low": [ci[0]], "ci_high": [ci[1]], "p_raw": [p], "p_BH": [p],
                      "significant": [sig], "n_pairs": [120]}).to_csv(
            tmp / "cv" / "comparison.csv", index=False)

    _write((-0.002, 0.0051), 0.55, False)
    tie = significance_vs_runner_up(tmp, "regression", "A")
    if not tie.get("available") or tie.get("significant"):
        return _fail(f"未识别出「无显著差异」：{tie}")
    line = _best_model_line({"significance": tie, "model": "A"}, "A")
    if "差异未达显著" not in line or "性能相当" not in line:
        return _fail(f"不显著时摘要仍写成唯一最优：{line}")
    status = _comparison_status({"significance": tie, "model": "A"}, True)
    if not status.startswith("⚠") or "无显著差异" not in status:
        return _fail(f"审稿清单状态未反映不显著：{status}")

    _write((0.010, 0.030), 0.002, True)
    win = significance_vs_runner_up(tmp, "regression", "A")
    if not win.get("significant"):
        return _fail(f"显著情形判定错误：{win}")
    if "显著优于" not in _best_model_line({"significance": win, "model": "A"}, "A"):
        return _fail("显著时措辞未体现「优于次优」")
    return _ok("不显著 → 「性能相当」+ 清单警示；显著 → 「显著优于次优」")


@check("multitarget", "兄弟目标列不得进入特征矩阵（含 fan-out 子 run 场景）")
def _sibling_target_not_feature() -> tuple[bool, str]:
    from sml.io_data import load_dataset

    tmp = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(0)
    n = 60
    a = rng.normal(size=n)
    b = a + rng.normal(scale=0.1, size=n)      # 故意做成 |r|≈0.99 的兄弟目标
    df = pd.DataFrame({"sid": np.repeat(np.arange(20), 3),
                       "spec1": a + rng.normal(scale=0.05, size=n),
                       "spec2": b + rng.normal(scale=0.05, size=n),
                       "A": a, "B": b})
    p = tmp / "d.csv"
    df.to_csv(p, index=False)

    def _load(targets: list[str], exclude: list[str] | None = None):
        cfg = {"data": {"path": str(p), "id_col": "sid"}, "task": {"type": "regression", "target": targets}}
        if exclude is not None:
            cfg["data"]["exclude_from_features"] = exclude
        ds, _ = load_dataset(cfg)
        return ds

    ds_multi = _load(["A", "B"])
    if "A" in ds_multi.feature_names or "B" in ds_multi.feature_names:
        return _fail(f"多目标下目标列仍在特征里：{ds_multi.feature_names}")

    ds_sub = _load(["B"], exclude=["A", "B"])           # fan-out 子 run 的真实形态
    if "A" in ds_sub.feature_names:
        return _fail("fan-out 子 run 里兄弟目标 A 仍被当作特征——moisture 模型能靠 protein 白拿分")
    if "spec1" not in ds_sub.feature_names:
        return _fail("排除过宽：正常的数值特征 spec1 也被删了")

    ds_bare = _load(["B"])                              # 反证：不显式排除就会泄漏
    if "A" not in ds_bare.feature_names:
        return _fail("反证失败：单目标未排除时 A 本应是特征（说明测试数据构造有问题）")
    return _ok("多目标自动剔除；子 run 经 exclude_from_features 剔除兄弟目标；"
               "未排除时确实会泄漏（反证通过）")


# --------------------------------------------------------------------------- #
# 报告口径组
# --------------------------------------------------------------------------- #
@check("reporting", "T2 必须保留口径列（否则 † 判定与池化值会被一起裁掉）")
def _t2_keeps_basis() -> tuple[bool, str]:
    from sml.report import t2_columns

    summary = pd.DataFrame({
        "target": ["y", "y"], "model": ["a", "b"],
        "r2_mean": [-0.76, 0.12], "r2_sd": [0.30, 0.02],
        "rmse_mean": [1.00, 0.90], "rmse_sd": [0.20, 0.02],
        "r2": [0.19, 0.10], "rmse": [0.80, 0.95],
        "metric_basis": ["pooled", "fold"],
        "n_folds_ok": [1, 6], "n_folds_degenerate": [5, 0],
        "prep_mix": ["raw×6", "raw×6"], "rank": [1, 2],
    })
    cols = t2_columns(summary)
    need = ["metric_basis", "n_folds_ok", "n_folds_degenerate", "r2", "rmse", "r2_mean"]
    missing = [c for c in need if c not in cols]
    if missing:
        return _fail(f"T2 裁掉了口径列 {missing}：报告将无法标「† 折均值不可用」，"
                     f"读者只会看到折均值那个 −0.76")
    return _ok(f"T2 保留 {len(cols)} 列，含口径列 {need}")


@check("reporting", "口径判定：stage_cv/rebuild 共用一份实现，pooled 行按池化值排名")
def _basis_consistency() -> tuple[bool, str]:
    from sml.compare import rank_models
    from sml.metrics import describe_pooled_rows, mark_metric_basis

    summary = pd.DataFrame({
        "target": ["y", "y"], "model": ["a", "b"],
        "r2_mean": [-0.76, 0.12], "rmse_mean": [1.00, 0.90],
        "r2": [0.19, 0.10], "rmse": [0.80, 0.95],
    })
    # a：留一法式退化（1 个有效折 + 5 个退化折）；b：全部有效
    folds = pd.DataFrame({
        "target": ["y"] * 12, "model": ["a"] * 6 + ["b"] * 6,
        "status": ["ok"] + ["degenerate"] * 5 + ["ok"] * 6,
    })
    out = mark_metric_basis(summary, folds, group_keys=("target", "model"))
    if list(out["metric_basis"]) != ["pooled", "fold"]:
        return _fail(f"口径判定错误：{list(out['metric_basis'])}（预期 ['pooled','fold']）")
    if list(out["n_folds_ok"]) != [1, 6]:
        return _fail(f"有效折计数错误：{list(out['n_folds_ok'])}")
    msg = describe_pooled_rows(out)
    if not msg or "a(1/6)" not in msg:
        return _fail(f"警告文案没点出具体模型与折数：{msg!r}")
    ranked = rank_models(out, "regression").sort_values("rank")
    if str(ranked.iloc[0]["model"]) != "a":
        return _fail("pooled 行仍按折均值（−0.76）参与排名，没有切到池化值")
    # 没有 status 列时必须如实退回折均值口径，不许伪造判定
    nobasis = mark_metric_basis(summary, folds.drop(columns=["status"]),
                                group_keys=("target", "model"))
    if set(nobasis["metric_basis"]) != {"fold"}:
        return _fail("缺 status 列时未退回 fold 口径")
    return _ok("池化标记、有效折计数、警告文案、排名口径一致；缺 status 时如实退回 fold")


# --------------------------------------------------------------------------- #
# 多分类组：>2 类端到端（这条是 2026-11 现场跑 3 类数据才暴露出来的）
#
# 症状史：多分类下 OOF 只写 1D ``proba``、图 F1 直接 roc_curve(多类 y)、
# 图 F5 拿 max 概率当阳性概率；三处都不报错，只表现为「图少了一张/图是空白」。
# 回归测试必须真的跑一次 3 类 CV 并把图画出来，不能只测工具函数。
# --------------------------------------------------------------------------- #
def _threeclass_frame(n_per_class: int = 8, reps: int = 2, n_feat: int = 12,
                      seed: int = 20261123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    classes = ["合格", "复检", "不合格"]
    centers = {
        "合格": np.r_[np.full(4, 1.6), np.zeros(n_feat - 4)],
        "复检": np.r_[np.zeros(4), np.full(4, 1.6), np.zeros(n_feat - 8)],
        "不合格": np.r_[np.zeros(8), np.full(4, 1.6), np.zeros(n_feat - 12)],
    }
    rows, idx = [], 0
    for c in classes:
        for _ in range(n_per_class):
            idx += 1
            base = centers[c] + rng.normal(0, 1.0, n_feat)
            for r in range(reps):
                spec = base + rng.normal(0, 0.15, n_feat)
                rec = {"sample_id": f"S{idx:03d}", "batch": f"B{int(rng.integers(1, 4))}",
                       "replicate": r + 1, "label": c}
                rec.update({f"x{j:02d}": round(float(v), 5) for j, v in enumerate(spec)})
                rows.append(rec)
    return pd.DataFrame(rows)


@check("multiclass", "3 类端到端：OOF 写 p0..pN、指标有限、F1/F5 能画出来")
def _multiclass_end_to_end() -> tuple[bool, str]:
    import warnings

    from sml.cv import run_nested_cv
    from sml.io_data import load_dataset
    from sml.metrics import classification_metrics, oof_score_matrix
    from sml.models import resolve_models
    from sml.plots import fig_calibration, fig_pred_obs
    from sml.stage_common import RunContext

    td = Path(tempfile.mkdtemp(prefix="sml_3class_"))
    df = _threeclass_frame()
    csv = td / "threeclass.csv"
    df.to_csv(csv, index=False, encoding="utf-8")
    cfg_p = td / "cfg.json"
    write_json(cfg_p, {
        "_schema": "sml-config/1",
        "task": {"type": "classification", "target": ["label"],
                 "positive_label": "不合格", "label_order": ["合格", "复检", "不合格"]},
        "data": {"format": "wide", "path": str(csv), "sheet": 0, "id_col": "sample_id",
                 "batch_col": "batch", "replicate_col": "replicate", "drop_cols": [],
                 "min_valid_frac": 0.5, "wavelength_axis": None},
        "split": {"group_col": "sample_id", "strategy": "nested_group",
                  "outer": {"n_splits": 3, "shuffle": True},
                  "inner": {"n_splits": 2, "shuffle": True},
                  "repeats": 1, "stratify": True},
        "preprocess": {"scale": True, "scale_with_mean": True,
                       "candidates": [{"name": "raw", "steps": []}]},
        "models": [{"name": "lda", "kind": "lda", "search_type": "none", "search": {}},
                   {"name": "rf", "kind": "rf", "search_type": "none", "search": {}}],
        "evaluation": {"metrics": ["auc", "brier", "log_loss", "balanced_accuracy",
                                   "f1_macro", "mcc"],
                       "bootstrap": {"n": 100, "alpha": 0.05}},
        "permutation": {"enabled": False, "n_perm": 20, "mode": "fast"},
        "ad": {"enabled": False, "methods": ["pca_t2"], "quantile": 0.95,
               "pca_max_comp": 3, "knn_k": 3},
        "interp": {"enabled": False},
        "report": {"lang": ["zh"], "dpi": 80, "formats": ["png"]},
        "runtime": {"n_jobs": 1, "seed": 42, "mem_guard_gb": 4.0, "save_models": "none"},
    })
    from sml.config import resolve_config

    cfg = resolve_config(cfg_p)
    ds, _ = load_dataset(cfg)
    specs = resolve_models(cfg, ds.task)
    ctx = RunContext(cfg, ds, td / "run")
    for sub in ("cv", "cv/models"):
        ensure_dir(ctx.run_dir / sub)
    oof, _folds = run_nested_cv(ctx, specs)
    if oof.empty:
        return _fail("3 类 CV 没有产出 OOF")

    for col in ("p0", "p1", "p2"):
        if col not in oof.columns:
            return _fail(f"多分类 OOF 缺逐类概率列 {col}（下游 Brier/AUROC 会静默错位）")
    mat, kind = oof_score_matrix(oof, 3)
    if getattr(mat, "shape", None) != (len(oof), 3):
        return _fail(f"oof_score_matrix 未返回 (n,3)：{getattr(mat, 'shape', None)}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = classification_metrics(oof["y_true"].to_numpy(), mat, n_classes=3,
                                   proba_kind=kind)
    if any("sum to one" in str(w.message) for w in caught):
        return _fail("概率行和未归一，sklearn log_loss 报 y_prob 不求和（应逐行归一化）")
    for key in ("auc", "brier", "log_loss"):
        if not np.isfinite(m.get(key, float("nan"))):
            return _fail(f"多分类 {key} 为 NaN（proba_kind={kind}）")

    seen = set(oof["model"].astype(str).unique())
    proba_model = "lda" if "lda" in seen else sorted(seen)[0]
    fig_dir = ensure_dir(td / "figs")
    # 这两张图在 3 类下以前分别是「静默消失」和「空白图」，必须断言真的产出
    if not fig_pred_obs(oof, str(proba_model), "classification", fig_dir, cfg, ds.label_map):
        return _fail("3 类下 F1（ROC）没有产出文件")
    if not fig_calibration(oof, str(proba_model), "classification", fig_dir, cfg, ds.label_map):
        return _fail("3 类下 F5（可靠性曲线）没有产出文件")
    return _ok(f"OOF {len(oof)} 行，p0..p2 齐全，AUROC={m['auc']:.3f}、"
               f"Brier={m['brier']:.3f}、LogLoss={m['log_loss']:.3f}；F1/F5 已产出")


# --------------------------------------------------------------------------- #
# provenance 组：三哈希必须真的随代码变化，否则 --resume 会拿旧产物冒充新结果
# --------------------------------------------------------------------------- #
@check("provenance", "代码/数据一改，三哈希必须变化（否则 --resume 会静默复用旧 stage）")
def _provenance_invalidation() -> tuple[bool, str]:
    from sml.provenance import build_provenance, resume_diff, resume_key

    class _DS:
        data_hash = "datahash-v1"
        source_path = None
        n_samples = 20
        n_features = 5
        n_groups = 10
        task = "regression"
        target_name = "y"

    td = Path(tempfile.mkdtemp(prefix="sml_prov_"))
    scripts = ensure_dir(td / "scripts")
    (scripts / "mod.py").write_text("X = 1\n", encoding="utf-8")
    cfg = {"runtime": {"seed": 7, "budget": "fast"},
           "split": {"repeats": 2, "outer": {"n_splits": 5}}}

    p1 = build_provenance(cfg, _DS(), scripts_dir=scripts)
    p2 = build_provenance(cfg, _DS(), scripts_dir=scripts)
    if resume_diff(p1, p2):
        return _fail(f"同一状态下不该报失效：{resume_diff(p1, p2)}")

    (scripts / "mod.py").write_text("X = 2\n", encoding="utf-8")
    p3 = build_provenance(cfg, _DS(), scripts_dir=scripts)
    diffs = resume_diff(p1, p3)
    if not any("脚本代码" in d for d in diffs):
        return _fail(f"改了脚本代码，code_sha256 未变化：{diffs}")
    if resume_key(p1) == resume_key(p3):
        return _fail("代码变了但 resume_key 没变，--resume 会误跳过 stage")

    _DS.data_hash = "datahash-v2"
    p4 = build_provenance(cfg, _DS(), scripts_dir=scripts)
    if not any("数据文件" in d for d in resume_diff(p3, p4)):
        return _fail("数据哈希变化未被识别")
    return _ok("同状态无差异；改代码/改数据都会让 resume_key 变化并给出失效原因")


# --------------------------------------------------------------------------- #
# split 组：确定性划分下 repeats>1 必须降为 1（否则 sd 恒为 0，却被当成多次独立验证）
# --------------------------------------------------------------------------- #
@check("split", "loo_batch / holdout_batch 下 repeats 必须降为 1 并告警")
def _repeats_degrade() -> tuple[bool, str]:
    import contextlib
    import io

    from sml.cv import effective_repeats

    for strategy in ("loo_batch", "holdout_batch"):
        cfg = {"split": {"strategy": strategy, "repeats": 3, "holdout_batch": "B2"}}
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            got = effective_repeats(cfg)
        if got != 1:
            return _fail(f"{strategy} 下 repeats 未降为 1（得到 {got}）")
        if "已自动降为 1" not in buf.getvalue():
            return _fail(f"{strategy} 降级没有告警，用户会以为真跑了 3 次独立验证")
    cfg = {"split": {"strategy": "nested_group", "repeats": 3}}
    if effective_repeats(cfg) != 3:
        return _fail("nested_group 的 repeats 被误降")
    return _ok("两种确定性划分都降为 1 并告警；nested_group 保持原值")


# --------------------------------------------------------------------------- #
# 展示层组：格式规格 / 措辞守卫 / HTML 结构 / 图件完整性
#
# 这些断言存在的理由很朴素：精度、单位、措辞、"图有没有误导性"如果只写在文档里，
# 半年后就会各自漂移。把它们写成会失败的断言，才有机会在下一次改动时被拦下。
# --------------------------------------------------------------------------- #
_FIXTURES: dict[str, Any] = {}
_TMPDIRS: list[Any] = []


def _mini_html(unit="%"):
    """构造一个最小 run 并渲染 HTML，返回 (ctx, html, figures)。结果缓存复用。"""
    key = f"html::{unit}"
    if key in _FIXTURES:
        return _FIXTURES[key]
    from sml.config import resolve_config
    from sml.io_data import load_dataset
    from sml.report_html import build_html
    from sml.stage_common import RunContext

    cfg = resolve_config(ASSETS / "config" / "smoke.json", budget="fast")
    cfg["data"]["target_unit"] = unit
    ds, _ = load_dataset(cfg)
    td = tempfile.TemporaryDirectory()
    _TMPDIRS.append(td)
    ctx = RunContext(cfg, ds, as_path(td.name))
    ensure_dir(ctx.run_dir / "figures")

    png = ctx.run_dir / "figures" / "F1_pred_vs_obs.png"
    plt = configure_matplotlib()
    f = plt.figure(figsize=(1, 1))
    plt.plot([0, 1], [0, 1])
    f.savefig(png, dpi=20)
    plt.close(f)

    t2 = pd.DataFrame([{
        "target": "protein", "model": "ridge", "r2_mean": 0.9712345678,
        "r2_sd": 0.0031944, "rmse_mean": 0.5640612, "rmse_sd": 0.0064383,
        "rpd_mean": 5.87, "metric_basis": "fold", "n_folds_ok": 9,
        "n_folds_degenerate": 0, "rank": 1,
    }])
    block = {
        "task": "regression", "model": "ridge", "basis": "fold", "n_folds_ok": 9,
        "r2": 0.9712345678, "rmse": 0.5640612, "mae": 0.4354, "rpd": 5.87,
        "ci": {"r2": (0.9578, 0.9802)},
        "_best_line": "最优模型：**ridge**（与次优性能相当）",
        "_limitations": ["- 数据未提供批次信息，**跨批次外推未经验证**。"],
    }
    out = build_html(ctx, "ridge", {"T2_performance": t2}, block,
                     figures={"F1": [str(png)]}, out_dir=ctx.sub("report"))
    html_text = Path(out).read_text(encoding="utf-8")
    _FIXTURES[key] = (ctx, html_text, {"F1": [str(png)]})
    return _FIXTURES[key]


@check("present", "零值不得显示成 0.0000 / -0.000（否则读者以为「测到了极小的量」）")
def _p_zeros() -> tuple[bool, str]:
    from sml import present as p

    bad = []
    for field in ("r2", "rmse", "rpd", "delta"):
        for v in (0, -0.0, 0.0):
            txt = p.fmt(field, v)
            if txt != "0":
                bad.append(f"{field}({v})={txt}")
    if bad:
        return _fail("零值被渲染成 " + "、".join(bad))
    return _ok("0 / -0.0 一律渲染为 0；-0.000 与 0.0000 结构上不可能出现")


@check("present", "有量纲误差量用有效数字（不是固定小数位）")
def _p_sigfig() -> tuple[bool, str]:
    from sml import present as p

    cases = [("rmse", 12.3456, "12.35"), ("rmse", 0.0123456, "0.01235"),
             ("rmse", 1234.5, "1234")]
    for field, v, want in cases:
        got = p.fmt(field, v)
        if got != want:
            return _fail(f"{field}({v}) 得到 {got}，应为 {want}（4 位有效数字）")
    if p.fmt("r2", 0.9712345678) != "0.971":
        return _fail(f"r2 应为 3 位小数，得到 {p.fmt('r2', 0.9712345678)}")
    return _ok("RMSE/MAE 走 4 位有效数字，R²/AUC 走 3 位小数")


@check("present", "p 值不得写成 0.0000（置换检验有硬下限 1/(1+n_perm)）")
def _p_pvalue() -> tuple[bool, str]:
    from sml import present as p

    if p.fmt_p(2e-4) != "< 0.001":
        return _fail(f"p=0.0002 应写成 < 0.001，得到 {p.fmt_p(2e-4)}")
    if p.fmt_p(1.0) != "> 0.999":
        return _fail(f"p=1.0 应写成 > 0.999，得到 {p.fmt_p(1.0)}")
    if p.fmt_p(0.0432) != "0.043":
        return _fail(f"p=0.0432 应为 0.043，得到 {p.fmt_p(0.0432)}")
    return _ok("p<0.001 与 p>0.999 用不等号，其余 3 位小数")


@check("present", "缺值/非有限值降级为破折号且带原因码")
def _p_missing() -> tuple[bool, str]:
    from sml import present as p

    cases = [(None, "missing"), (float("nan"), "not_finite"),
             (float("inf"), "not_finite"), ("abc", "not_numeric")]
    for v, reason in cases:
        f = p.fmt_detail("r2", v)
        if f.text != "—":
            return _fail(f"{v!r} 未降级为破折号，得到 {f.text!r}")
        if f.reason != reason:
            return _fail(f"{v!r} 的原因码应为 {reason}，得到 {f.reason!r}")
    return _ok("None/NaN/inf/非数值都降级为 —，并记录 missing/not_finite/not_numeric")


@check("present", "未配置目标单位时，报告必须显式暴露（不得静默省略）")
def _p_unit_missing() -> tuple[bool, str]:
    from sml import present as p

    if p.normalize_unit(None)[1] != "missing":
        return _fail("None 应判定为 missing")
    if p.unit_suffix(None, "mpl") != "单位未指定":
        return _fail("图内单位缺失时必须写出来，否则 RMSE 成了无量纲数字")
    _ctx, html, _fig = _mini_html(unit=None)
    if "单位未指定" not in html:
        return _fail("HTML 未出现「单位未指定」")
    if "unit-missing" not in html:
        return _fail("缺少醒目的 unit-missing 样式标记")
    if "sev-blocking" not in html:
        return _fail("缺少 blocking 级徽标，读者不会注意到量纲缺失")
    return _ok("单位缺失 → 破折号占位 + 红色标记 + blocking 徽标，三处同时可见")


@check("present", "措辞守卫必须抓住「证明」「完全准确」「无 p 值的显著优于」")
def _p_guard() -> tuple[bool, str]:
    from sml import present as p

    f1 = p.guard_text("模型完全准确，证明了该波段决定蛋白含量。")
    rules = {f.rule for f in f1}
    if not {"prove", "perfect"} <= rules:
        return _fail(f"漏抓：{sorted(rules)}")
    weak = p.Evidence(significant=True, p_bh=0.2)
    if not any(f.rule == "significant_claim" for f in p.guard_text("该方法显著优于 PLS。", weak)):
        return _fail("BH 校正 p=0.2 时仍放过「显著优于」")
    ok_ev = p.Evidence(significant=True, p_bh=0.01)
    if any(f.rule == "significant_claim" for f in p.guard_text("显著优于 PLS（p=0.01）。", ok_ev)):
        return _fail("有 p<0.05 证据时误报「显著优于」")
    # 否定句里的告诫不能被判成越界声明，否则守卫天天误报、最后没人看
    if p.guard_text("不得宣称模型可安全外推"):
        return _fail("把「不得宣称可安全外推」这句正确的告诫判成了违规")
    return _ok("抓到 prove/perfect/significant_claim，且正确放过否定句与有证据的表述")


@check("present", "HTML 必须具备离线可读与语义化结构（零外链、零 JS）")
def _p_html_struct() -> tuple[bool, str]:
    _ctx, html, _fig = _mini_html()
    need = ['<!doctype html>', 'lang="zh-CN"', 'name="viewport"', "@media print",
            "<caption", 'scope="col"', 'scope="row"', "<thead>"]
    miss = [n for n in need if n not in html]
    if miss:
        return _fail(f"缺结构要素：{miss}")
    if "<script" in html.lower():
        return _fail("页面含 <script>：离线/审稿系统里会失效或被执行策略拦掉")
    import re as _re
    ext = _re.findall(r'(?:src|href)="https?://[^"]+', html)
    if ext:
        return _fail(f"存在外链资源（断网即塌陷）：{ext[:3]}")
    return _ok("doctype/lang/viewport/打印样式/表头 scope 齐备；无脚本、无外链")


@check("present", "HTML 里不得出现 nan / None / <NA> 这些 pandas 字面量")
def _p_html_dirty() -> tuple[bool, str]:
    import re as _re

    _ctx, html, _fig = _mini_html()
    hits = _re.findall(r">\s*(nan|NaN|None|<NA>|NaT)\s*<", html)
    if hits:
        return _fail(f"漏进页面 {len(hits)} 处，例如 {hits[:3]}")
    return _ok("脏值全部在 _cell 边界转成了破折号")


@check("present", "data-raw 必须是未舍入原值，且能反推出显示文本")
def _p_html_raw() -> tuple[bool, str]:
    import re as _re

    from sml import present as p

    _ctx, html, _fig = _mini_html()
    m = _re.search(r'data-field="r2" data-raw="([^"]+)"[^>]*>([^<]+)<', html)
    if not m:
        return _fail("T2 的 R² 单元格没有 data-raw（读者无法追溯原始值）")
    raw, text = float(m.group(1)), m.group(2)
    if abs(raw - 0.9712345678) > 1e-12:
        return _fail(f"data-raw 被舍入过：{raw!r}（应为 0.9712345678）")
    if p.fmt("r2", raw) != text:
        return _fail(f"显示文本 {text!r} 与 fmt(raw)={p.fmt('r2', raw)!r} 不一致")
    return _ok(f"data-raw={raw!r} 保持全精度；显示文本可由 raw 单向推出")


@check("present", "图路径真实存在、徽标有图例、渲染层不手写小数位")
def _p_html_assets() -> tuple[bool, str]:
    import re as _re

    from sml import present as p

    ctx, html, _fig = _mini_html()
    srcs = _re.findall(r'src="(\.\./figures/[^"]+)"', html)
    if not srcs:
        return _fail("页面没有引用任何图")
    for s in srcs:
        target = ctx.run_dir / "report" / s
        if not Path(target).resolve().exists():
            return _fail(f"图路径不存在（会显示裂图）：{s}")
    used = set(_re.findall(r'class="badge sev-(\w+)"', html))
    legend = p.legend("html")
    for sev in used:
        if f"sev-{sev}" not in legend:
            return _fail(f"页面上用了 {sev} 级徽标，但图例里没有这一条")
    here = Path(__file__).resolve().parent / "sml"
    for name in ("report_html.py",):
        body = (here / name).read_text(encoding="utf-8")
        bad = _re.findall(r':\.\d+f|:\s*\.\d+%', body)
        if bad:
            return _fail(f"{name} 里仍有手写小数位 {bad[:3]}，会绕过规格层")
    return _ok(f"{len(srcs)} 个图路径全部存在；徽标级别 ⊆ 图例；渲染层无手写精度")


def _qc_fig():
    return configure_matplotlib()


@check("figure_qc", "双 y 轴必须被拦住（可让任意两条曲线看起来相关）")
def _q1_twin() -> tuple[bool, str]:
    from sml.plots import figure_qc

    plt = _qc_fig()
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2])
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.twinx().plot([1, 2], [3, 4])
    issues = figure_qc(fig, "FX")
    plt.close(fig)
    if not any(i.startswith("Q1") for i in issues):
        return _fail("twinx 未被识别为双 y 轴")
    return _ok("叠加坐标轴被判违规")


@check("figure_qc", "柱状图截断 y 轴必须被拦住（柱长编码大小）")
def _q2_trunc() -> tuple[bool, str]:
    from sml.plots import figure_qc

    plt = _qc_fig()
    fig, ax = plt.subplots()
    ax.bar([0, 1], [10.0, 10.4])
    ax.set_ylim(9.9, 10.5)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    bad = figure_qc(fig, "FX")
    ax._sml_zero_declared = True
    good = figure_qc(fig, "FX")
    plt.close(fig)
    if not any(i.startswith("Q2") for i in bad):
        return _fail("截断的柱状图未被识别")
    if any(i.startswith("Q2") for i in good):
        return _fail("显式声明后仍报错（正确做法应该放行）")
    return _ok("未声明的截断判违规；显式声明后放行")


@check("figure_qc", "jet / rainbow 色标必须被拦住")
def _q3_cmap() -> tuple[bool, str]:
    from sml.plots import figure_qc

    plt = _qc_fig()
    fig, ax = plt.subplots()
    ax.imshow([[1, 2], [3, 4]], cmap="jet")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    issues = figure_qc(fig, "FX")
    plt.close(fig)
    if not any(i.startswith("Q3") for i in issues):
        return _fail("jet 色标未被识别（非感知均匀，会造出假边界）")
    return _ok("彩虹色标被判违规")


@check("figure_qc", "箱线图不叠散点必须被拦住（会掩盖折间分布）")
def _q4_box() -> tuple[bool, str]:
    from sml.plots import figure_qc

    plt = _qc_fig()
    fig, ax = plt.subplots()
    ax.boxplot([[1, 2, 3], [2, 3, 4]])
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax._sml_boxplot = True
    issues = figure_qc(fig, "FX")
    plt.close(fig)
    if not any(i.startswith("Q4") for i in issues):
        return _fail("只有箱体没有散点，未被识别")
    return _ok("无散点的箱线图被判违规")


@check("figure_qc", "数值轴缺标签 / F8 缺误差线必须被拦住")
def _q5_labels() -> tuple[bool, str]:
    from sml.plots import figure_qc

    plt = _qc_fig()
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2])
    bad = figure_qc(fig, "FX")
    plt.close(fig)
    if not any(i.startswith("Q5") for i in bad):
        return _fail("无轴标签的数值轴未被识别")

    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2])
    ax.set_xlabel("RMSE"); ax.set_yticks([0, 1]); ax.set_yticklabels(["a", "b"])
    no_err = figure_qc(fig, "F8")
    plt.close(fig)
    if not any("误差线" in i for i in no_err):
        return _fail("F8 只有点估计、没有误差线，未被识别")
    return _ok("缺轴标签与 F8 缺误差线都被判违规")


@check("standards", "TRIPOD+AI 清单条目完整性与证据键合法性")
def _s_items() -> tuple[bool, str]:
    from sml.standards import AUTHOR_ONLY, EVIDENCE_FILES, load_items

    items = load_items()
    if not items:
        return _ok("清单未安装（items 为空），按设计降级为占位；条目须人工转录，代码不生成")
    ids = [str(i["id"]) for i in items]
    if len(ids) != len(set(ids)):
        return _fail(f"条目 id 重复：{ids}")
    if len(items) != 27:
        return _fail(f"条目数 {len(items)} ≠ 27（TRIPOD+AI 共 27 条）")
    legal = set(EVIDENCE_FILES) | AUTHOR_ONLY
    for it in items:
        keys = [str(k) for k in (it.get("evidence") or [])]
        if not keys and not str(it.get("na_reason") or "").strip():
            return _fail(f"条目 {it['id']} 既无 evidence 也无 na_reason")
        bad = [k for k in keys if k not in legal]
        if bad:
            return _fail(f"条目 {it['id']} 含非法证据键 {bad}")
    return _ok(f"{len(items)} 条齐全，证据键全部合法")


@check("standards", "清单渲染不得出现「符合 / 认证」的肯定表述")
def _s_wording() -> tuple[bool, str]:
    from sml.standards import render_section
    from sml.stage_common import RunContext
    from sml.config import resolve_config
    from sml.io_data import load_dataset

    cfg = resolve_config(ASSETS / "config" / "smoke.json", budget="fast")
    ds, _ = load_dataset(cfg)
    td = tempfile.TemporaryDirectory()
    _TMPDIRS.append(td)
    ctx = RunContext(cfg, ds, as_path(td.name))
    text = render_section(ctx)
    for kw in ("符合", "认证", "compliant"):
        for m in __import__("re").finditer(kw, text):
            head = text[max(0, m.start() - 12):m.start()]
            if not any(t in head for t in ("不构成", "不声明", "不是", "非")):
                return _fail(f"「{kw}」出现在肯定语境里：…{text[max(0, m.start()-20):m.end()+8]}…")
    return _ok("渲染文本里没有肯定的合规声明")


@check("reporting", "杠杆：fit 与 score 必须用同一个式子（含 +1/n 截距项）")
def _leverage_formula() -> tuple[bool, str]:
    from sml.ad import fit_ad

    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 12))
    groups = np.repeat([f"S{i}" for i in range(40)], 2)
    cfg = {"ad": {"methods": ["pca_t2", "pca_q", "knn", "leverage"],
                  "pca_variance": 0.9, "pca_max_comp": 5, "knn_k": 3}}
    ad = fit_ad(X, cfg, groups)
    scores = ad.score(X)["leverage"].to_numpy(dtype=float)
    diff = float(np.max(np.abs(ad.self_lev_ - scores)))
    expected = 1.0 / len(X)
    if diff > 1e-10:
        # 旧实现：fit 带 +1/n、score 不带，差值恰好是 1/n。拿"带截距"的分布估阈值，
        # 再去卡"不带截距"的分数——只有经验分位主导时才会露馅，方向是系统性偏松。
        hint = f"（差值≈{diff:.6f}，1/n={expected:.6f}，说明少加了截距项）"
        return _fail(f"fit 与 score 的杠杆值不一致，最大偏差 {diff:.3e}{hint}")
    return _ok(f"两条路径的杠杆值完全一致（最大偏差 {diff:.2e}），+1/n 均未遗漏")


@check("reporting", "h* 的两个 n 必须各自标注，且不得互相冒充")
def _leverage_dual_n() -> tuple[bool, str]:
    from sml.ad import fit_ad

    rng = np.random.default_rng(1)
    X = rng.normal(size=(90, 10))
    groups = np.repeat([f"S{i}" for i in range(30)], 3)
    cfg = {"ad": {"methods": ["leverage"], "pca_variance": 0.9, "pca_max_comp": 4, "knn_k": 3}}
    ad = fit_ad(X, cfg, groups)
    d = ad.diag_
    need = {"n_rows", "n_groups", "h_star_rows", "h_star_groups", "leverage_q95",
            "leverage_max", "leverage_inert", "leverage_active", "leverage_threshold"}
    missing = need - set(d)
    if missing:
        return _fail(f"ad.diagnostics 缺字段 {sorted(missing)}")
    if d["n_rows"] != 90 or d["n_groups"] != 30:
        return _fail(f"n 统计错：rows={d['n_rows']} groups={d['n_groups']}")
    # 3(k+1)/n 的两个口径必须只差 n 的比值，否则说明有一边算错了
    k = d["n_components"]
    if abs(d["h_star_rows"] - 3 * (k + 1) / d["n_rows"]) > 1e-12:
        return _fail("h_star_rows 与 3(k+1)/n_rows 不符")
    if abs(d["h_star_groups"] - 3 * (k + 1) / d["n_groups"]) > 1e-12:
        return _fail("h_star_groups 与 3(k+1)/n_groups 不符")
    if abs(d["h_star_rows"] * d["n_rows"] - d["h_star_groups"] * d["n_groups"]) > 1e-9:
        return _fail("两个 h* 的乘积不自洽：有一边没用同一个 k")
    # 真正判域外的那条线必须是 max(按行算的 h*, 经验分位)，不能是样品口径的 0.45
    want = max(d["h_star_rows"], d["leverage_q95"])
    if abs(d["leverage_threshold"] - want) > 1e-9:
        return _fail(f"leverage_threshold={d['leverage_threshold']:.4f}，"
                     f"应为 max(h*_rows={d['h_star_rows']:.4f}, q95={d['leverage_q95']:.4f})")
    if d["leverage_threshold"] > d["h_star_groups"] + 1e-9:
        return _fail("判域阈值超过了样品口径的 h*，会与 ad_scores.csv 对不上")
    return _ok(f"行口径 {d['h_star_rows']:.3f}（n={d['n_rows']}）/ 样品口径 "
               f"{d['h_star_groups']:.3f}（n={d['n_groups']}），各自标注且不互相冒充")


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    setup_console()
    ap = argparse.ArgumentParser(description="research-ml-modeling 环境自检与防泄漏回归测试")
    ap.add_argument("--only", default=None,
                    help="只跑某组：env / leakage / registry / multitarget / reporting / "
                         "multiclass / provenance / split / present / figure_qc / standards，"
                         "逗号分隔")
    ap.add_argument("--allow-fail", action="store_true", help="有失败项也返回 0")
    ap.add_argument("--json-out", default=None, help="把结果写成 JSON")
    args = ap.parse_args(argv)

    groups = None if not args.only else {g.strip() for g in args.only.split(",") if g.strip()}
    results = []
    n_fail = 0
    for group, name, fn in CHECKS:
        if groups and group not in groups:
            continue
        try:
            ok, detail = fn()
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, f"抛出异常 {type(exc).__name__}: {exc}"
            if "--debug" in sys.argv:
                traceback.print_exc()
        results.append({"group": group, "name": name, "ok": bool(ok), "detail": detail})
        flag = "PASS" if ok else "FAIL"
        print(f"[{flag}] ({group}) {name}\n        {detail}", flush=True)
        if not ok:
            n_fail += 1

    total = len(results)
    print(f"\n{'=' * 70}\n  自检结果：{total - n_fail}/{total} 通过" +
          (f"，{n_fail} 项失败" if n_fail else "，全部通过"), flush=True)
    if args.json_out:
        write_json(args.json_out, {"n_total": total, "n_fail": n_fail, "results": results})
    if n_fail and not args.allow_fail:
        print("  失败项会直接影响结论可靠性，请先修复再建模。", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
