# Copyright 2026 The research-ml-modeling Authors
# SPDX-License-Identifier: Apache-2.0
"""配置加载：JSON（本机无 PyYAML）、默认值深合并、--set 点号覆盖、字段校验。

设计要点
--------
* 配置只支持 JSON。需要注释用保留键 ``_comment``，解析后所有 ``_`` 开头的键被剔除。
* ``--set a.b.0.c=1`` 支持点号路径、列表下标、自动类型推断。
* 校验失败抛 ``ConfigError``，错误码形如 ``E-CFG-014``，便于在 references/config-schema.md 中索引。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .utils import as_path, read_json, write_json

__all__ = [
    "ConfigError",
    "DEFAULTS",
    "BUDGETS",
    "load_config",
    "resolve_config",
    "apply_overrides",
    "apply_removals",
    "validate_config",
    "config_defaults_doc",
    "stage_list",
]

SCHEMA_ID = "sml-config/1"

DEFAULTS: dict[str, Any] = {
    "_schema": SCHEMA_ID,
    "task": {
        "type": "regression",          # regression | classification
        "target": ["y"],
        "positive_label": None,        # 分类阳性类；None 表示取 label 的第二个值（排序后）
        "label_order": None,           # 显式给定分类标签顺序，如 ["正常", "失效"]
        "higher_is_better": None,      # 主指标方向，None 自动推断
    },
    "data": {
        "format": "wide",              # wide | long
        "path": None,
        "sheet": 0,
        "encoding": "utf-8",
        "sep": None,                   # 仅 CSV；None 自动嗅探
        "id_col": None,                # sample_id，分组键（防泄漏核心）
        "batch_col": None,             # 批次/仪器/日期，做留一批次外部验证
        "replicate_col": None,         # 同一样品第几次测量
        "feature_mode": "all_numeric_except_roles",
        "feature_cols": None,          # 显式列出时优先使用
        "drop_cols": [],
        "long_cols": {"id": None, "x": None, "y": None},
        "wavelength_axis": None,       # {start, stop, step, unit} 仅用于画图与峰映射
        # 目标的物理单位（"%"、"mg/kg"、"mV"…）。不填时报告与图上会显式写"单位未指定"，
        # 而不是静默省掉——没有量纲的 RMSE 读者无法判断模型能不能用。
        # 支持 {目标列名: 单位} 的字典写法；多目标时缺 key 即视为未指定，不回退到别的目标。
        "target_unit": None,
        "target_resolution": None,     # 目标的测量分辨率（同单位），用于说明 RMSE 是否低于分辨率
        "na_values": ["", "NA", "N/A", "nan", "NaN", "null", "-", "--"],
        "min_valid_frac": 0.5,         # 特征列非缺失比例低于此值 -> 丢弃
    },
    "split": {
        "group_col": None,             # None -> 用 data.id_col
        "strategy": "nested_group",    # nested_group | loo_batch | holdout_batch | holdout_random
        "outer": {"n_splits": 5, "shuffle": True},
        "inner": {"n_splits": 3, "shuffle": True},
        "repeats": 5,
        "stratify": True,              # 分类任务且每组单类时自动生效
        "holdout_batch": None,
        "test_size": 0.2,
    },
    "preprocess": {
        "candidates": [{"name": "raw", "steps": []}],
        "scale": True,
        "scale_with_mean": False,
    },
    "models": [
        {"name": "pls", "kind": "pls", "search_type": "grid",
         "search": {"n_components": [3, 5, 8, 10, 12, 15]}},
    ],
    "evaluation": {
        "metrics": None,               # None -> 按任务类型取默认
        "bootstrap": {"n": 2000, "alpha": 0.05, "mode": "percentile"},
    },
    "permutation": {
        "enabled": True,
        "n_perm": 200,
        "mode": "fast",                # fast（固定超参）| full（完整嵌套）
        "alpha": 0.05,
    },
    "ad": {
        "enabled": True,
        "methods": ["pca_t2", "pca_q", "knn", "leverage"],
        "alpha": 0.05,
        "quantile": 0.95,
        "pca_variance": 0.95,
        "pca_max_comp": 10,
        "knn_k": 5,
        "combine": "any",              # any | majority | votes>=N
    },
    "interp": {
        "enabled": True,
        "perm_repeats": 10,
        "band_size": 25,
        "top_k": 20,
        "vip": True,
        "coef_stability": True,
        "shap": "auto",                # auto | on | off
    },
    "report": {
        "lang": ["zh", "en"],
        "dpi": 300,
        "formats": ["png", "pdf"],
        "title": None,
        "authors": None,
        "html": True,                  # 额外产出 report/report.html（单文件、零外链）
        "precision_profile": "paper",  # paper（期刊，默认）| compact（投屏/网页更紧凑）
        "strict_text": False,          # True 时措辞守卫命中 error 直接抛异常
    },
    "runtime": {
        "n_jobs": -1,
        "seed": 42,
        "mem_guard_gb": 0.0,
        "save_models": "last",         # none | last | all
        "save_final_model": True,      # stage_cv 末尾保存全量重训模型（供 predict.py）
        "verbose": True,
        "fail_fast": False,
    },
}

# budget 只调整"贵的旋钮"，不动模型与预处理定义
BUDGETS: dict[str, dict[str, Any]] = {
    "fast": {
        "split.repeats": 3,
        "split.inner.n_splits": 3,
        "permutation.n_perm": 100,
        "evaluation.bootstrap.n": 500,
        "runtime._tree_n_iter": 15,
        "interp.perm_repeats": 5,
    },
    "standard": {
        "split.repeats": 5,
        "split.inner.n_splits": 3,
        "permutation.n_perm": 200,
        "evaluation.bootstrap.n": 2000,
        "runtime._tree_n_iter": 30,
        "interp.perm_repeats": 10,
    },
    "thorough": {
        "split.repeats": 10,
        "split.inner.n_splits": 4,
        "permutation.n_perm": 1000,
        "evaluation.bootstrap.n": 5000,
        "runtime._tree_n_iter": 60,
        "interp.perm_repeats": 10,
    },
}

REG_METRICS = ["r2", "rmse", "mae", "bias", "rpd", "rpiq", "ccc", "slope"]
CLF_METRICS = ["auc", "balanced_accuracy", "f1_macro", "mcc", "brier", "sensitivity", "specificity"]

ALL_STAGES = ["audit", "cv", "interp", "perm", "ad", "report"]
# 置换检验是 SKILL.md「严谨性红线」里的必做项，默认必须包含。
# 想省时间请显式 --stages audit,cv,... 或 --budget fast，而不是靠默认值漏掉它。
DEFAULT_STAGES = list(ALL_STAGES)


class ConfigError(ValueError):
    """配置错误。message 里带错误码，便于定位。"""

    def __init__(self, code: str, msg: str):
        self.code = code
        super().__init__(f"[{code}] {msg}")


def config_conflicts(old: dict, new: dict) -> list[str]:
    """同一个 run 目录被两套"本质不同"的配置复用时，列出冲突项。

    为什么要拦：``prepare()`` 每次都会重写 ``config.resolved.json``。若用户先跑
    ``--set models=[pls,pcr]``，之后又直接 ``--config 预设 --out 同一目录`` 跑下游
    stage，配置会被覆盖成预设默认值——此时 cv/ 里躺着 pcr 的结果、配置里却写着
    ridge，报告与结果永久对不上账，而整个过程中没有任何报错。

    只比较"换了就意味着换了一个研究问题"的字段：任务与目标、数据来源、模型清单。
    ``repeats``/``n_perm``/``bootstrap`` 这类预算项不在其列（换预算是合法的重跑）。
    """
    out: list[str] = []
    o_task, n_task = old.get("task") or {}, new.get("task") or {}
    if o_task.get("type") != n_task.get("type"):
        out.append(f"task.type：{o_task.get('type')} → {n_task.get('type')}")
    if [str(x) for x in (o_task.get("target") or [])] != [str(x) for x in (n_task.get("target") or [])]:
        out.append(f"task.target：{o_task.get('target')} → {n_task.get('target')}")
    o_data, n_data = old.get("data") or {}, new.get("data") or {}
    if str(o_data.get("path")) != str(n_data.get("path")):
        out.append(f"data.path：{o_data.get('path')} → {n_data.get('path')}")
    o_models = [(m.get("name"), m.get("kind"), m.get("search"), m.get("fixed"))
                for m in (old.get("models") or [])]
    n_models = [(m.get("name"), m.get("kind"), m.get("search"), m.get("fixed"))
                for m in (new.get("models") or [])]
    if o_models != n_models:
        out.append(f"models：{[m[0] for m in o_models]} → {[m[0] for m in n_models]}")
    return out


# --------------------------------------------------------------------------- #
# 深合并
# --------------------------------------------------------------------------- #
def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in over.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _strip_private(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_private(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [_strip_private(v) for v in obj]
    return obj


# --------------------------------------------------------------------------- #
# --set 覆盖
# --------------------------------------------------------------------------- #
def _coerce(text: str) -> Any:
    low = text.strip().lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("null", "none", ""):
        return None
    try:
        if text.strip() and text.strip().lstrip("+-").isdigit():
            return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    if text.startswith(("[", "{")):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    # 字符串值允许写 JSON 风格的引号：--set runtime.seed_scheme="per_model"。
    # 不剥引号的话校验会收到 '"per_model"'，报错信息里多两层引号，很容易让人
    # 以为是取值写错了。只剥成对的同种引号，不碰单引号文本内部的引号。
    stripped = text.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in "\"'":
        inner = stripped[1:-1]
        if stripped[0] not in inner:
            return inner
    return text


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """应用 ``a.b.0.c=value`` 形式的覆盖。"""
    out = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ConfigError("E-CFG-001", f"--set 需要 key=value 形式，收到 {item!r}")
        key, _, raw = item.partition("=")
        path = [tok for tok in key.strip().split(".") if tok != ""]
        if not path:
            raise ConfigError("E-CFG-002", f"--set 键为空：{item!r}")
        cursor: Any = out
        for tok in path[:-1]:
            if isinstance(cursor, list):
                try:
                    cursor = cursor[int(tok)]
                except (ValueError, IndexError):
                    raise ConfigError("E-CFG-003", f"--set 路径越界：{item!r}") from None
            elif isinstance(cursor, dict):
                if tok not in cursor or not isinstance(cursor[tok], (dict, list)):
                    cursor[tok] = {}
                cursor = cursor[tok]
            else:
                raise ConfigError("E-CFG-004", f"--set 路径不可达：{item!r}")
        last = path[-1]
        value = _coerce(raw)
        if isinstance(cursor, list):
            try:
                cursor[int(last)] = value
            except (ValueError, IndexError):
                raise ConfigError("E-CFG-003", f"--set 列表下标越界：{item!r}") from None
        elif isinstance(cursor, dict):
            cursor[last] = value
        else:
            raise ConfigError("E-CFG-004", f"--set 路径不可达：{item!r}")
    return out


_MISSING = object()


def _get_path(obj: Any, key: str) -> Any:
    """按点号路径取值；不存在返回 ``_MISSING``（用于 budget 改写检测）。"""
    cur = obj
    for tok in str(key).split("."):
        if isinstance(cur, dict):
            if tok not in cur:
                return _MISSING
            cur = cur[tok]
        elif isinstance(cur, list):
            try:
                cur = cur[int(tok)]
            except (ValueError, IndexError):
                return _MISSING
        else:
            return _MISSING
    return cur


def apply_removals(cfg: dict, removals: list[str] | None) -> dict:
    """从配置里**移除**列表元素/字典键，而不必整串重写。

    典型用法（敏感性分析里去掉一个泄漏列）::

        --set-remove data.drop_cols=time
        --set-remove data.drop_cols=日期 --set-remove data.drop_cols=批次

    语义：
    * ``path=value``：若 ``path`` 指向列表，删除与 ``value`` 相等的元素；
      若指向字典，删除键 ``value``。
    * 只写 ``path``（无 ``=``）：列表清空、字典键删除。
    * 若 ``path`` 本身带列表下标（如 ``models.0.search...``），删除该下标元素。
    """
    out = copy.deepcopy(cfg)
    for item in removals or []:
        has_value = "=" in item
        key, _, raw = item.partition("=")
        path = [tok for tok in key.strip().split(".") if tok != ""]
        if not path:
            raise ConfigError("E-CFG-002", f"--set-remove 键为空：{item!r}")
        value = _coerce(raw) if has_value else None
        cursor: Any = out
        for tok in path[:-1]:
            if isinstance(cursor, list):
                try:
                    cursor = cursor[int(tok)]
                except (ValueError, IndexError):
                    raise ConfigError("E-CFG-003", f"--set-remove 路径越界：{item!r}") from None
            elif isinstance(cursor, dict):
                if tok not in cursor or not isinstance(cursor[tok], (dict, list)):
                    raise ConfigError("E-CFG-004", f"--set-remove 路径不可达：{item!r}")
                cursor = cursor[tok]
            else:
                raise ConfigError("E-CFG-004", f"--set-remove 路径不可达：{item!r}")

        last = path[-1]
        if isinstance(cursor, list):
            try:
                idx = int(last)
                cursor.pop(idx)
            except (ValueError, IndexError):
                raise ConfigError("E-CFG-003", f"--set-remove 列表下标越界：{item!r}") from None
            continue
        if not isinstance(cursor, dict):
            raise ConfigError("E-CFG-004", f"--set-remove 路径不可达：{item!r}")

        target = cursor.get(last, _MISSING)
        if not has_value:
            if isinstance(target, list):
                cursor[last] = []
            else:
                cursor.pop(last, None)
            continue
        if isinstance(target, list):
            cursor[last] = [v for v in target if not (v == value or str(v) == str(value))]
        elif isinstance(target, dict):
            cursor[last] = {k: v for k, v in target.items() if str(k) != str(value)}
        else:
            raise ConfigError(
                "E-CFG-004",
                f"--set-remove 只能作用于列表或字典：{item!r}（当前是 {type(target).__name__}）",
            )
    return out


def _budget_override_notes(raw: dict, before: dict, after: dict, final: dict,
                           budget: str) -> list[str]:
    """列出「配置文件里写了、但被 budget 静默改写」的键。

    只在用户**没有**用 --set 再改回去时提示：此时最终值确实等于 budget 值，
    用户很可能没意识到自己配置里的 repeats/n_perm 没生效。
    """
    notes: list[str] = []
    spec = BUDGETS.get(budget) or {}
    for key in spec:
        if key.startswith("runtime._"):
            continue
        raw_val = _get_path(raw, key)
        if raw_val is _MISSING:
            continue
        budget_val = _get_path(after, key)
        final_val = _get_path(final, key)
        if raw_val != budget_val and final_val == budget_val:
            notes.append(
                f"E-CFG-032 配置文件里的 {key}={raw_val!r} 被 --budget {budget} 改写为 "
                f"{budget_val!r}（最终生效 {final_val!r}）。"
                f"如需保留请改用 --set {key}={raw_val!r}，或去掉 --budget。"
            )
    raw_models = raw.get("models") or []
    final_models = final.get("models") or []
    if isinstance(raw_models, list):
        for i, rm in enumerate(raw_models):
            if not isinstance(rm, dict) or "n_iter" not in rm:
                continue
            if i >= len(final_models) or not isinstance(final_models[i], dict):
                continue
            after_models = after.get("models") or []
            after_i = after_models[i] if i < len(after_models) and isinstance(after_models[i], dict) else {}
            final_i = final_models[i]
            raw_iter = rm.get("n_iter")
            budget_iter = after_i.get("n_iter")
            final_iter = final_i.get("n_iter")
            if raw_iter != final_iter and final_iter == budget_iter:
                name = final_i.get("name") or final_i.get("kind") or f"#{i}"
                notes.append(
                    f"E-CFG-033 模型 {name} 的 n_iter={raw_iter!r} 被 --budget {budget} 降为 "
                    f"{final_iter!r}。如需保留请用 --set models.{i}.n_iter={raw_iter!r}。"
                )
    return notes


def _apply_budget(cfg: dict, budget: str) -> dict:
    if budget not in BUDGETS:
        raise ConfigError("E-CFG-005", f"未知 budget={budget!r}，可选 {sorted(BUDGETS)}")
    out = copy.deepcopy(cfg)
    tree_iter = BUDGETS[budget].get("runtime._tree_n_iter")
    for key, val in BUDGETS[budget].items():
        if key.startswith("runtime._"):
            continue
        out = apply_overrides(out, [f"{key}={json.dumps(val)}" if isinstance(val, str) else f"{key}={val}"])
    # 树模型的 n_iter 按 budget 缩放（用户在配置里显式写了更大值则保留较小者）
    for model in out.get("models", []):
        if model.get("search_type") == "random":
            base = model.get("n_iter", 30)
            model["n_iter"] = int(min(base, tree_iter)) if base else int(tree_iter)
            model["_n_iter_budget"] = tree_iter
    return out


# --------------------------------------------------------------------------- #
# 校验
# --------------------------------------------------------------------------- #
def validate_config(cfg: dict) -> list[str]:
    """返回 warning 列表；致命问题抛 ConfigError。"""
    warnings: list[str] = []

    if cfg.get("_schema") != SCHEMA_ID:
        warnings.append(f"E-CFG-010 配置缺少 _schema={SCHEMA_ID}，按当前版本解析")

    task = cfg["task"]["type"]
    if task not in ("regression", "classification"):
        raise ConfigError("E-CFG-011", f"task.type 必须是 regression / classification，收到 {task!r}")

    targets = cfg["task"]["target"]
    if not targets:
        raise ConfigError("E-CFG-012", "task.target 不能为空")
    if task == "regression" and len(targets) > 1:
        warnings.append("E-CFG-013 多目标回归当前按逐个目标建模（不共享特征），耗时按目标数倍增")

    models = cfg.get("models") or []
    if not models:
        raise ConfigError("E-CFG-014", "models 不能为空；科研建模必须至少包含 PLS 基线")
    if not any(m.get("kind") == "pls" for m in models):
        warnings.append("E-CFG-015 未配置 PLS 基线，管线会自动注入（化学计量学审稿标配）")

    strategy = cfg["split"]["strategy"]
    if strategy not in ("nested_group", "loo_batch", "holdout_batch", "holdout_random"):
        raise ConfigError("E-CFG-016", f"split.strategy 非法：{strategy!r}")
    if strategy in ("loo_batch", "holdout_batch") and not cfg["data"]["batch_col"]:
        raise ConfigError("E-CFG-017", "strategy 需要批次信息，但 data.batch_col 未设置")
    if strategy == "holdout_batch" and not cfg["split"]["holdout_batch"]:
        raise ConfigError("E-CFG-018", "strategy=holdout_batch 时必须指定 split.holdout_batch")

    # 目标单位：给了就必须是字符串或 {目标列名: 单位}。留空是合法的降级（报告会显式写
    # 「单位未指定」），但写错类型会一路传到图和表里变成 "1234" 这种没有量纲的数字。
    unit = cfg["data"].get("target_unit")
    if unit is not None and not isinstance(unit, (str, dict)):
        raise ConfigError(
            "E-CFG-043",
            f"data.target_unit 必须是字符串（如 \"mg/kg\"、\"%\"）或 {{目标列名: 单位}} 字典，"
            f"收到 {type(unit).__name__}：{unit!r}")
    if isinstance(unit, dict):
        given = {str(k) for k in unit}
        missing = [str(t) for t in targets if str(t) not in given]
        if missing:
            warnings.append(
                f"E-CFG-043 data.target_unit 缺少以下目标的单位：{missing}。"
                f"报告中它们会显示「单位未指定」，不会借用其它目标的单位")
    profile = str(cfg["report"].get("precision_profile", "paper"))
    if profile not in ("paper", "compact"):
        raise ConfigError(
            "E-CFG-044", f"report.precision_profile 只能是 paper / compact，收到 {profile!r}")

    # holdout_random 只产出一次随机划分（GroupShuffleSplit n_splits=1）。若 repeats>1，
    # 旧行为是"跑 N 个不同划分、把 N 个测试集的行混在一起算池化 OOF 指标"，于是：
    #   · 同一样品被重复计入池化指标（R²/RMSE 的分母不再是独立样品）；
    #   · n_folds_ok 从 1 变成 N，metric_basis 由 pooled 静默变成 fold，
    #     排名与报告跟着换口径 —— 用户以为"重复了 N 次更稳"，实际拿到的是另一个统计量。
    # 这类口径漂移不会报错，只会让人把数字引错，所以在这里直接拦住。
    # 注意与 loo_batch/holdout_batch 的区别：那两种划分是确定的，重复跑得到同一折，
    # 降为 1 只省算力、不改结果；holdout_random 重复跑会改结果，故必须由用户显式决定。
    if strategy == "holdout_random" and int(cfg["split"]["repeats"]) > 1:
        raise ConfigError(
            "E-CFG-042",
            f"split.strategy=holdout_random 只做一次随机划分，但 split.repeats="
            f"{cfg['split']['repeats']}。\n"
            f"  旧实现会跑 {cfg['split']['repeats']} 个不同划分，把它们的测试集行混在一起算池化\n"
            f"  OOF 指标（同一样品被重复计入），同时把 metric_basis 从 pooled 静默改成 fold。\n"
            f"  想要「复现别人论文里的单次 70/30 划分」→ 设 split.repeats=1；\n"
            f"  想要「多次随机划分的均值/分布」→ 外层循环改 runtime.seed 各跑一次，或用\n"
            f"  nested_group（外层分组 K 折 + 重复），不要用 repeats 硬凑。",
        )

    n_outer = int(cfg["split"]["outer"]["n_splits"])
    n_inner = int(cfg["split"]["inner"]["n_splits"])
    if n_outer < 2 or n_inner < 2:
        raise ConfigError("E-CFG-019", "outer/inner n_splits 必须 >= 2")

    cands = cfg["preprocess"]["candidates"]
    if not cands:
        raise ConfigError("E-CFG-020", "preprocess.candidates 不能为空（至少一条 raw）")
    if len(cands) > 5:
        warnings.append(f"E-CFG-021 预处理候选 {len(cands)} 条 > 5，组合爆炸且易过拟合，建议精简")

    for cand in cands:
        for step in cand.get("steps", []):
            op = step.get("op")
            if op not in ("sg", "snv", "msc", "baseline_poly", "airpls", "band", "normalize", "diff"):
                raise ConfigError("E-CFG-022", f"未知预处理算子 op={op!r}")
            if op == "sg":
                win = int(step.get("window", 11))
                if win % 2 == 0:
                    raise ConfigError(
                        "E-CFG-023",
                        f"Savitzky-Golay window 必须为奇数（scipy 1.18 会静默接受偶数），收到 {win}",
                    )
                if int(step.get("polyorder", 2)) >= win:
                    raise ConfigError("E-CFG-024", "Savitzky-Golay polyorder 必须 < window")

    if cfg["evaluation"]["metrics"] is None:
        cfg["evaluation"]["metrics"] = list(REG_METRICS if task == "regression" else CLF_METRICS)

    ad_methods = cfg["ad"]["methods"]
    known_ad = {"pca_t2", "pca_q", "knn", "leverage"}
    if set(ad_methods) - known_ad:
        raise ConfigError("E-CFG-025", f"未知适用域方法：{sorted(set(ad_methods) - known_ad)}")

    # 两个可选开关（缺省即现状，只有显式写出来才生效）
    scheme = str(cfg.get("runtime", {}).get("seed_scheme", "legacy")).lower()
    if scheme not in ("legacy", "per_model"):
        raise ConfigError(
            "E-CFG-040",
            f"runtime.seed_scheme 只能是 legacy / per_model，收到 {scheme!r}。\n"
            f"  - legacy（默认）：模型序数 ×1 参与 fold 种子，模型数 ≥10 时会与下一折 0 号撞种子；\n"
            f"  - per_model：模型序数乘到 100 上，彻底隔开，但候选抽样会变（旧结果不可直接对拍）。")
    if scheme == "per_model":
        warnings.append("E-CFG-040 已启用 seed_scheme=per_model：抽样序列与默认不同，"
                        "结果不能与 legacy 运行直接对拍")

    strat = cfg.get("preprocess", {}).get("stratify_sampling", False)
    if not isinstance(strat, bool):
        raise ConfigError("E-CFG-041",
                          f"preprocess.stratify_sampling 只能是 true/false，收到 {strat!r}")
    if strat:
        warnings.append("E-CFG-041 已启用 stratify_sampling：prep__which 按层均分 n_iter，"
                        "候选集合与默认不同（仅 random 搜索生效）")

    return warnings


# --------------------------------------------------------------------------- #
# 加载 / 解析
# --------------------------------------------------------------------------- #
def load_config(path: str | Path | None = None) -> dict:
    if path is None:
        cfg = copy.deepcopy(DEFAULTS)
    else:
        p = as_path(path)
        if not p.exists():
            raise ConfigError("E-CFG-006", f"配置文件不存在：{p}")
        try:
            raw = read_json(p)
        except json.JSONDecodeError as exc:
            raise ConfigError("E-CFG-007", f"JSON 解析失败（配置不支持 // 注释）：{exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError("E-CFG-008", "配置根节点必须是对象")
        cfg = _deep_merge(DEFAULTS, _strip_private(raw))
    return cfg


def resolve_config(
    path: str | Path | None = None,
    overrides: list[str] | None = None,
    budget: str | None = None,
    seed: int | None = None,
    data_path: str | None = None,
    out_dir: str | None = None,
    removals: list[str] | None = None,
) -> dict:
    """加载 -> 套 budget -> 套 --set / --set-remove -> 套命令行 -> 校验。返回生效配置。"""
    raw_cfg: dict = {}
    if path is not None:
        try:
            raw_cfg = _strip_private(read_json(as_path(path)))
        except (OSError, json.JSONDecodeError):
            raw_cfg = {}
    cfg = load_config(path)
    if path is not None and cfg["data"].get("path"):
        # 相对路径按配置文件所在目录解析，便于把整套 assets 目录搬来搬去
        raw = str(cfg["data"]["path"]).replace("\\", "/")
        p = Path(raw)
        if not p.is_absolute():
            cand = (as_path(path).parent / raw).resolve()
            cfg["data"]["path"] = str(cand)
    before_budget = copy.deepcopy(cfg)
    if budget:
        cfg = _apply_budget(cfg, budget)
    after_budget = copy.deepcopy(cfg)
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    if removals:
        cfg = apply_removals(cfg, removals)
    if seed is not None:
        cfg["runtime"]["seed"] = int(seed)
    if data_path:
        cfg["data"]["path"] = str(data_path)
    if out_dir:
        cfg["runtime"]["out"] = str(out_dir)
    cfg.setdefault("runtime", {})["budget"] = budget or "custom"
    warnings = validate_config(cfg)
    if budget and raw_cfg:
        warnings.extend(_budget_override_notes(raw_cfg, before_budget, after_budget, cfg, budget))
    cfg["_warnings"] = warnings
    return cfg


def stage_list(text: str | list[str] | None) -> list[str]:
    if text is None:
        return list(DEFAULT_STAGES)
    if isinstance(text, list):
        items = text
    else:
        items = [tok.strip() for tok in str(text).split(",") if tok.strip()]
    items = [it for it in items if it]
    unknown = [it for it in items if it not in ALL_STAGES]
    if unknown:
        raise ConfigError("E-CFG-026", f"未知 stage：{unknown}，可选 {ALL_STAGES}")
    # 保持固定执行顺序
    return [s for s in ALL_STAGES if s in items]


def config_defaults_doc() -> dict:
    return {k: v for k, v in DEFAULTS.items() if not str(k).startswith("_")}


def dump_resolved(cfg: dict, path: str | Path) -> Path:
    clean = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
    return write_json(path, clean)
