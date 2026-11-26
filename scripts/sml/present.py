# Copyright 2026 The research-ml-modeling Authors
# SPDX-License-Identifier: Apache-2.0
"""展示规格层：字段标签、精度、单位、状态徽标、措辞守卫的唯一定义处。

为什么要有这一层
----------------
改造前，同一个 RMSE 会以三种精度出现在三个地方：``cv/metrics_summary.csv`` 是全精度、
``T2_performance.csv`` 恒 3 位小数、``report.md`` 正文又是 4 位。读者对不上账时无法判断
是笔误还是真算错了。把所有"人看的字符串"收进本模块后，精度、单位、符号方向只有一处定义。

三条硬约束（由代码保证，不靠自觉）
--------------------------------
1. **零值只输出 ``0``**：``-0.0`` 归一为 ``0``，因此 ``-0.000`` / ``0.0000`` 结构上不可能出现。
   旧实现里 ``-0.0004`` 显示成 ``-0.000``，读者会读成"没有偏差"，实际上是有偏差。
2. **禁止二次舍入**：``fmt`` 只接受原始数值；传入已格式化的 ``Formatted`` 直接 ``TypeError``。
   差值必须由 raw 计算（见 ``fmt_delta``），否则 ``round(round(x,3),3)`` 的方向由浮点误差决定。
3. **单位三套写法互斥**：md 用 ``cm^-1``、html 用 ``cm<sup>−1</sup>``、matplotlib 用 ``cm$^{-1}$``。
   混用就是"图内方块、网页正常"的成因，所以由 ``unit_suffix`` 断言拦住。

本模块**只依赖标准库**。``metrics`` / ``report`` / ``plots`` 都要 import 它；若它反过来依赖
numpy/pandas，就会形成 ``metrics → present → numpy`` 这类无谓耦合，纯函数也不好自检。
"""

from __future__ import annotations

import html as _html
import math
import re
from dataclasses import dataclass, replace
from typing import Any, Iterable, NamedTuple

__all__ = [
    "Precision", "FieldSpec", "SPECS", "LABELS_ZH", "LABELS_EN", "Formatted",
    "fmt", "fmt_detail", "fmt_legacy", "fmt_ci", "fmt_delta", "fmt_p", "fmt_ratio",
    "spec_for_column", "num_format", "xlsx_sheet",
    "normalize_unit", "unit_suffix", "axis_label",
    "Badge", "BADGES", "badges_for", "legend",
    "Evidence", "Finding", "guard_text", "evidence_for",
    "CHAPTERS", "MISSING",
]

MISSING = "—"


# --------------------------------------------------------------------------- #
# 精度
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Precision:
    """一种数值呈现规则。

    ``mode`` 的取值不是装饰：``sigfig``（有效数字）与 ``decimals``（小数位）对
    ``RMSE = 0.0042`` 这类值的结果差 10 倍量级，而 ``references/reviewer-checklist.md``
    的 H6 要求的是**有效数字**。有量纲的量必须用 ``sigfig``。
    """

    mode: str            # decimals | sigfig | pvalue | percent | integer | sci
    digits: int
    signed: bool = False

    def num_format(self) -> str | None:
        """Excel 数字格式。返回 None 表示 Excel 表达不了（有效数字）。"""
        if self.mode == "decimals":
            return "0." + "0" * self.digits if self.digits > 0 else "0"
        if self.mode == "pvalue":
            return "0." + "0" * max(1, self.digits)
        if self.mode == "percent":
            return "0." + "0" * max(1, self.digits) + "%"
        if self.mode == "integer":
            return "0"
        if self.mode == "sci":
            return "0." + "0" * max(1, self.digits) + "E+00"
        return None


_D3 = Precision("decimals", 3)
_D2 = Precision("decimals", 2)
_D4 = Precision("decimals", 4)
_S4 = Precision("sigfig", 4)
_INT = Precision("integer", 0)
_P = Precision("pvalue", 3)
_PCT = Precision("percent", 1)
_SCI = Precision("sci", 2)


# --------------------------------------------------------------------------- #
# 字段规格
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FieldSpec:
    """一个可展示字段的全部呈现约定。

    ``unit is None`` 表示"跟随目标变量的单位"；``unit == ""`` 表示显式无量纲。
    这两者不能合并：前者缺失时要显示"（单位未指定）"，后者应当什么都不显示。
    """

    id: str
    label_zh: str
    label_en: str
    kind: str = "metric"          # metric | p | count | ratio | ci | text
    direction: int = 0            # +1 越大越好 / -1 越小越好 / 0 中性
    precision: Precision = _D3
    unit: str | None = None
    carries_ci: bool = False
    note: str = ""


SPECS: dict[str, FieldSpec] = {}

_BIAS_NOTE = ("方向为「实测 − 预测」（metrics.bias）。Bland-Altman 图的偏差方向相反，"
              "见 ba_bias；两者不可混引。")


def _s(fid: str, zh: str, en: str, *, kind: str = "metric", direction: int = 0,
       prec: Precision = _D3, unit: str | None = "", ci: bool = False,
       note: str = "") -> None:
    """注册一个字段。``unit=""`` 是默认值（无量纲）；有量纲的量必须显式写 ``unit=None``
    表示"继承目标单位"——默认不继承，否则 R² 后面会跟一个「（单位未指定）」。"""
    SPECS[fid] = FieldSpec(fid, zh, en, kind, direction, prec, unit, ci, note)


# 回归
_s("r2", "R²", "R²", direction=1, ci=True, note="1 − SS_res/SS_tot；OOF 池化与折均值是两套口径")
_s("r2_train", "R²(训练)", "R² (train)", direction=1, note="只看训练集，不能用于评价泛化")
_s("rmse", "RMSE", "RMSE", direction=-1, prec=_S4, unit=None, ci=True, note="有量纲，必须带单位报")
_s("rmse_train", "RMSE(训练)", "RMSE (train)", direction=-1, prec=_S4, unit=None)
_s("mae", "MAE", "MAE", direction=-1, prec=_S4, unit=None, ci=True)
_s("bias", "Bias(ME)", "Bias (ME)", direction=0, prec=_S4, unit=None, ci=True, note=_BIAS_NOTE)
_s("rpd", "RPD", "RPD", direction=1, prec=_D2, ci=True,
   note="SD(y)/RMSE；依赖 y 的方差，必须与 RMSE 同报并说明样本范围")
_s("rpiq", "RPIQ", "RPIQ", direction=1, prec=_D2, note="IQR(y)/RMSE；y 偏斜时比 RPD 可靠")
_s("ccc", "CCC", "CCC", direction=1, ci=True, note="同时惩罚不相关与不一致")
_s("slope", "斜率", "slope", direction=0, ci=True, note="ŷ~y 回归斜率；<1 表示范围压缩")
_s("intercept", "截距", "intercept", direction=0, prec=_S4, unit=None)
# 分类
_s("auc", "AUROC", "AUROC", direction=1, ci=True)
_s("auprc", "AUPRC", "AUPRC", direction=1, ci=True)
_s("balanced_accuracy", "平衡准确率", "balanced accuracy", direction=1, ci=True)
_s("f1_macro", "F1(macro)", "F1 (macro)", direction=1)
_s("f1", "F1", "F1", direction=1)
_s("mcc", "MCC", "MCC", direction=1, ci=True)
_s("brier", "Brier", "Brier", direction=-1, ci=True, note="越小越好；需与基线 Brier 对比才有意义")
_s("log_loss", "LogLoss", "log loss", direction=-1)
_s("sensitivity", "灵敏度", "sensitivity", direction=1)
_s("specificity", "特异度", "specificity", direction=1)
_s("accuracy", "准确率", "accuracy", direction=1)
_s("ece", "ECE", "ECE", direction=-1, note="分箱估计，对箱数敏感；不要当成精确的概率误差")
_s("auprc_base", "AUPRC 基线", "AUPRC baseline", direction=1, ci=True,
   note="随机猜测的 AUPRC = 阳性率；不看基线就报 AUPRC 是误导")
# 校准（方向与 bias 相反，必须分开注册）
_s("ba_bias", "偏差(预测−实测)", "bias (pred − obs)", direction=0, prec=_S4, unit=None,
   note="Bland-Altman 的偏差方向：预测 − 实测。与 metrics.bias 符号相反")
_s("ba_loa_low", "LoA 下限", "LoA lower", direction=0, prec=_S4, unit=None)
_s("ba_loa_high", "LoA 上限", "LoA upper", direction=0, prec=_S4, unit=None)
_s("cal_slope", "校准斜率", "calibration slope", direction=0, prec=_D3)
_s("cal_intercept", "校准截距", "calibration intercept", direction=0, prec=_S4, unit=None)
_s("pearson_r", "Pearson r", "Pearson r", direction=1, unit="")
# 不确定度
_s("sd", "标准差", "SD", kind="ci", prec=_D4, note="跨 repeat 的折均值标准差；不是重复测量标准差")
_s("ci_low", "CI 下限", "CI lower", kind="ci", prec=_D4)
_s("ci_high", "CI 上限", "CI upper", kind="ci", prec=_D4)
_s("alpha", "α", "alpha", kind="p", prec=_D2, unit="")
_s("p_value", "p", "p", kind="p", prec=_P, note="置换检验 p；受 n_perm 下限约束（1/(1+n_perm)）")
_s("p_raw", "p(原始)", "p (raw)", kind="p", prec=_P)
_s("p_bh", "p(BH)", "p (BH)", kind="p", prec=_P, note="Benjamini–Hochberg 校正；判断显著性应以此为准")
_s("p_bonf", "p(Bonferroni)", "p (Bonferroni)", kind="p", prec=_P)
# 计数
_s("n", "n", "n", kind="count", prec=_INT, unit="")
_s("n_groups", "独立样品数", "independent specimens", kind="count", prec=_INT, unit="",
   note="论文里必须报这个，不是光谱条数")
_s("n_rows", "测量条数", "measurements", kind="count", prec=_INT, unit="")
_s("n_features", "特征维数", "features", kind="count", prec=_INT, unit="")
_s("n_folds_ok", "有效折数", "valid folds", kind="count", prec=_INT, unit="")
_s("n_folds_degenerate", "退化折数", "degenerate folds", kind="count", prec=_INT, unit="")
_s("n_perm", "置换次数", "permutations", kind="count", prec=_INT, unit="")
_s("n_boot", "bootstrap 次数", "bootstrap resamples", kind="count", prec=_INT, unit="")
_s("n_repeats", "重复次数", "repetitions", kind="count", prec=_INT, unit="")
_s("rank", "排名", "rank", kind="count", prec=_INT, unit="", note="在目标内按选模主指标（回归 R²、分类 AUC）排名，与「最优模型」同口径")
_s("n_inside", "域内样品", "in-domain", kind="count", prec=_INT, unit="")
_s("n_outside", "域外样品", "out-of-domain", kind="count", prec=_INT, unit="")
# 比例（存 0–1，显示 %）
_s("rate", "比例", "proportion", kind="ratio", prec=_PCT, unit="")
_s("rate_in", "域内比例", "in-domain share", kind="ratio", prec=_PCT, unit="")
_s("rate_out", "域外比例", "out-of-domain share", kind="ratio", prec=_PCT, unit="")
_s("miss_rate", "失败折比例", "failed-fold share", kind="ratio", prec=_PCT, unit="")
# 其他
_s("delta", "差值", "difference", direction=0, prec=_D4, unit=None,
   note="成对差值；精度不得超过被比较的两个量本身的精度（ASTM E29）")
# ΔR² 是 R² 的差值 → 无量纲。别让它落到 delta 的 unit=None（"作者没填"）分支，
# 否则页面上会给一个本来就没有量纲的量挂「单位未指定」徽标，反而误导读者。
_s("delta_r2", "ΔR²", "ΔR²", direction=0, prec=_D4, unit="",
   note="R² 的差值，无量纲；正号表示该特征/波段退出后 R² 下降（越正越重要）")
_s("threshold", "阈值", "threshold", direction=0, prec=_S4, unit=None)
_s("pred_range_ratio", "预测范围比", "prediction range ratio", direction=0, prec=_D2,
   note="预测值极差 / 训练集目标极差；>>1 说明外推爆炸，不是比例，不能当百分比显示")
_s("vip", "VIP", "VIP", direction=1, prec=_D2, unit="", note=">1 只是经验阈值，不等于显著")
_s("importance", "重要性", "importance", direction=1, prec=_S4, unit="")
_s("t2", "Hotelling T²", "Hotelling T²", direction=0, prec=_D2, unit="")
_s("q", "Q 残差(SPE)", "Q residual", direction=0, prec=_S4, unit="")
_s("knn_dist", "kNN 距离", "kNN distance", direction=0, prec=_S4, unit="")
_s("leverage", "杠杆值 h", "leverage", direction=0, prec=_D3, unit="")
_s("seconds", "耗时", "elapsed", kind="count", prec=_D2, unit="s")
_s("cond_number", "条件数", "condition number", prec=_SCI,
   note="标准化设计矩阵的条件数；>1e4 就说明共线严重，不要用未正则的线性模型下结论")

LABELS_ZH: dict[str, str] = {k: v.label_zh for k, v in SPECS.items()}
LABELS_EN: dict[str, str] = {k: v.label_en for k, v in SPECS.items()}


# --------------------------------------------------------------------------- #
# 数值格式化
# --------------------------------------------------------------------------- #
class Formatted(NamedTuple):
    """格式化结果。``str()`` 可直接用，同时把原始值留给 ``data-raw``。"""

    text: str
    raw: float | int | None
    field: str
    reason: str
    unit: str

    def __str__(self) -> str:  # 让 f"{fmt(...)}" 写起来跟字符串一样
        return self.text


def _coerce(value: Any) -> tuple[float | int | None, str]:
    """转成数值；失败时返回原因而不是抛错——报告要能"部分生成"。"""
    if value is None:
        return None, "missing"
    if isinstance(value, bool):
        return (1 if value else 0), ""
    if isinstance(value, int):
        return value, ""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None, "not_numeric"
    if not math.isfinite(f):
        return None, "not_finite"
    return f, ""


def _sig(v: float, digits: int) -> str:
    """按有效数字格式化。定点的 ``f"{v:.4f}"`` 会把 ``0.0042`` 写成 ``0.0042``（2 位有效），
    也会把 ``1234.5`` 写成 ``1234.5000``（假精度）。"""
    if v == 0:
        return "0"
    exp = math.floor(math.log10(abs(v)))
    dec = digits - 1 - exp
    if dec > 6:
        # 4 位有效数字作用在 0.0004 上会得到 -0.0004000 这种读不出量级的写法
        return f"{v:.{max(1, digits - 1)}e}"
    if dec >= 0:
        return f"{v:.{dec}f}"
    return f"{round(v, dec):.0f}"


def _decimals(v: float, digits: int) -> str:
    if v == 0:
        return "0"
    # 0.00004 在 3 位小数下会被写成 0.000，读者会以为"测到的是 0"。
    # 优先补位到 6 位小数，仍压成 0 才退科学计数法——表格里 3.00e-04 比 0.0003 更难读。
    if 0 < abs(v) < 10 ** (-digits):
        for extra in range(digits + 1, 7):
            text = f"{v:.{extra}f}"
            if float(text) != 0:
                return text
        return f"{v:.2e}"
    return f"{v:.{digits}f}"


def _p_text(v: float) -> str:
    # 写 0.000 会被读成"p 精确等于 0"，而置换检验的 p 有硬下限 1/(1+n_perm)。
    if v < 0.001:
        return "< 0.001"
    if v > 0.999:
        return "> 0.999"
    return f"{v:.3f}"


def _render(prec: Precision, v: float) -> str:
    if prec.mode == "integer":
        return str(int(round(v)))
    if prec.mode == "pvalue":
        return _p_text(v)
    if prec.mode == "percent":
        if abs(v) > 1.0000001:
            raise ValueError(
                f"比例必须以 0–1 存储，收到 {v!r}。"
                "传 3.3 这类百分数会被显示成 330.0%，且不会报任何错。")
        return f"{v * 100:.{prec.digits}f}%"
    if prec.mode == "sigfig":
        return _sig(v, prec.digits)
    if prec.mode == "sci":
        return f"{v:.{prec.digits}e}"
    return _decimals(v, prec.digits)


def _render_adaptive(prec: Precision, v: float) -> str:
    """按精度渲染；若结果被压成 ``0`` 而原值非零，补一位小数。

    这条是有意为之的折中：``0.030 ± 0.000`` 会让读者以为离散度为零，但把 sd 一律
    加一位又会给 ``0.003`` 写成 ``0.0030``（假精度）。只在"要丢信息"时才加位。
    """
    text = _render(prec, v)
    if text == "0" and v != 0 and prec.mode in ("decimals", "sigfig"):
        return _render(replace(prec, digits=prec.digits + 1), v)
    return text


def fmt_detail(field: str, value: Any, *, unit: str | None = None,
               digits: int | None = None) -> Formatted:
    """格式化一个字段，返回文本 + 原始值。

    ``value`` 必须是原始数值。若传入 ``Formatted`` 会 ``TypeError``——这是"禁止二次舍入"
    的执行点：差值要先用 ``fmt_delta`` 从 raw 算，不能拿两次显示值相减。
    """
    if isinstance(value, Formatted):
        raise TypeError(
            "fmt_detail 只接受原始数值。要对差值格式化请用 fmt_delta(a_raw, b_raw)，"
            "否则就是在已舍入的值上再舍入一次。")
    spec = SPECS.get(field)
    prec = spec.precision if spec is not None else _D4
    if digits is not None:
        prec = replace(prec, digits=int(digits))
    canon, _state = normalize_unit(unit if unit is not None else (spec.unit if spec else None))
    if isinstance(value, bool):
        return Formatted("是" if value else "否", int(value), field, "", canon)
    num, reason = _coerce(value)
    if num is None:
        return Formatted(MISSING, None, field, reason, canon)
    return Formatted(_render_adaptive(prec, float(num)), num, field, "", canon)


def fmt(field: str, value: Any, *, unit: str | None = None,
        digits: int | None = None) -> str:
    return fmt_detail(field, value, unit=unit, digits=digits).text


def fmt_p(value: Any) -> str:
    return fmt_detail("p_value", value).text


def fmt_ratio(value: Any) -> str:
    return fmt_detail("rate", value).text


def fmt_ci(field: str, lo: Any, hi: Any, *, unit: str | None = None) -> str:
    """区间。与点估计同精度，只有被压成 0 时才补位。

    不给 CI 一律加一位：``[0.950, 0.980]`` 是论文里的标准写法，写成
    ``[0.9500, 0.9801]`` 只是假精度。真正需要补位的是 ``[0.000, 0.004]`` 这种跨零区间。
    """
    spec = SPECS.get(field)
    prec = spec.precision if spec is not None else _D4
    a, _ra = _coerce(lo)
    b, _rb = _coerce(hi)
    if a is None or b is None:
        return MISSING
    a, b = float(a), float(b)
    if prec.mode == "sigfig":
        # 区间两端必须对齐小数位，否则 [0.9801, 1.234] 看起来像两个不同精度的数
        def _dec(v: float) -> int:
            if v == 0:
                return prec.digits
            return max(0, prec.digits - 1 - math.floor(math.log10(abs(v))))
        d = min(6, max(_dec(a), _dec(b)))
        return f"[{a:.{d}f}, {b:.{d}f}]"
    return f"[{_render_adaptive(prec, a)}, {_render_adaptive(prec, b)}]"


def fmt_delta(field: str, a: Any, b: Any, *, unit: str | None = None) -> str:
    """两个原始值之差。唯一允许算差值再格式化的入口。"""
    x, _ = _coerce(a)
    y, _ = _coerce(b)
    if x is None or y is None:
        return MISSING
    return fmt_detail(field, float(x) - float(y), unit=unit).text


def fmt_legacy(v: Any, digits: int = 4) -> str:
    """改造前的 ``report._fmt`` 行为，逐字节保持。

    保留它是为了让"接入规格层"这一步能做到**输出零变化**：先用它接管旧调用点，
    确认 report.md 与表格逐字节不变，再逐处换成带字段语义的 ``fmt``。
    """
    if v is None:
        return MISSING
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(f):
        return MISSING
    # 负零归一：``f"{-0.0004:.3f}"`` 得到 ``-0.000``。报告里出现「-0.000」既难看又会被
    # 读成"有负向效应"（bias / ΔR² 恰是最容易落到 -0.0004 的量）。
    # 注意必须在**字符串**上判：``-0.0004 != 0``，只在数值上判 ``f == 0`` 拦不住舍入后的负零。
    text = f"{f:.{digits}f}"
    if text.startswith("-") and not text.lstrip("-").strip("0."):
        text = text[1:]
    return text


# --------------------------------------------------------------------------- #
# 列名 → 规格
# --------------------------------------------------------------------------- #
_SUFFIXES: tuple[tuple[str, int], ...] = (
    ("_fold_mean", 0), ("_mean", 0), ("_sd", 0), ("_ci_low", 0), ("_ci_high", 0),
    ("_ci_lo", 0), ("_ci_hi", 0), ("_se", 0),
)


def spec_for_column(col: Any) -> FieldSpec | None:
    """把 ``r2_mean`` / ``rmse_sd`` / ``rpd_ci_low`` 这类列名映射到规格。"""
    name = str(col)
    # 先拦 ΔR² 系列：否则 "delta_mean" 会被下面的 _mean 后缀分支拆成 "delta"，
    # 拿到 unit=None（作者没填），页面上就给一个无量纲的差值挂了「单位未指定」。
    if name.lower().startswith("delta_"):
        return SPECS["delta_r2"]
    spec = SPECS.get(name)
    extra = 0
    if spec is None:
        for suf, add in _SUFFIXES:
            if name.endswith(suf):
                base = name[: -len(suf)]
                if base in SPECS:
                    spec, extra = SPECS[base], add
                    break
    if spec is None:
        spec = _spec_by_pattern(name)
    if spec is None:
        return None
    if extra:
        p = spec.precision
        spec = replace(spec, precision=replace(p, digits=p.digits + extra))
    return spec


def _spec_by_pattern(name: str) -> FieldSpec | None:
    low = name.lower()
    # ΔR² 系列（delta_mean / delta_sd / delta_min / delta_max）只来自置换重要性表，
    # 是 R² 的差值：无量纲，不能当"作者忘填单位"处理。
    if low.startswith("delta_"):
        return SPECS["delta_r2"]
    if low.startswith("p_") or low in {"p", "pvalue", "p_value"}:
        return SPECS["p_value"]
    if low in {"rank", "n", "n_train", "n_test"} or (
            low.startswith(("n_", "num_")) and "rate" not in low):
        return SPECS["n"]
    # 名字里带 ratio 不等于"比例"：pred_range_ratio 是比值，可能远大于 1，
    # 按百分比渲染会写成 320.0%，把外推爆炸说成"三成"。
    if low.endswith("_ratio"):
        return SPECS["pred_range_ratio"]
    if "rate" in low or "比例" in name or "占比" in name or "share" in low:
        return SPECS["rate"]
    # 中文列名（如 "域内RMSE"、"域外准确率"）里嵌着指标名，也要认出来，
    # 否则这几列会掉到"无规格"分支，按 4 位有效数字随手格式化。
    for token, fid in (("rmse", "rmse"), ("mae", "mae"), ("r²", "r2"), ("r2", "r2"),
                       ("auc", "auc"), ("准确率", "accuracy"), ("mcc", "mcc"),
                       ("ccc", "ccc"), ("rpd", "rpd"), ("rpiq", "rpiq")):
        if token in low:
            return SPECS[fid]
    if "ci_low" in low or "ci_lo" in low:
        return SPECS["ci_low"]
    if "ci_high" in low or "ci_hi" in low:
        return SPECS["ci_high"]
    if low.endswith("_sd"):
        return SPECS["sd"]
    return None


def num_format(spec: FieldSpec | None, series=None) -> str | None:
    """Excel 数字格式。``sigfig`` 模式 Excel 表达不了，按列内最大量级近似到 3–4 位有效。"""
    if spec is None:
        return None
    if spec.precision.mode != "sigfig":
        return spec.precision.num_format()
    if series is None:
        return None
    try:
        vals = [abs(float(v)) for v in series if v is not None and str(v) not in ("", "nan", "—")]
        mag = max(vals) if vals else 0.0
    except (TypeError, ValueError):
        return None
    if mag >= 100:
        return "0.0"
    if mag >= 10:
        return "0.00"
    if mag >= 1:
        return "0.000"
    return "0.0000"


def xlsx_sheet(writer, sheet_name: str, df, *, freeze: bool = True) -> None:
    """写一个工作表并加显示格式。

    只设 ``num_format``——单元格里存的仍是原始 float。Excel 的数字格式是显示层属性，
    不会改变值，所以补充材料里做二次统计时拿到的还是全精度。
    """
    name = str(sheet_name)[:31]
    df.to_excel(writer, sheet_name=name, index=False)
    book = writer.book
    sheet = writer.sheets[name]
    cache: dict[str, Any] = {}
    for j, col in enumerate(df.columns):
        spec = spec_for_column(col)
        nf = num_format(spec, df[col])
        cell_format = None
        if nf:
            if nf not in cache:
                cache[nf] = book.add_format({"num_format": nf})
            cell_format = cache[nf]
        head = max([len(str(col))] + [len(str(v)) for v in df[col].head(200) if v is not None])
        sheet.set_column(j, j, min(38, max(9, head + 2)), cell_format)
    if len(df.columns):
        sheet.set_row(0, None, book.add_format({"bold": True, "bottom": 1}))
        if len(df):
            sheet.autofilter(0, 0, len(df), len(df.columns) - 1)
    if freeze:
        sheet.freeze_panes(1, 0)


# --------------------------------------------------------------------------- #
# 单位
# --------------------------------------------------------------------------- #
_UNIT_ALIASES = {
    "cm-1": "cm^-1", "cm^-1": "cm^-1", "cm^{-1}": "cm^-1", "cm⁻¹": "cm^-1",
    "1/cm": "cm^-1", "cm–1": "cm^-1",
    "pct": "%", "百分数": "%", "百分比": "%",
}
_DIMENSIONLESS = {"-", "", "1", "无量纲", "none", "None", "NA", "na"}
_UNSPEC_MD = "（单位未指定）"
_UNSPEC_HTML = '<span class="unit-missing">（单位未指定）</span>'


def normalize_unit(unit: Any) -> tuple[str, str]:
    """返回 ``(规范写法, 状态)``，状态 ∈ specified / dimensionless / missing。"""
    if unit is None:
        return "", "missing"
    s = str(unit).strip()
    if s in _DIMENSIONLESS:
        # Python 的 None 已经在上一步返回 missing；这里的 "-"/"1" 是人明确写的"没有单位"
        return "", "dimensionless"
    return _UNIT_ALIASES.get(s, s), "specified"


def _assert_target_safe(text: str, target: str) -> None:
    bad = {"md": ("$", "<", "&"), "html": ("$",), "mpl": ("<sup>", "&")}.get(target, ())
    for token in bad:
        if token in text:
            raise ValueError(
                f"单位写法与目标载体不匹配：{target} 里出现了 {token!r}（{text!r}）。"
                "md/html/mpl 三套写法必须互斥，混用会出现图内方块或网页显示 $^{-1}$。")


def unit_suffix(unit: Any, target: str = "md") -> str:
    """单位后缀。``target`` ∈ md / html / mpl。

    三种载体对"未指定"的处理必须都可见：md/html 给醒目的占位，mpl 给"单位未指定"，
    否则图上就会静默出现一个没有量纲的 RMSE。
    """
    canon, state = normalize_unit(unit)
    if state == "missing":
        return {"md": _UNSPEC_MD, "html": _UNSPEC_HTML, "mpl": "单位未指定"}.get(
            target, _UNSPEC_MD)
    if state == "dimensionless":
        return ""
    if canon == "cm^-1":
        out = {"md": "cm^-1", "html": "cm<sup>−1</sup>", "mpl": "cm$^{-1}$"}.get(target, "cm^-1")
    else:
        out = _html.escape(canon, quote=False) if target == "html" else canon
    _assert_target_safe(out, target)
    return out


_AXIS = {
    "observed": ("实测值", "Observed"),
    "predicted": ("预测值", "Predicted"),
    "residual": ("残差", "Residual"),
    "wavenumber": ("波数", "Wavenumber"),
    "score": ("得分", "Score"),
    "importance": ("重要性", "Importance"),
    "count": ("样品数", "Samples"),
    "index": ("索引", "Index"),
}


def axis_label(kind: str, unit: Any = None, target: str = "md", lang: str = "zh") -> str:
    """轴标签。``unit=None`` 视作"未指定"并显式写出，不视作"不要后缀"。

    要表示无量纲请传 ``""``。这个区分很重要：``None`` 是"作者没填"，
    静默省略等于默认读者知道 RMSE 的量纲——他不可能知道。
    """
    zh, en = _AXIS.get(kind, (kind, kind))
    name = zh if lang == "zh" else en
    suffix = unit_suffix(unit, target)
    return f"{name} ({suffix})" if suffix else name


# --------------------------------------------------------------------------- #
# 状态徽标
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Badge:
    code: str
    mark: str
    label_zh: str
    severity: str      # info | caution | blocking
    explain: str


BADGES: tuple[Badge, ...] = (
    Badge("pooled", "†", "池化口径", "caution",
          "该行有效折不足，*_mean 只来自极少数折，不可引用；请引用同行的池化值"),
    Badge("degenerate_folds", "‡", "含退化折", "caution",
          "部分折的测试集过小或单类，折级指标不可比"),
    Badge("ad_ineffective", "⊘", "适用域无效", "caution",
          "域内外误差无显著差异，不得宣称模型可外推"),
    Badge("ad_outside", "○", "域外样品", "info",
          "该样品至少一种判定方法越界，预测可靠性未知"),
    Badge("pred_range_extreme", "⇱", "预测值越界", "caution",
          "预测值超出训练集目标量级，属于外推爆炸的前兆"),
    Badge("proba_non_native", "≈", "概率非原生", "caution",
          "概率由决策函数转换而来，Brier/ECE/LogLoss 不可与原生概率模型并列比较"),
    Badge("not_significant", "=", "差异不显著", "info",
          "与次优模型的配对 bootstrap 差值跨 0，只能写「性能相当」"),
    Badge("single_repeat", "¹", "单次重复", "caution",
          "重复次数为 1，无法给出重复间稳定性"),
    Badge("no_ci", "–", "无置信区间", "caution",
          "缺少 bootstrap CI，性能估计只有点值"),
    Badge("unit_missing", "?", "单位未指定", "blocking",
          "目标单位未配置，RMSE/MAE/bias 无物理量纲，读者无法判断可否使用"),
    Badge("failed_model", "×", "模型失败", "blocking",
          "该模型在交叉验证中整体失败，已从性能表中剔除，原因必须写进正文"),
    Badge("perfect_score", "!", "满分指标", "caution",
          "指标达到理论边界，先查泄漏与标签编码，不要先报喜"),
    Badge("baseline_absent", "b", "缺基线", "caution",
          "缺少 PLS 或随机基线，无法判断增益来自模型还是数据"),
)

_BY_CODE = {b.code: b for b in BADGES}


def badges_for(*, block: dict | None = None, perm: dict | None = None,
               ad_eval: dict | None = None, unit_known: bool = True,
               failed_models: Iterable[str] | None = None,
               has_baseline: bool = True) -> list[Badge]:
    """从结果块推导状态徽标。全项目只有这一处把状态映射成符号。"""
    out: list[Badge] = []
    block = block or {}
    if str(block.get("basis", "fold")) == "pooled":
        out.append(_BY_CODE["pooled"])
    if int(block.get("n_folds_degenerate") or 0) > 0:
        out.append(_BY_CODE["degenerate_folds"])
    sig = block.get("significance") or {}
    if sig.get("available") and not sig.get("significant"):
        out.append(_BY_CODE["not_significant"])
    if int(block.get("n_repeats") or 0) == 1:
        out.append(_BY_CODE["single_repeat"])
    if not block.get("ci"):
        out.append(_BY_CODE["no_ci"])
    if not unit_known:
        out.append(_BY_CODE["unit_missing"])
    if failed_models:
        out.append(_BY_CODE["failed_model"])
    if not has_baseline:
        out.append(_BY_CODE["baseline_absent"])
    per = (ad_eval or {}).get("per_model") or {}
    first = next(iter(per.values()), {})
    if first.get("ad_effective") is False:
        out.append(_BY_CODE["ad_ineffective"])
    # 置换不显著（perm["significant"] is False）不在这里打标：它已经由 limitations
    # 与审稿清单承担，再挂一个符号只会让同一件事在页面上出现三次。
    _ = perm
    return out


def legend(target: str = "md") -> str:
    """徽标图例。**渲染全部条目**——"某条没出现"本身也是信息。"""
    if target == "html":
        rows = "".join(
            f'<li><span class="badge sev-{b.severity}"><span class="mark">{b.mark}</span>'
            f'{_html.escape(b.label_zh)}</span> {_html.escape(b.explain)}</li>'
            for b in BADGES)
        return f'<ul class="badge-legend">{rows}</ul>'
    lines = ["| 标记 | 含义 | 说明 |", "|---|---|---|"]
    lines += [f"| {b.mark} | {b.label_zh} | {b.explain} |" for b in BADGES]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 措辞守卫
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Evidence:
    """判定一句话是否越界所依据的事实。缺省即"证据不足"，此时任何强断言都不成立。"""

    significant: bool | None = None
    p_bh: float | None = None
    ad_effective: bool | None = None
    has_batch: bool = False
    accuracy: float | None = None
    n_groups: int | None = None
    bias: float | None = None


@dataclass(frozen=True)
class Finding:
    rule: str
    level: str        # error | warn
    snippet: str
    advice: str


_NEGATIONS = ("不得", "不要", "不能", "禁止", "不应", "无法", "未能", "不可", "尚", "有待")


def _negated(text: str, start: int, window: int = 14) -> bool:
    """往前看一小段判断是否处于否定句里。

    没有这一步，"不得宣称模型可安全外推" 这句正确的告诫本身会被判成越界声明，
    守卫就会天天误报，最后所有人都学会忽略它。
    """
    head = text[max(0, start - window):start]
    return any(tok in head for tok in _NEGATIONS)


_RULES: list[tuple[str, str, re.Pattern, Any, str]] = [
    ("prove", "error", re.compile(r"证明|证实了|验证了模型"),
     lambda ev: False, "研究只能说「支持/与…一致」，不能说「证明了」"),
    ("perfect", "error", re.compile(r"完全准确|100\s*%\s*准确|零误差|完美预测"),
     lambda ev: (ev.accuracy is not None and ev.accuracy >= 1.0
                 and (ev.n_groups or 0) >= 30), "满分指标先查泄漏；未核验不得写入正文"),
    ("significant_claim", "error", re.compile(r"显著(优于|高于|好于|改善|提升)"),
     lambda ev: bool(ev.significant) and (ev.p_bh is not None and ev.p_bh < 0.05),
     "「显著」必须紧跟校正后的 p 值，否则删掉这个词"),
    ("best_claim", "warn", re.compile(r"最优|最佳模型|唯一最好"),
     lambda ev: bool(ev.significant), "无显著性支撑时写「表现最好」而非「最优」"),
    ("extrapolate", "error", re.compile(r"可(安全)?外推|可用于新样品|适合推广"),
     lambda ev: bool(ev.ad_effective) and bool(ev.has_batch),
     "适用域无效或无批次外部验证时，不得声明外推能力"),
    ("no_bias", "error", re.compile(r"无系统偏差|不存在系统误差|没有偏差"),
     lambda ev: ev.bias is not None and abs(ev.bias) < 1e-6, "报「无偏」必须给出 bias 与其 CI"),
    ("cause", "warn", re.compile(r"由于.{0,12}(导致|造成)|归因于|原因是"),
     lambda ev: False, "交叉验证给出相关性，不给因果；因果表述需实验设计支撑"),
]


def evidence_for(block: dict | None, ds=None, perm: dict | None = None,
                 ad_eval: dict | None = None) -> Evidence:
    """从结果块组装措辞判定所需的事实。全项目只此一处，避免两边各猜一套。"""
    block = block or {}
    sig = block.get("significance") or {}
    first = next(iter(((ad_eval or {}).get("per_model") or {}).values()), {})
    return Evidence(
        significant=sig.get("significant"),
        p_bh=sig.get("p_bh"),
        ad_effective=first.get("ad_effective"),
        has_batch=bool(getattr(ds, "batch", None) is not None),
        accuracy=block.get("accuracy"),
        n_groups=getattr(ds, "n_groups", None),
        bias=block.get("bias"),
    )


def guard_text(text: str, ev: Evidence | None = None, *, strict: bool = False) -> list[Finding]:
    """扫描文本里的越界表述。

    默认**不抛异常**：报告阶段抛错会让数值完全正确的 run 因为一句模板措辞整阶段失败，
    并在 report.md 已落盘处留下半成品。error 级 finding 交给调用方渲染成红色合规横幅，
    并写进 ``report/compliance.json``；``strict=True`` 时才抛，供自检使用。
    """
    ev = ev or Evidence()
    findings: list[Finding] = []
    for rule, level, pattern, ok, advice in _RULES:
        for m in pattern.finditer(text):
            if _negated(text, m.start()):
                continue
            if ok(ev):
                continue
            findings.append(Finding(rule, level, m.group(0), advice))
    if strict and any(f.level == "error" for f in findings):
        detail = "；".join(f"{f.rule}: {f.snippet}" for f in findings if f.level == "error")
        raise ValueError(f"文本包含数据不支持的强断言（{detail}）")
    return findings


# --------------------------------------------------------------------------- #
# 章节
# --------------------------------------------------------------------------- #
#: report.md 与 report.html 共用这一份标题，避免两版措辞漂移。
CHAPTERS: tuple[tuple[str, str], ...] = (
    ("一", "结论摘要"),
    ("二", "数据与任务"),
    ("三", "验证方案"),
    ("四", "性能结果"),
    ("五", "模型比较与不确定度"),
    ("六", "是否在拟合噪声"),
    ("七", "适用域"),
    ("八", "可解释性"),
    ("九", "局限（必须写入论文）"),
    ("十", "复现信息"),
)


def chapter_heading(i: int) -> str:
    num, title = CHAPTERS[i]
    return f"## {num}、{title}"
