# 标准依据与合规边界

本文件说明：这套管线在"结果怎么呈现"上参照了什么、**没有**参照什么，以及为什么。

---

## 一、先说结论

| 层 | 依据 | 本管线的用法 | 措辞边界 |
|---|---|---|---|
| L1 报告规范 | TRIPOD+AI（Collins et al., BMJ 2024;385:e078378，27 条，取代 TRIPOD 2015） | `report/tripod_ai_checklist.md` 逐条给 ✓/⚠/N/A + 证据路径 | 可写"对照清单自查"。原文明确它**不规定建模方法**、也**不是质量评价工具**，不存在"认证"，**不得写"符合/通过认证"** |
| L2 方法与验证标准 | ASTM E1655（红外多元定量）、ISO 12099 / 21543（饲料/乳品 NIR）、ICH Q2(R2)（2023-11-01 采纳）+ Q14、USP <1032> | 仅在报告页脚列为"方法学实践参考" | **只写"参考但不声明符合"**。见 §三 |
| L3 计量与数值表述 | GUM（JCGM 100:2008）、EURACHEM CG4、ASTM E29（有效数字）；**ICH Q2(R2) §3.3.2.4** | `sml/present.py` 的精度注册表、"CI 必须随点估计同报"的依据 | 可直接执行，见 §四 |
| L4 视觉设计 | Tufte（data-ink）、Cleveland & McGill（图形感知排序）、Okabe-Ito 色板、Crameri et al. 2020（Nature Comms，停用 jet/rainbow）、Rougier et al. 2014（PLOS CB）、WCAG 2.1 对比度 | `plots.py` 的配色与字号、`figure_qc` 的判定、HTML 的对比度 | 目标函数是"传递效率"，不是"好看" |

> 历史说明：1.1.x 及更早版本的仓库里，上述标准编号的检索命中为 **0**；`ICH/USP/CLSI/GB`
> 只出现在 `NOTICE` 与 `SKILL.md` 的免责声明中。1.2.0 起才有本文件与 L1 的清单映射。

---

## 二、L1：TRIPOD+AI 怎么接（以及为什么代码里没有条目文本）

- 条目文本放在 `assets/standards/tripod_ai_2024.json` 的 `items` 数组，**由人工转录**。
  当前是空骨架，渲染为"清单未安装"的占位说明。
- 代码（`sml/standards.py`）只做三件事：把条目的 `evidence` 键映射到本次 run 的产物、
  判定 ✓/⚠/N/A、渲染表格。**不生成、不翻译、不猜测条目内容**——凭记忆写 27 条标准文本
  等于伪造出处。
- 转录要求：
  1. 对照 BMJ 原文与其补充材料逐条抄录 `id`、`topic_zh`、`topic_en`、`applies_to`；
  2. 先核对 BMJ 的许可条款与署名要求（TRIPOD 系列历来开放获取，仍需确认具体版本）；
  3. 每条给 `evidence`（见下表），管线无法提供的用 `AUTHOR_ONLY` 键；
  4. 确实不适用的（如非临床场景下的"患者参与"）写 `na_reason`，状态显示 N/A；
     不写 `na_reason` 会按"需作者补"处理——**默认从严**。

### 证据键 → 产物

| 证据键 | 对应文件 | 说明 |
|---|---|---|
| `audit` | `audit/audit.json` | 数据审计（泄漏、共线、分组结构） |
| `protocol` | `cv/cv_summary.json` | 验证方案、折数、失败模型 |
| `models` | `cv/models.json` | 生效模型清单与搜索空间 |
| `performance` | `cv/metrics_summary.csv` | 性能汇总（含 `metric_basis`） |
| `ci` | `cv/metrics_ci.csv` | 按样品 bootstrap 的 95% CI |
| `comparison` | `cv/comparison.csv` | 两两配对比较 + BH 校正 |
| `oof` | `cv/oof_predictions.csv` | 逐行折外预测（唯一真源） |
| `figures` | `figures/` | F1–F8 |
| `perm` | `perm/permutation.json` | Y-randomization |
| `ad` | `ad/ad_eval.json` | 适用域与有效性检验 |
| `interp` | `interp/interp.json` | 重要性与系数稳定性 |
| `provenance` | `provenance.json` | 三哈希、版本、种子 |
| `model_card` | `model/model_card.json` | 最终模型卡 |
| `instrument` `sampling` `fairness` `ppi` `registration` `ethics` | — | **仅作者可提供**，恒显示 ⚠ |

---

## 三、L2：为什么"不声明符合"

| 标准 | 声称符合需要什么 | 本管线有没有 |
|---|---|---|
| ASTM E1655 | 明确的 reference method、其误差不得劣于待验证方法、规定的样本选择与验证集设计 | 无（只见 `data.path`，不知参考方法） |
| ISO 12099 / 21543 | 特定基质（饲料/乳品）的样品集与验证程序 | 无（通用管线，不绑定基质） |
| ICH Q2(R2) §2.5 | **独立验证集**，并以 RMSEP 与 RMSEC 的可比性论证准确度；可接受标准（ATP）须在建模前定义 | 无独立验证集（只有交叉验证的折外预测） |
| USP <1032> / <1033> | 方法生命周期与多变量模型验证的完整证据链 | 无 |

因此页脚固定写：

> 方法学实践参考 ASTM E1655、ISO 12099/21543 与 ICH Q2(R2)，但本管线未建立独立验证集与
> reference method，**不声明符合**上述标准；亦未针对监管申报、临床诊断或生产实时控制做过验证。

---

## 四、L3：可直接执行的数值约束

| 约束 | 依据 | 代码位置 | 违反后果 |
|---|---|---|---|
| 点估计必须与不确定度同报 | ICH Q2(R2) §3.3.2.4 要求 SD、RSD 与 100(1−α)% CI | HTML 的 KPI 卡、T2 的 CI 列、`*_sd` 列 | 只有点值时挂 `no_ci` 徽标 |
| 精度不得超过测量分辨率 | ASTM E29；有效数字而非固定小数位 | `present.SPECS` 的 `Precision(mode="sigfig")` | `RMSE=0.0042` 不会被写成 `0.004` |
| 不得对已舍入值再运算 | 通用数值原则 | `fmt_detail` 拒绝 `Formatted`；差值走 `fmt_delta(raw, raw)` | `TypeError` |
| 零值不写成 `0.000` | 同上 | `_render` 的零值分支 | 自检 P1 失败 |
| 比例为 0–1，显示为百分数 | 同文档内一致性 | `fmt_ratio` 对 `|v|>1` 直接抛错 | 传 3.3 会报错而不是显示 330% |
| 阈值须在看到结果前定义 | Q2(R2) + Q14（ATP 概念） | 本管线不做自动阈值"优化"，所有判据写在 `references/` 里 | — |

---

## 五、L4：视觉规范落到哪

| 规范 | 落地 |
|---|---|
| 色盲友好 | `assets/style/palette.json`（Okabe-Ito 10 色），`plots.PALETTE` 实际读取 |
| 停用彩虹色标 | `figure_qc` Q3：`jet/rainbow/gist_rainbow/gist_ncar/hsv` 判违规 |
| 不用双 y 轴 | `figure_qc` Q1 |
| 柱状图不截断纵轴 | `figure_qc` Q2（需显式声明才放行） |
| 箱体必须叠散点 | `figure_qc` Q4 |
| 轴标签必须说明是什么量 | `figure_qc` Q5 + `present.axis_label`（带单位） |
| 对比度 ≥ 4.5:1 | HTML 正文 `#1a1a1a`/白 = 16.9:1；`#D55E00` 仅 3.9:1，只用于边框与图内元素，不用于小字 |
| 单栏 85 mm 可读 | `assets/style/paper.mplstyle`（8pt/7.5pt/去上右框/刻度向内） |
