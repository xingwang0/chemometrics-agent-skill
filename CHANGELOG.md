# CHANGELOG

格式参照 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)；版本号遵循语义化版本。

## [1.1.1] - 2026-11-25

这一版只修缺陷，不改任何方法学口径。核心原则：**把"静默改变结果"改成"显式失败"**。
升级后旧配置可能报错，每条报错都写了改法。

### 修复（会改变行为）

- **`holdout_random` + `split.repeats>1` 现在报 `E-CFG-042`**（原为静默）
  旧实现会跑 N 个不同划分，把 N 个测试集的行混在一起算池化 OOF 指标（同一样品被重复
  计入），同时让 `n_folds_ok` 从 1 变 N、`metric_basis` 由 `pooled` 悄悄改成 `fold`。
  复现"别人论文的单次 70/30 划分"必须 `repeats=1`。
- **`models[].fixed` 里拼错参数名现在报 `E-MOD-004`**（原为 `warn` 后继续）
  旧行为是"警告一句、照常用默认值训练"，导致用户明确想钉死的超参（如 `subsample=0.8`）
  静默失效。`search` 早就有白名单校验，`fixed` 现在用同一套（含 catboost 盲区例外表）。
- **`split.group_col` 现在真正生效**（原为死字段）
  之前它只用于报告/审计里拼显示字符串，真正的分组键永远是 `data.id_col`——用户写了
  `group_col` 却看到 CV 按 `id_col` 分组，且没有任何提示。现在：给了就用（列不存在报
  `E-IO-024`），并且**自动从特征矩阵剔除**（否则等于把样品身份当预测依据）。
- **`data.feature_mode` 现在做取值校验**（原为死字段）
  只实现 `all_numeric_except_roles` 一种语义，写别的值报 `E-IO-025`，不再让它安静地
  什么都不做。要精确控制特征请用 `feature_cols` / `drop_cols`。

### 新增

- **预测值越界检查（外推爆炸报警）**
  逐折比对预测值与训练集目标极差，越界超过 2 倍即判定为量纲跑飞
  （阈值定为 2 倍而非 5 倍：健康数据的越界量级实测 <0.1 倍，而真实故障案例
  ——真实 153 mV 预测成 2431 mV——比值是 4.6 倍，5 倍阈值恰好会漏掉它）。
  产物：`fold_results.csv` 新增 `pred_range_ratio`、`pred_n_extreme` 两列；
  `cv_summary.json` 新增 `prediction_range_flags`；有越界时控制台 warn、
  审稿清单出现"预测值有没有跑出目标量纲？"一行。
  动机：实测 PLS 把真实 153/176 mV 的样品预测成 2431/2457 mV，单折 R² 掉到 −37.8，
  旧版只是安静地显示 −37.8，读者会误读成"这个模型很差"而不是"数值爆炸了"。
- **VIP 可靠性标记**
  `vip.csv` 新增 `pls_cv_r2`、`vip_reliable` 两列；PLS 的 CV R² < 0.10 时
  `vip_reliable=false`，`interp.json` 写入原因，报告与审稿清单自动改 ⚠ 并建议
  改用置换重要性 / SHAP。
  动机：PLS 失效时 VIP 排序仍是"看起来很专业"的噪声投影，且常与置换重要性冲突。

### 修正

- `--dry-run` 对 `holdout_random` 谎报折数（打印"外层 5 折"、实际 1 折）；
  `stage_cv` 的启动日志同错。现在按实际执行数显示，并注明"无重复"。
- `SKILL.md` 的 `version: 1.1.0` 与 `sml.__version__ = "1.0.0"` 不一致；
  现在统一为 `1.1.1`（`provenance.json` 记录的值随之正确）。
- 配置期校验 `fixed` 时会先打一句"已忽略"再报错，自相矛盾；
  `build_pipeline` 增加 `apply_fixed=False` 供校验专用。

### 文档

- `SKILL.md` 新增「v1.1.1 行为变更」一节（升级前必读）。
- `references/config-schema.md`：补 `holdout_random` 语义、四种划分对照表、
  `group_col` / `feature_mode` 的真实行为，新增 `E-CFG-042` / `E-IO-024` /
  `E-IO-025` / `E-MOD-004` 错误码。
- `references/interpretation.md`：补 VIP 的"来源模型必须有预测力"前提与两列含义。

### 验证

- `selfcheck.py` 34/34 通过。
- 定点用例 11 项：`holdout_random` 拒绝多重复、`group_col` 生效且不进特征、
  `feature_mode` 非法值报错、`fixed` 拼错报错/写对放行、dry-run 折数、
  越界判据的数学边界（含训练集目标为常数时不除零）。
- 端到端 9 项：有界目标 + 域外测试批次触发越界告警（99.4 倍极差）、
  纯噪声目标触发 VIP 不可引用（PLS R²=−0.270）并贯通到审稿清单。

## [1.1.0] - 2026-11-22

- 首次开源：六阶段管线（audit / cv / interp / perm / ad / report），
  嵌套分组重复 CV、按组 bootstrap CI、Y-randomization、四法适用域、
  8 张图 + 6 张表 + Methods 草稿 + 57 条审稿清单。
- 可选开关：`runtime.seed_scheme`（模型数 ≥10 时避免跨折撞种子）、
  `preprocess.stratify_sampling`（random 搜索按预处理层均分候选）。
