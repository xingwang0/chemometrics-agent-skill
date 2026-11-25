# CHANGELOG

格式参照 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)；版本号遵循语义化版本。

## [1.2.1] - 2026-11-25

修 1.2.0 遗留的杠杆阈值口径问题。**一个 run 里曾同时出现两个 h\*（0.15 与 0.45），
相差恰好等于每样品重复次数，且都不报错。**

### 修复

- **`fit` 与 `score` 的杠杆公式不一致**（真 bug）
  `fit` 算 `h = tᵀ(TᵀT)⁻¹t + 1/n`，`score` 算 `tᵀ(TᵀT)⁻¹t`——差一个截距项 `1/n`。
  而阈值 `max(3(k+1)/n, 95% 分位)` 是在**带** `1/n` 的分布上估的，却拿去卡**不带**的分数。
  由 `3(k+1)/n` 主导时看不出来，一旦经验分位主导就是系统性偏松（差 1/n）。
  现在两条路径共用 `ad._leverage()`，自检断言两者偏差 < 1e-10。
  影响：`ad_scores.csv` 的 `leverage` 列整体抬高 `1/n`（本次 180 行 ⇒ +0.00556）；
  `ad_votes` / `ad_flag` / `ad_majority` 与 1.2.0 逐行相同（该数据上杠杆法本来零触发）。

- **`ad_reliability.leverage_threshold` 名不副实**（表述缺陷）
  它其实是 `3(k+1)/n_groups`，只用于"独立样品够不够撑起经验分位"的判断，**从不参与判域**；
  但名字与 `ad_scores.csv` 的 `leverage_thr`（`3(k+1)/n_rows`）一样，两个数并排放着，
  读者只能认为其中一个算错了。现在拆成：
  `leverage_threshold`（= 判域那条线，与 `ad_scores.csv` 同值）、
  `leverage_threshold_groups`（仅可靠性判断用），`n_rows` / `n_groups` 各自随行标注，
  并附 `rule` 说明。`ad.diagnostics` 同步给出 `h_star_rows` / `h_star_groups` / `leverage_q95` /
  `leverage_max` / `leverage_inert` / `leverage_active`。

- **阈值来源不可见**（透明度）
  原来只输出一个数，看不出它来自 `3(k+1)/n` 还是 95% 分位。现在 `leverage_q95` 与
  `leverage_rule` 一并落盘。

- **"永不触发"没有任何提示**（静默失效）
  `h* ≥ 1` 才判退化，但 `h* = 0.15` 而数据最大杠杆只有 0.1254 时，杠杆法一次也不会触发，
  却仍计入多数票分母（4 法时"过半"需要 3 票而不是 2 票）。现在输出
  `leverage_inert` + `leverage_max`，日志给一行说明。**不改投票规则**——那会改变
  `ad_majority` 的结果，属于方法学变更，留待裁决。

- **F4 加注 `n` 与投票状态**：`h* = 0.150（n=180 行）`；若本次 `ad.methods` 里没有
  leverage，标题注明"leverage 未参与本次判定，h\* 仅为参考线"。

### 未改（需裁决）

- 杠杆法"零触发"是否应退出多数票分母（影响 `ad_majority`，会让域外样品变多）。
- `95% 分位` 在**行**上估计，重复测量等于给样品加权；是否改为按样品聚合后再估分位
  （会让 5% 变成样品级假阳性率，同样影响判定结果）。

## [1.2.0] - 2026-11-25

新增"结果展示层"。核心原则：**只改怎么显示，不改算出什么**。所有 `cv/*.csv` 的数值与
1.1.1 逐字节相同；改变的是报告、表格、图与页面上的呈现方式。

### 新增

- **`sml/present.py`（展示规格层）**：字段标签、精度、单位、状态徽标、措辞守卫的唯一定义处。
  只依赖标准库。同一份规格同时驱动 Markdown、HTML、xlsx 显示格式与图内注记。
- **`report/report.html`（单文件响应式报告）**：CSS 全内联、零外链、零 JavaScript，
  三档断点（≥1100 / 760–1100 / <760）与 A4 打印样式；图引用 `../figures/*.png` 原图，
  不重采样；每个数字带 `data-raw`（全精度原值）与悬停提示（来源文件 + 字段）。
- **`figures/figure_qc.json`（图件完整性检查）**：落盘前检查双 y 轴、柱状图截断纵轴、
  jet/rainbow 色标、箱线图不叠散点、数值轴缺标签、F8 缺误差线。自检断言该文件为空。
- **`report/compliance.json`（措辞检查）**：扫描"证明/完全准确/无 p 值的显著优于/可外推"
  等表述，命中即写记录并在页面上挂红色横幅。
- **TRIPOD+AI 清单映射**：`sml/standards.py` 只做"条目 → 本次产物"的映射与三态渲染；
  27 条条目的文本放在 `assets/standards/tripod_ai_2024.json`，**须人工对照 BMJ
  2024;385:e078378 转录**（当前为空骨架，渲染为占位说明，不生成任何条目）。
- **配置新增** `data.target_unit`（字符串或 `{目标: 单位}`）、`data.target_resolution`、
  `report.html`、`report.precision_profile`、`report.strict_text`。

### 变更（会改变显示，不改数值）

- **精度按字段裁决**：R²/AUC/MCC 等有界量 3 位小数；RPD/RPIQ 2 位；RMSE/MAE/bias/intercept
  **4 位有效数字**（原来各处一律 3–4 位小数，`RMSE=0.0042` 与 `RMSE=1234.5` 用同一套小数位）；
  计数不带千分位；零值只输出 `0`（`-0.000` / `0.0000` 不再可能出现）。
- **比例统一为百分数**：T5 的 `域外样品比例` 从 `0.033` 改为 `13.3%`，与正文一致。
  CSV 里仍存 0–1，只有显示层变。
- **p 值**：4 位小数改为 3 位，且 `p<0.001` / `p>0.999` 用不等号（置换检验的 p 有硬下限
  `1/(1+n_perm)`，写 `0.0000` 是错的）。
- **xlsx 增加显示格式**：数字格式、冻结首行、自动筛选、列宽自适应。**单元格存储值仍是
  未舍入的原始 float**——补充材料做二次统计时拿到的不是被四舍五入过的数。
- **图内轴标签带单位**：`实测值 (mg/kg)`；未配置 `target_unit` 时写 `实测值 (单位未指定)`，
  不再静默省略量纲。
- **F4 标题加免责**：纵轴是标准化残差、未按 `√(1−h)` 校正，标注"仅作诊断"。
  **F8 标 `†`**：`metric_basis=pooled` 的行在图上直接标出，避免把不可引用的折均值并排画。
- **RPD 分档出处**：由"Williams 的常用分档"改为"常用经验分档（口径见
  `references/metrics-and-inference.md`）"，不伪造文献归属。
- `assets/style/palette.json` 补到 10 色并由 `plots.py` 实际读取（原文件无人读，第 9、10 个
  模型会静默同色）；新增 `assets/style/figure_specs.json` 收口图幅与字号。

### 破坏性 / 注意事项

- **旧 run 的 `--resume` 会判失效**：改 `.py` 会改变 `code_sha256`，这是设计意图。
  补救：`--stages report` 只重跑报告阶段，`cv/` 产物不受影响。
- 自检从 **34 项增至 53 项**（新增 `present` 10、`figure_qc` 5、`standards` 2、`reporting` 杠杆口径 2）。
- 新增错误码 `E-CFG-043`（`target_unit` 类型/缺目标）、`E-CFG-044`（`precision_profile` 取值）。

### 本轮未改（需人工裁决后再动）

- **`bias` 符号仍不统一**：`metrics.py` 是 `mean(y−ŷ)`，`calib.py` 的 Bland-Altman 是
  `mean(ŷ−y)`，文档写的是后者。本轮只在标签上标明方向（`bias` = 实测−预测，
  `ba_bias` = 预测−实测），不改算法——统一符号会改变已有 run 的数值语义，需重跑。
- **`rank` 用 RMSE、选模用 R²** 的双轨主指标保持原样。
- **F4 的统计定义**（标准化残差 vs 真学生化残差）保持原样，属方法学变更。

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
