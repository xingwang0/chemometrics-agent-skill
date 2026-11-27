---
name: research-ml-modeling
description: "科研级机器学习预测建模流程（材料/化学/光谱为主）。当用户要做近红外 NIR、红外 FTIR、拉曼 Raman、XRD、XPS、UV-Vis 光谱建模，或配方/工艺参数等表格数据的回归与分类预测；要写论文 Methods、画预测散点/Williams plot/校准曲线；要算 RPD、VIP、适用域、Y-randomization 置换检验；要回答审稿人关于交叉验证、数据泄漏、过拟合、重复测量、批次效应的质疑时使用。内置分组交叉验证 + 嵌套调参 + 重复 CV + 置换检验 + 按组 bootstrap 置信区间 + 适用域四法 + 审稿质疑清单。"
description_en: "Reviewer-grade machine-learning predictive modelling for materials, chemistry and spectroscopy (NIR, FTIR, Raman, XRD, XPS, UV-Vis) and general tabular regression/classification. Provides grouped and nested cross-validation, repeated CV, Y-randomization permutation tests, applicability domain (Hotelling T2, Q residuals, kNN distance, Williams leverage), calibration, cluster-bootstrap confidence intervals, multiple-comparison correction, publication-quality figures, a draft Methods paragraph and a reviewer-response checklist."
version: "1.2.7"
license: "Apache-2.0"
license_url: "https://www.apache.org/licenses/LICENSE-2.0"
allowed-tools: Read, Write, Edit, Glob, Grep, Bash
agent_created: true
---

# research-ml-modeling

把「审稿级严谨」的预测建模流程固化成一条命令：**原始表格 → 数据审计 → 嵌套分组 CV → 置换检验 → 适用域 → 论文级图表 + Methods 草稿 + 审稿清单**。

不适用于：纯统计假设检验（ANOVA/t 检验）、深度学习训练（本机无 torch）、图像分割、NLP。

## v1.2.x 变更（升级前先看这一段）

核心原则：**只改怎么显示，不改算出什么**。`cv/*.csv` 的数值与 1.1.1 逐字节相同。

| 变更 | 影响 | 处理 |
|---|---|---|
| **AD 结论在 Methods 里取「第一个模型」（1.2.5）** | 各模型判定不一致时，Methods/limitations 会与正文矛盾——真实数据上出现过正文写「1/5 有效」、Methods 写「无显著差异」，作者照抄 Methods 就丢掉了唯一有效的那条 | 用 `--stages ad,report` 重出；正文、Methods、limitations、审稿清单现在统一用汇总口径 |
| **旧 `ad_model.joblib` 现在直接报错（1.2.5）** | 缺训练参考集的旧模型会让 `predict.py` 报 `E-PRED-006`，不再静默输出「全判域外」却看起来正常的预测 | 用 `--stages ad,report` 重建；只想预测不判域就加 `--no-ad` |
| **多目标 run 末行给出总耗时（1.2.5）** | 以前只逐目标打印，容易把最后一个目标的时间当成整次运行 | 无需操作 |
| **`comparison.csv` 的 `metric_primary` 改标在 R² 行（1.2.5）** | 回归的主指标从 RMSE 改为 R²，与选模同一把尺子；数值不变 | 按 `metric_primary` 筛主指标的下游脚本行为会变 |
| **新样品判域修复（1.2.4，重要）** | `ADModel.score()` 曾用**待判样品自己**当杠杆参考集，单行打分恒得 h=2.0 → `predict.py` 把几乎所有新样品判为域外。现改为用训练得分矩阵作参考集 | **旧 run 必须重跑 `--stages ad,report`** 才会生成新的 `ad_model.joblib`（`--retrain` 只重建最终模型）；`ad_scores.csv`（自打分）数值不变 |
| **`permutation.mode=full` 改为显式拒绝（1.2.4）** | 旧版接受 full、实际跑 fast，却把 full 写进 `permutation.json`，报告会写出「每轮重搜超参」这句没执行过的协议 | 现在报 `E-CFG-045`；`permutation.json` 只记录实际执行的 `mode=fast` |
| **列表字段类型校验（1.2.4）** | `--set data.drop_cols=time` 曾把列表降级成字符串，`roles.update(str)` 按字符拆开 → 该列照旧进特征矩阵（静默泄漏） | 现在报 `E-CFG-046`，并提示用 `--set-remove` 或 JSON 列表 |
| **AD 有效性检验改按样品（1.2.4）** | 旧版在 OOF 行上做 Mann-Whitney，6 行（=1 个样品）即可"显著有效"；现在按样品平均绝对误差检验，行口径另存 `row_mannwhitney_p_greater` 备查 | `ad_eval.json` 的 p 值会变（幅度收敛），结论更保守 |
| **AD 有效性检验方向（1.2.2）** | 检验方向写反，`effective` 分支不可达，T5 的「AD 是否有效」恒为"否" | 用 `--stages ad,report` 重出；`ad_flag` / `ad_majority` 不变 |
| **排名与选模同口径（1.2.2）** | `rank` 与 `r2_gap_to_best` 改用选模主指标（回归 R²、分类 AUC），不再按池化 RMSE | 读 `rmse_gap_to_best` 的下游脚本需改名 |
| **md 表头带单位（1.2.2）** | `report.md` 的 RMSE/MAE 列头带 `data.target_unit`；ΔR² 不再误挂「单位未指定」 | 建议都填 `data.target_unit` |
| **「最优模型」需显著性（1.2.2）** | 未过 BH 显著性时改称「排序第一的模型」，不再出现无依据的"最优" | 无需操作 |
| **dry-run 不落 cwd（1.2.2）** | 不传 `--out` 时写系统临时目录；stage 列表按 `enabled` 过滤；估时改区间 | 无需操作 |
| **新增 `report/report.html`** | 单文件响应式报告，CSS 全内联、零外链零 JS，断网/邮件附件可原样打开；`report.md` 仍在 | 无需操作；`report.html: false` 可关闭 |
| **自检 34 → 55 项** | 新增 `present` 10、`figure_qc` 5、`standards` 2、`reporting` 口径 4（1.2.4 加「杠杆参考集」，1.2.5 加「AD 汇总结论」） | 换机先跑 `selfcheck.py`，55/55 再动数据 |
| **显示层收口到 `sml/present.py`** | 精度按字段裁决（R² 3 位、RPD 2 位、RMSE 4 位有效数字）；比例统一为百分数；`-0.000` 结构上不可能出现 | 只影响文字，`tables/*.csv` 与 `tables.xlsx` 的单元格值仍是未舍入 float |
| **p 值 4 位改 3 位**，`p<0.001` / `p>0.999` 用不等号 | 置换检验的 p 有硬下限 `1/(1+n_perm)`，写 `0.0000` 是错的 | 无需操作 |
| **图内轴标签带单位** | 需配 `data.target_unit`（如 `"%"`、`"mg/kg"`）；不填则写「单位未指定」并挂 blocking 徽标，**不再静默省略量纲** | 建议都填上 |
| **F4 杠杆口径修复（1.2.1）** | `fit` 与 `score` 的杠杆公式差一个 `1/n`，阈值却在带 `1/n` 的分布上估；现已共用 `ad._leverage()` | `ad_scores.csv` 的 `leverage` 列整体抬高 `1/n`；`ad_flag` / `ad_majority` 不变 |
| **`ad_reliability` 拆两个阈值** | `leverage_threshold`（判域线，与 `ad_scores.csv` 同值）与 `leverage_threshold_groups`（仅可靠性判断）分开，`h_star` 一并给出 | 无需操作 |
| **新增 `leverage_inert` / `leverage_max`** | 杠杆法"一次都没触发"却仍计入多数票分母，原来毫无提示 | 看日志一行说明；投票规则本轮未改 |
| **旧 run 的 `--resume` 会判失效** | 改 `.py` 会改变 `code_sha256`，这是设计意图 | 用 `--stages report` 只重跑报告，`cv/` 产物不受影响 |
| **新增 `figure_qc` / `compliance`** | 图件六类硬伤（双 y 轴、截断纵轴、jet 色标等）与措辞违规（"证明""可外推"）会落盘并挂红色横幅 | 自检断言 `figure_qc.json` 为空 |
| **TRIPOD+AI 清单映射** | `sml/standards.py` 只做映射与三态渲染，27 条条目**须人工转录**，当前为空骨架 | 未转录时渲染为占位说明，不生成任何条目 |

## v1.1.1 行为变更（历史记录）

上个版本有 8 处在「静默改变结果」或「该拦不拦」，这一版全部改成显式失败或显式告警。
**升级后旧配置可能直接报错，这是故意的**——每条的报错里都写了怎么改。

| 变更 | 影响 | 处理 |
|---|---|---|
| `holdout_random` + `split.repeats>1` 现在报 `E-CFG-042` | 旧实现会跑 N 个划分、把测试集混着算池化指标，还悄悄把 `metric_basis` 从 pooled 改成 fold | 复现别人的单次划分 → `repeats=1`；要多次划分的分布 → 外层循环改 `--seed` |
| `models[].fixed` 里拼错参数名现在报 `E-MOD-004` | 旧实现只 warn 然后**用默认值训练**，你钉死的值根本没生效 | 按报错给的最接近候选名改 |
| `split.group_col` 现在真的生效 | 旧实现只在报告里拼字符串，分组永远按 `data.id_col`；现在给了就用它，且自动剔出特征矩阵 | 本来想按 `sample_id` 分组却写了别的列，现在会真的按那列分组 |
| `data.feature_mode` 只接受 `all_numeric_except_roles` | 旧实现是死字段，写什么都不生效 | 要精确控制特征请用 `feature_cols` / `drop_cols` |
| **新增**：逐折预测值越界检查 | 线性模型在小样本上外推爆炸时，旧版只会显示"R²=−38"，看不出是数值跑飞 | 看 `fold_results.csv` 的 `pred_range_ratio` / `pred_n_extreme`，超 2 倍极差会 warn |
| **新增**：VIP 可靠性标记 | PLS 自己没预测力时，VIP 排序仍是"看起来很专业"的噪声 | `vip.csv` 多 `pls_cv_r2` / `vip_reliable` 两列；不可引用时报告自动改 ⚠ |
| `--dry-run` 不再对 `holdout_random` 谎报折数 | 旧版打印"外层 5 折"，实际只跑 1 折 | 无需操作，数字改对了 |
| 版本号统一 | `SKILL.md` 与 `sml.__version__` 现在一致（旧版是 1.1.0 / 1.0.0） | 引用时写明 1.1.1 |

## 路径与首次使用

开发与验证所用环境（**不是**对你的要求，换机时按自己的填）：

```
解释器  python                # 验证环境为 3.14，版本矩阵见 references/environment.md
技能根  /path/to/research-ml-modeling
```

**换机或换人**：把上面两行换成你自己的解释器与技能根，然后**先跑自检**再谈建模：

```bash
python scripts/selfcheck.py          # 55 项：解释器、依赖版本矩阵、模型注册表、泄漏探针、口径一致性、展示规格、图件完整性
```

自检会逐项告诉你缺哪个包、版本差在哪、哪条 API 已移除，不需要猜。全绿之后才可能跑出可信数字。

Python 3.14.7 + sklearn 1.9.1 + xgboost/lightgbm/catboost + shap + matplotlib。
**没有** PyYAML / statsmodels / optuna / imbalanced-learn / torch / kaleido —— 不要写依赖它们的代码。
详细版本矩阵与已知 API 差异见 `references/environment.md`。

## 拿到数据先问 5 个问题

| 问题 | 答案 → 必须做的事 |
|---|---|
| 有多少**独立样品**？（不是光谱条数） | < 15 → 只做探索，不下结论；15–30 → 外层 3 折；> 30 → 5 折 |
| 同一样品是否测了多次？ | 是 → **必须有 `sample_id`**，否则 R² 虚高（见 `references/validation-protocols.md`） |
| 有没有批次/仪器/日期？ | 有 → 主结果 `nested_group`，另跑 `loo_batch` 作真外推 |
| 回归还是分类？ | 分类 → 检查类别不平衡；阳性类要用 `task.label_order` 显式指定 |
| 要不要外推到新样品？ | 要 → 必须看适用域（`ad/`）且验证其有效性 |

## 三条命令

```bash
PY="python"                                    # 换成你自己的解释器
ROOT="/path/to/research-ml-modeling"           # 换成技能目录
DATA="/path/to/your_spectra.xlsx"

# 1) 环境自检（首次/换机必跑，55 项）
$PY "$ROOT/scripts/selfcheck.py"

# 2) 预演：只审计 + 报"要拟合多少次、大概多久"，不训练
$PY "$ROOT/scripts/run_pipeline.py" --config "$ROOT/assets/config/regression_spectra.json" \
    --data "$DATA" --dry-run

# 3) 正式跑
$PY "$ROOT/scripts/run_pipeline.py" --config "$ROOT/assets/config/regression_spectra.json" \
    --data "$DATA" --out runs/nir_v1 --budget standard \
    --stages audit,cv,interp,perm,ad,report
```

`--config` 有 4 个预设：`regression_spectra`（光谱回归）、`regression_tabular`（普通表格回归）、
`classification_tabular`（分类）、`smoke`（1 分钟冒烟，用合成数据）。
改参数用 `--set a.b=c`，不要手写整份配置；字段全表见 `references/config-schema.md`。

**列表列要用 `--set-remove` 删元素，别用 `--set` 重写整串**：`--set` 对列表是整体替换，
做「去掉 time 的敏感性运行」时漏抄 `data.drop_cols` 里的任何一个字段，那个字段就回到特征
矩阵里——泄漏就是这么来的。

```bash
$PY "$ROOT/scripts/run_pipeline.py" --config ... --set-remove data.drop_cols=time
$PY "$ROOT/scripts/run_pipeline.py" --config ... --set split.repeats=10 --set permutation.n_perm=1000
```

## 模型注册表（kind × 任务）

同一个 `kind` 按任务自动落到对应估计器，**但能力边界是显式的**：写错在开跑前就报错，
不会出现"跑了两小时、结果表里少了一个模型"。

| kind | 回归 | 分类 | 备注 |
|---|---|---|---|
| `pls` | PLS 回归 | PLS-DA（有概率） | 强制基线，删不掉 |
| `pcr` | PCA + OLS | PCA + 逻辑回归（有概率） | `n_components` 按折内样本数裁到合法上界 |
| `ridge` `lasso` `enet` `knn` `rf` `et` `gbr` `xgb` `lgbm` `cat` `svr` | ✓ | ✓ | `svr` 在分类下落到 SVC；`lasso`/`enet` 分类下变成 L1/ElasticNet 逻辑回归，超参是 `C` 不是 `alpha` |
| `gp` | GP 回归 | GP 分类（Laplace） | 分类无 `alpha`，默认搜索空间为空；O(n³)，高维光谱慎用 |
| `logreg` `svc` `ridge_clf` `gnb` `lda` | **明确拒绝**（`E-MOD-003`） | ✓ | 分类器拟合连续 y 只会得到 `Unknown label type`，不如直接说清楚 |

**搜索键规则**

- 键一律写**参数名**，`model__` 前缀由程序加；嵌套估计器必须写完整路径。
- 键名在开跑前用 `set_params` 校验，拼错报 `E-MOD-002` 并给出最接近的候选名。
  这一步不能省：xgboost / catboost 会**静默接受**任意未知 kwarg，拼错只会安静地用默认值训练。
- `fixed` 优先于 `search`，按末级参数名匹配（`fixed:{n_components:10}` 能压住任何层级的同名键）。

**概率三态**：模型若无原生 `predict_proba`（只有 `decision_function`，如 `ridge_clf`），
打分器自动换成 `balanced_accuracy`，Brier/ECE/log-loss 与 F5 校准图按缺失处理
（`fold_results.csv` 的 `proba_kind=decision`），AUC/AP 照常——绝不用 sigmoid 假概率算 Brier。

## 六阶段与 Gate

| 阶段 | 做什么 | 产物 | Gate（不满足就别往下走） |
|---|---|---|---|
| `audit` | 数据审计 | `audit/audit.md` | 有 `critical` 项（目标泄漏/ID 当特征）必须先修 |
| `cv` | 嵌套分组重复 CV | `cv/oof_predictions.csv`（唯一真源） | 失败折数 > 20% 要查原因；**整模型全部失败会在报告顶部告警并从表中剔除** |
| `interp` | 波段重要性/VIP/系数稳定性 | `interp/*.csv` | — |
| `perm` | Y-randomization | `perm/permutation.json` | p ≥ 0.05 必须在文中说明"不能排除拟合噪声" |
| `ad` | 适用域 + 有效性验证 | `ad/ad_scores.csv`、`ad/ad_eval.json` | `ad_effective=false` 时禁止宣称可外推 |
| `report` | 图/表/报告/Methods | `report/report.md`、`figures/F1..F8` | 数字必须能回溯到 OOF |

单跑某个阶段：`--run-dir <已有 run 目录>`（此时不需要 `--config`）。

## 数据最小契约（5 条硬规则）

1. **一行一次测量**；`sample_id` 必须能标识"同一份材料/样品"。
2. 角色列（`id / batch / replicate / target`）绝不进特征矩阵——管线会自动剥离。
3. 光谱列名写成波数（`4000`、`4008`…），画图与波段分组才不需要额外配置。
4. 目标缺失的行会被剔除；特征缺失 > 50% 的列会被丢弃；其余用列中位数插补（审计会警告）。
5. 中文路径可用，但 `--out` 建议 ASCII；多进程报 pickle 错就 `--n-jobs 1`。

细节（长表格式、批次复合键、多目标、NA 取值）见 `references/data-format.md`。

## 严谨性红线

| 必须做 | 视情况做 | 禁止做 |
|---|---|---|
| 分组 CV（按 sample_id） | 留一批次外部验证 | 随机 KFold 报论文结果 |
| 嵌套 CV（调参与评估分离） | 重复 10 次（`--budget thorough`） | 用测试折选超参/定阈值 |
| 重复 CV 报 mean±sd（先按 repeat 聚合） | 多模型比较 + BH 校正 | 把 R×k 折混算 sd |
| 按组（cluster）bootstrap CI | DeLong（仅参考） | 按行 bootstrap |
| Y-randomization（样品级置换） | 波段级特征置换的 FDR | 按行置换 y |
| PLS 基线 + RPD/RPIQ | GP/贝叶斯优化 | 只报 GBDT 不报 PLS |
| 适用域 + 有效性验证 | 域外样本单独报性能 | 剔除域外样本后再报性能 |
| 预处理进搜索空间（`prep__which`） | 波段裁剪也作为候选 | 在全量数据上 fit 预处理 |
| AD 报**独立样品数**（含判定规则） | 同时给"多数测量越界"口径 | 把 OOF 行数当样品数（60 样品×3 重复×2 repeat=360 行，放大 6 倍） |
| 同页出现两个口径就标注（表=repeat 均值，图=OOF 池化） | 图注直接写口径 | 图 0.971 与表 0.969 并列却不解释 |

方法学理由见 `references/validation-protocols.md`、`metrics-and-inference.md`、
`applicability-domain.md`、`interpretation.md`。

## 两个可选开关（默认关闭，开了会改抽样）

默认配置 = 历史行为，结果逐位可复现。只有明确要"更严格的抽样语义"时才打开：

```bash
--set runtime.seed_scheme=per_model          # 模型数 ≥10 时 fold 种子不再跨折撞车
--set preprocess.stratify_sampling=true      # random 搜索按 prep__which 分层均分 n_iter
```

| 开关 | 默认 | 打开后 | 代价 |
|---|---|---|---|
| `runtime.seed_scheme` | `legacy` | 种子改 `(seed_r*100+k)*100+模型序`，`provenance.seeds` 多出 `per_fold_model` | 候选抽样序列变，旧 run 不能直接对拍 |
| `preprocess.stratify_sampling` | `false` | 5 条预处理 × n_iter=30 → 每条恰好 6 个候选（旧版实测会偏成 7/4/6/5/8） | 同上；grid 搜索不受影响（只告警） |

开任何一个都会在启动时打 `E-CFG-040` / `E-CFG-041` 提示。细则见 `references/config-schema.md`。

## 审稿人会说 → 你交什么

| 质疑 | 证据 |
|---|---|
| "随机划分导致同一样品泄漏" | `cv/fold_results.csv`（含 `n_train_groups`）+ `selfcheck` 泄漏探针 |
| "R² 是调参调出来的" | 嵌套 CV：`cv/metrics_summary.csv` |
| "换个划分就变了" | `figures/F3_cv_distribution.png` |
| "怎么证明不是拟合噪声" | `perm/permutation.json`、`figures/F6_y_randomization.png` |
| "新样品能用吗" | `ad/ad_eval.json`、`figures/F4_williams.png` |
| "指标没不确定度" | `cv/metrics_ci.csv` |
| "两个模型差异显著吗" | `cv/comparison.csv`（含 `p_BH`、`p_bonf`） |
| "为什么不用 PLS" | `interp/vip.csv` + T2 里的 PLS 行 |
| "有没有系统偏差" | `figures/F5_calibration.png`、`calibration` 字段 |
| "预处理是不是挑过的" | `cv/fold_results.csv` 的 `prep` 列 |
| "怎么复现" | `provenance.json`（三哈希 + 版本 + 种子） |

完整 57 条见 `references/reviewer-checklist.md`；每次运行会自动生成带 ✓/⚠ 的版本
`report/reviewer_checklist.md`。

## 降级措辞（数据不支持时必须这么写）

| 情形 | 论文里怎么写 |
|---|---|
| 无重复测量 | "本研究每个样品仅单次测量，无法评估重复测量带来的泄漏风险" —— 不要写"采用分组 CV 避免泄漏" |
| 无批次信息 | "模型仅在当前批次内验证，跨批次/跨仪器外推未经验证" |
| AD 无效 | "适用域分析未能识别出误差显著更大的域外样本，因此不对模型外推能力作声明" |
| 置换 p≥0.05 | "置换检验未能排除模型拟合噪声的可能，结果应视为探索性" |
| 最优与次优差异不显著 | "两者性能相当（配对 bootstrap，BH 校正 p=…），不能宣称其中某一个更优" |
| 独立样品 < 30 | "样本量有限（n=…），性能估计的置信区间较宽，模型尚未定型" |

## 产出物

```
runs/<id>/
├── config.resolved.json   provenance.json
├── audit/audit.md         数据审计（先看这个）
├── cv/                    oof_predictions.csv · metrics_summary.csv（按模型聚合）·
│                          metrics_by_prep.csv（按模型×预处理细分）· metrics_ci.csv · comparison.csv
├── interp/                importance.csv · vip.csv · coef_stability.csv
├── perm/permutation.json  Y-randomization
├── ad/ad_eval.json        ad_model.joblib（新样品判域）
├── model/                 final_model.joblib · model_card.json（全量重训，供新样品预测）
├── figures/F1..F8         png(300dpi) + pdf(矢量)
├── tables/T1..T6          csv + tables.xlsx
└── report/                report.html（单文件响应式）· report.md · methods_zh.md ·
                           methods_en.md · reviewer_checklist.md ·
                           compliance.json（措辞检查）· tripod_ai_checklist.md（清单已安装时）
```

`report.html` 与 `report.md` 十章同名同序：CSS 全内联、零外链零 JS（断网可读）、
三档响应式断点 + A4 打印样式；图引用 `../figures/*.png` 原图；每个数字带 `data-raw`
（全精度原值）与来源提示。`report.md` 仍是纯文本交付物，两者不会各说一套——正文措辞
由 `report.py` 产出后由渲染层复用。

按目标单位的配置：`data.target_unit`（如 `"%"`、`"mg/kg"`）。不填时表头与图内轴标签会写
「单位未指定」并在页面挂 blocking 徽标——**不会静默省略量纲**。

多目标（`task.target` 给多列）时**按目标拆成独立子 run**，每个子目录都是一套完整产物
（**兄弟目标列会自动从特征里剔除**——否则会用「蛋白」预测「水分」，分数虚高且审计难以识别）：

```
runs/<id>/
├── targets_index.json     targets_summary.md      ← 各目标最优模型并列（不做跨目标排序）
└── targets/01_<目标A>/    02_<目标B>/             ← 与上面的单目标结构完全相同
```

原因：不同目标的指标尺度与难度不同，全表排名只会选出"跨目标最强"的那个模型，
报告与 CI 也会串目标。若确实要把多目标塞进一个目录（`--no-fanout`），
`report` 阶段会直接拒绝（`E-STAGE-004`）而不是给出错数字。

**聚合口径**：`metrics_summary.csv` 按 **(target, model)** 聚合，`rank` 与 `gap_to_best`
都在**目标内**计算，且用的是**与选模相同的主指标**（回归 R²、分类 AUC，列名
`r2_gap_to_best`）；早先 rank 按池化 RMSE 排、选模按折均值 R²，会出现
"rank=1 的模型不是报告里那个最优模型"。`prep_mix` 列记录该模型在各折里分别选中了哪条预处理链。
**不要**按"模型×预处理"拆分排名——预处理是逐折内部选出的超参，
拆开排名等于在几十个组合里挑最优（winner's curse）。细分结果另存 `metrics_by_prep.csv`。

`metrics_summary.csv` 另有 `n_folds_ok` / `n_folds_degenerate` / `metric_basis` 三列：
`metric_basis=pooled` 表示该行有效折不足 3 个（留一法下必然如此，逐折 R² 在单样品上退化），
**折均值不可引用**，排名、报告与 F1/F8 都会改用池化 OOF 指标，报告里也会标 † 并说明。
`rebuild_tables.py` 走同一份判定，重算表同样会标 pooled 并给出警告——不会静默退回折均值。

**「最优」必须有显著性支撑**：报告会取该模型与同目标次优模型的配对 bootstrap 差值
（BH 校正）。若 CI 跨 0，全文写"二者性能相当"，不写"最优模型 X 优于 Y"，
并自动进入 limitations 与审稿清单。`comparison.csv` 是证据位置。

改了聚合口径或指标定义、又不想重跑 CV 时：

```bash
$PY "$ROOT/scripts/rebuild_tables.py" --run-dir D:/runs/nir_v1          # 只重算表，不重新拟合
$PY "$ROOT/scripts/rebuild_tables.py" --run-dir D:/runs/nir_v1 --bootstrap 5000
```

## 预测新样品（工艺窗口、外推）

`ad/ad_model.joblib` 只判域，不给预测值。**最终模型在 `cv` 阶段就存好了**：
`model/final_model.joblib`（用 CV 选出的最优模型 + 各折最常用超参，在全部有效行上重训）
和 `model/model_card.json`（模型名、`selection_basis`、实际应用的超参、训练行数、特征名）。
性能数字仍然只引用 `cv/metrics_summary.csv`，不要拿训练集拟合值当性能。

```bash
# 1) 预测一张新样品表（默认写到 <run>/predictions/predictions.csv）
$PY "$ROOT/scripts/predict.py" --run-dir D:/runs/nir_v1/targets/01_水分 --data D:/data/new.csv

# 2) 只重建最终模型（旧 run 没有 model/，或改了选模口径后想同步）
$PY "$ROOT/scripts/predict.py" --run-dir D:/runs/nir_v1/targets/01_水分 --retrain

# 3) 指定输出、跳过适用域判定
$PY "$ROOT/scripts/predict.py" --run-dir ... --data new.xlsx --out D:/pred/ --no-ad
```

- 新样品表的**列名必须与训练特征完全一致**；缺列报 `E-PRED-002` 并列出前 10 个缺失列，
  不会按 0 悄悄填。缺失格用训练集列中位数插补。
- 输出列：`prediction`；分类另有概率列——二分类为 `proba_positive`（正类概率），
  多分类（>2 类）为 `p0..pN`（逐类概率）加 `proba_max`（最大类概率）。另有
  `ad_flag`、`ad_votes`、`ad_majority`。**`ad_flag=1` 的行落在适用域外，只能作参考，不能写进结论。**
- 工艺窗口：把候选工艺参数/光谱组合写成一张表（一行一个待预测点），整表喂 `--data`
  一次批量预测，再拿 `ad_flag=0` 的边界去画窗口。
- 多目标 run 请指到 `targets/0N_<目标>/` 子目录：不同目标不共用一个 joblib。

不要自己 `import sml` 手搓这套流程——超参键的 `model__` 前缀、预处理链、中位数插补和
判域调用都得对齐，`predict.py` 已经把它们封装好了。

## references 索引

| 文件 | 什么时候读 |
|---|---|
| `data-format.md` | 列角色怎么设、长表怎么办、分组键怎么定 |
| `config-schema.md` | 改配置、查字段默认值、看错误码 |
| `validation-protocols.md` | 为什么必须分组/嵌套/重复，折数怎么选 |
| `spectral-preprocessing.md` | SNV/MSC/SG/airPLS 的公式、顺序、参数范围 |
| `metrics-and-inference.md` | RPD/RPIQ/CCC 定义、cluster bootstrap、FDR、Y-randomization 统计 |
| `applicability-domain.md` | T²/Q/kNN/leverage 阈值、AD 有效性检验、新样品判域 |
| `interpretation.md` | 为什么 SHAP 不能当主图、波段置换怎么做 |
| `reporting-and-figures.md` | 8 张图的规范、表结构、Methods 15 项清单、展示规范（精度/单位/徽标/图件检查/HTML） |
| `standards-mapping.md` | 结果呈现参照了哪些标准、为什么"不声明符合"、TRIPOD+AI 怎么转录 |
| `reviewer-checklist.md` | 投稿前逐条自查（57 条） |
| `environment.md` | 版本矩阵、已知 API 差异、换机检查清单 |
| `troubleshooting.md` | 报错速查 |

## 三条容易忘的

1. `--dry-run` 一定先跑：这个技能失败的方式不是报错，而是"跑了六小时才发现折数设错"。
2. `--resume` 按三哈希判定：改任何一个 `.py`（哪怕注释）都会让产物失效重跑，这是故意的。
3. 报告里的每个数字都能从 `cv/oof_predictions.csv` 重算；如果对不上，先怀疑自己手改了表。

## 许可与免责

以 **Apache License 2.0** 发布，全文见 `LICENSE`，归属与详细免责见 `NOTICE`。

- 允许商用、修改、再分发；须保留版权与许可声明，**修改过的文件须标注改动**。
- 不授予商标权，不得以本项目名义为衍生作品背书。
- 专利许可按 Apache-2.0 第 3 条授予；使用者若就本项目提起专利诉讼，其许可自动终止。
- **AS IS**：不提供任何明示或默示担保。全部输出（指标表、图、Methods 草稿、审稿清单）
  都是计算产物，**投稿前必须人工核对**样品来源、仪器型号与测量条件、指标口径。
- **未针对**监管申报（ICH/USP/CLSI/GB）、临床诊断、生产实时控制等场景做过验证，
  不得将本管线的输出直接作为上述用途的判定依据。
- **作者不对**据此作出的科研结论、生产决策、监管申报或商业交付承担任何责任。
- 上述免责受适用法律约束；中国法下依法不得排除的责任（故意或重大过失致财产损失）
  不因本声明而免除，其余部分仍有效。

引用时请注明 `version` 与对应 run 目录下 `provenance.json` 的三哈希。
