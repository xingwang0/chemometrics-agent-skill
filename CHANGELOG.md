# CHANGELOG

格式参照 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)；版本号遵循语义化版本。

## [1.2.6] - 2026-11-26

### 新增

- **DeepSeek Harness 插件封装**（`package.json` / `index.js` / `cordis.patch.yml`）
  仓库现在同时是一个 dsh 插件包 `dsh-research-ml-modeling`：`index.js` 在加载时把
  `SKILL.md` 注册到 `ctx.skills`，`resourceBase` 指向包根，所以 `references/`、
  `scripts/`、`assets/` 的相对路径不用改；升级插件即升级技能，不往 `~/.dsh/skills` 复制。
  安装：`dsh plugin --profile desktop add github:xingwang0/research-ml-modeling`。
  零依赖、无构建脚本，包内文件清单见 `package.json` 的 `files`。
  **技能正文未变**（除版本号），`cv/` 产物与 55 项自检结果不受影响。

## [1.2.5] - 2026-11-26

第四轮真实数据复现（一份 100+ 行的中等规模光谱数据、2 个目标、B 套 5×3×5）暴露的 5 条问题。
**数值零变化**：重算 `metrics_summary` 的浮点差 ≤1e-15，`rank` 与 `p_BH` 完全相同。

### 修复（结论一致性）

- **Methods / limitations / 审稿清单的 AD 结论，取的是「第一个模型」**（`sml/report.py`）
  三处都写 `next(iter(per_model.values()))`。当各模型判定不一致时（本轮目标二：1 个
  `effective` + 4 个 `no_difference`），报告正文写「各模型不一致」，而 Methods 与
  limitations 写「无显著差异」——同一份交付物自相矛盾，作者照抄 Methods 就会把唯一
  那条有效结论丢掉（反过来，第一个模型是 effective 时会把无效写成有效）。
  新增 `ad_verdict_summary()` 汇总五态（all_effective / none_effective / mixed /
  all_conflict / untested），上述三处统一用它；`mixed` 时点名哪些模型有效、哪些无差异。

- **多目标 run 不给总耗时**（`run_pipeline.py`）
  每个子 run 打印自己的「总耗时」，末行 `多目标完成` 没有时长，读者会把最后一个目标的
  时间当成整次运行的时间（本轮复现因此把 21.3 分钟引成 10.2 分钟，算错加速倍数）。
  现在末行输出 `总耗时 X 分钟`（整个 fanout 的墙钟）。

- **`comparison.csv` 的 `metric_primary=True` 标在 `rmse` 行**（`sml/compare.py`）
  `PRIMARY_METRIC` 的回归值改为 `r2`，与 `selection.PRIMARY_BY_TASK`、`_best_by_primary`、
  `pick_best_model` 同一把尺子。RMSE 仍会在比对表里，只是不再冒充主指标。
  BH 家族成员不变，`p_BH` 数值 0 变化。
  **连带影响**：`comparison.csv` 的 `wilcoxon_p` 也改为按 R² 做配对 Wilcoxon（原按 RMSE）。
  该列只写不读（`report` / `plots` 都不消费它），故不改变任何报告结论；
  需要按 RMSE 做 Wilcoxon 的下游请自行重算。
  顺带修 `rank_models` 的兜底分支：不再退回「刚被判定缺失的那个指标」，改为按
  mae → rmse → bias 依次找存在的列。

### 修复（显式失败）

- **旧 `ad_model.joblib` 静默降级**（`sml/ad.py`、`sml/final_model.py`）
  1.2.4 起 `score()` 在缺少训练参考集时只 `warn` 一句，然后退回「待判样品自己当参考集」，
  于是 `predictions.csv` 的 `ad_flag` 几乎恒为 1，而文件看起来完全正常。
  现在：`predict.py` 直接报 `E-PRED-006`（附重建命令与 `--no-ad` 两种出路），
  `score()` 抛 `RuntimeError`。
  新增 `AD_SCHEMA_VERSION = 2` 与 `ad_model_version()`；1.2.4 生成的模型带 `T_train_`
  但没有版本号，按 v2 处理——只拦真正缺参考集的 ≤1.2.3 模型，避免逼用户重建本来就没问题的模型。

### 修复（噪音）

- **中文目录 + 多进程的告警重复 4 次**（`run_pipeline.py`）
  改为每个 run 目录只提醒一次；判断从 `int(n_jobs) > 1` 换成 `resolve_workers(...) > 1`，
  `n_jobs=-1`（默认「用满核」）也能触发——旧写法 `-1 > 1` 为假，最需要提醒的场景反而不提醒。

### 其他

- 自检 54 → 55 项：新增「AD 汇总结论：各模型不一致时不得概称『有效』或『无差异』」，
  对应上面 Methods / limitations / 审稿清单三处取「第一个模型」的修复。
- `README.md` 与 `references/environment.md` 的自检项数与 `SKILL.md` 对齐（此前仍写 53）。

### 已知未修（截至 1.2.5）

以下条目在 2026-11-26 的审计里记录在案，**不改变任何已跑出结论**，但会影响他人按报告复算时的对账：

| 位置 | 问题 | 影响 |
|---|---|---|
| `sml/audit.py` vs `sml/ad.py` | audit 用 `pca_max_comp` 估杠杆退化，ad 用实际解释方差成分数 | 小样本时一个报「AD 退化」、另一个报 `ad_reliable=true` |
| `sml/final_model.py` | 分类 `predict.py` 输出标签编码 0..N−1，`label_map` / `positive_label` 存了不用 | 用户与原表按类名对照会错位 |
| `sml/final_model.py` | 中位数向量尺寸不符时静默 `nan_to_num(nan=0)` | 光谱填 0 看起来"正常" |
| `sml/metrics.py` | RPIQ 无置信区间（充分统计量给不出四分位） | `metrics_ci.csv` 永远缺 RPIQ 行，与文档「各自 95% CI」不符 |
| `sml/tune.py` + `sml/cv.py` | 内层某折只剩单一类别 → `log_loss` 全 NaN → 该外层折整体报废；`n_folds_ok` 的分母不含失败折 | 「3(3/3)」实为 3/25 |
| `sml/selection.py` | 仅 1 个有效折的模型仍可被选为最优并写入 `final_model` | 无最小成功折比例门槛 |
| `scripts/run_pipeline.py` | `--run-dir` 模式静默忽略 `--n-jobs/--seed/--budget` | 用户以为参数生效 |
| `sml/utils.py` | 中文字体缺失时静默出方块，不进 `figure_qc.json` | 换机跑批时整批图是方块而日志干净 |

## [1.2.4] - 2026-11-26

审计修复版本。**CV 指标（`cv/*.csv`）数值不变**；变的是 AD 判域、配置校验、AD 有效性
检验口径与展示层文案。其中前两条属于「会让论文写出没做过的事」的级别。

### 修复（正确性）

- **`ADModel.score()` 的杠杆参考集用错（阻塞级）**（`sml/ad.py`）
  `h_i = t_iᵀ(TᵀT)⁻¹t_i + 1/n` 里的 `T` 与 `n` 取自**被打分的那批样品自己**，而不是训练集。
  m 行批量时 `Σh ≈ min(k,m)+1`，单行打分恒得 `h=2.0`，而阈值通常 0.1–0.3 →
  **`predict.py` 把所有新样品判为域外**，`ad_flag` 恒为 1。
  实测（k=10, n_train=180, h*=0.183）：m=1/5/10 越界率 100%，m=50 为 62%，m=180 才恢复正常。
  报告里查不出来：`stage_ad` 对全量自打分时 `T_new ≡ T_train`，两条路径天然一致；
  53 项自检的 `_leverage_formula` 也用同一个 X 做 fit 与 score，结构上无法发现。
  现 `_leverage(T, T_ref)` 显式区分「被打分的点」与「参考集」，`ADModel` 保存 `T_train_`。
  **旧 run 需用 `--stages ad,report` 重跑**（`ad/ad_model.joblib` 里没有训练参考集，旧文件会走兼容分支并告警）；`--retrain` 只重建 `final_model.joblib`，不重建 AD 模型。

- **`permutation.mode=full` 标注为 full、实际跑 fast**（`sml/permutation.py`、`sml/config.py`）
  `full` 从未实现，但 `mode` 被原样写进 `permutation.json`，`report.py` 见到 `!= fast`
  就渲染「连超参数一起重新搜索」——一句没有执行过的方法学陈述。
  现在配置阶段即报 `E-CFG-045`；`permutation.json` 只记录实际执行的 `mode=fast`（另存 `mode_requested`）。

- **`--set` 把列表字段降级成字符串导致静默泄漏**（`sml/config.py`）
  `--set data.drop_cols=time` → `"time"` → `io_data.roles.update("time")` 按**字符**拆成
  `{'t','i','m','e'}`，`time` 列照旧进特征矩阵，不报错不告警——正是文档反复警告的泄漏路径。
  现在对 9 个列表字段强制类型，报 `E-CFG-046` 并提示 `--set-remove` / JSON 列表写法。
  同时修 `--set data.id_col=001` 前导零被吃成 `1`、`--set task.target=2024` 变 int 的问题。

- **AD 有效性检验按 OOF 行做**（`sml/ad.py`）
  60 样品 ×3 重复 ×2 repeat = 360 行，`len>5` 门槛等价于「1 个样品」；同一样品残差高度相关，
  Mann-Whitney 的独立性假设被破坏 → p 偏小、"有效"偏容易成立。
  现按样品平均绝对误差检验（`test_unit=group`），行口径另存 `row_mannwhitney_p_greater` 备查。
  实测同一数据：行口径 p=1.9e-06，样品口径 p=5.1e-04。

- **`ad_reliability.leverage_threshold` 报的不是判域线**（`stage_ad.py`）
  旧代码把理论线 `3(k+1)/n_rows` 赋给它，注释却写「与 `ad_scores.csv` 同值」；
  q95 主导时两者不等（例 0.165 vs 0.565）。现改用 `diag["leverage_threshold"]`，
  另存 `leverage_threshold_theory` / `leverage_q95` / `leverage_threshold_rule`。

- **`score()` 的 kNN 少算一阶邻居**（`sml/ad.py`）
  fit 用「第 k 个**他人**」估阈值，score 取 `dist[:,-1]`；自打分时最近邻是自己（距离 0），
  实际落到第 k−1 个他人 → `ad_scores.csv` 的 kNN 判据系统性偏松。现多取一列并识别自身命中。

- **`config_conflicts` 只比较 4 个字段**（`sml/config.py`）
  换 `split.strategy` / `drop_cols` / `preprocess.candidates` 复用同一 `--out` 时，
  `config.resolved.json` 被静默覆盖，报告描述与实际数字错配。现扩到 13 项。

- **`--resume` 会让 `--seed` 之外的无害改动也触发全量重跑**（文档已如实说明）
  见 `SKILL.md`「三条容易忘的」。

### 修复（展示层）

- `present.fmt_legacy` 加负零归零：`-0.000` 结构上不再可能出现（旧版只在 HTML 成立）。
- p 值统一走 `present.fmt_p`（3 位 + `p<0.001` / `p>0.999`）；`report.md` 不再出现 `p=0.0005`。
- T2 表格单元格按字段规格裁决精度（R² 3 位、RPD 2 位、RMSE 4 位有效数字），不再一律 3 位。
- Methods 与 HTML 验证方案改用**实际折数/重复次数**（`cv_summary.n_outer_folds`、
  `effective_repeats`），`loo_batch` 下不再虚报「5 折 × 重复 5 次」。

### 修复（文档与实现对齐）

- `permutation.mode=full`、Youden 阈值两处「声称做了但没做」的表述改为事实描述：
  管线分类阈值固定 0.5，`_youden_threshold()` 无调用点。
- 版本号统一为 1.2.4（`SKILL.md` / `sml.__version__` / `CITATION.cff`）。
- 自检 53 → 54 项：新增「杠杆判域必须相对训练参考集」，覆盖旧版结构性盲区。

## [1.2.3] - 2026-11-25

纯性能版本：**产物数值与 1.2.2 逐位相同**（同一数据实测 `oof_predictions.csv`、
`fold_results.csv`、`importance.csv` 数值最大偏差 0.000e+00，置换零分布 200 个值
全等），改的只是"谁来算、在哪算"。

同一份 B 配置（2 目标 × 5 重复 × 5 外折 × 5 模型）实测：**41.6 min → 13.5 min（3.09×）**。

### 修复（性能）

- **置换检验用线程后端，等于没并行**（`sml/permutation.py`，最高优先级）
  旧实现 `Parallel(..., prefer="threads")`。实测 xgb 在 4/12 线程下只有 **0.76–0.83×**
  （GIL 争用，比串行还慢），rf 也只有 1.23×。改为 loky 进程 + 按 (置换, 折) 细粒度派任务
  （200×25 = 5000 个任务，尾部空转更少），折模型通过 initializer 每个 worker 只传一次。
  实测 perm 阶段 **828s → 95s（8.7×）**、**767s → 114s（6.7×）**。

- **波段置换重要性完全串行**（`sml/interp.py`）
  旧实现双层 for 循环：波段 × 重复 × 折，2000 列光谱上就是上万次 predict 一个接一个跑。
  现在按 (波段 × 重复) 并行；**置换索引仍在主进程按原 rng 调用顺序生成**再分发给 worker，
  所以每一次置换的内容与串行实现逐位相同。实测 **214s → 63s（3.4×）**。

- **n_jobs=-1 被解析成核数**（`sml/utils.py`、`sml/cv.py`）
  14 核异构大小核机器上把 worker 开满，反而不如留 8–10 个：E 核单核吞吐只有 P 核
  六成，长任务派到慢核上拖尾。实测同一批 xgb 任务：4→3.0×、8→3.4~5.3×、10→4.4×、
  14→3.5×。新增 `resolve_workers()`：`-1`/`0` → `round(0.7×核数)`（上限 24），
  显式正整数原样使用。

- **小数据上并行会倒亏**（`sml/utils.py` `parallel_pays_off`）
  loky 起 10 个 worker 固定成本 2.4s，initializer 带大数组时 6s+。smoke 那种整个 run
  只要 7 秒的数据上，perm 被从 0.2s 拖到 22s。现在 cv / interp / perm 都**先串行跑一个
  单元实测耗时**，再据此决定剩下的是并行还是继续串行；实测 smoke 已回到 5.8s
  （cv 1.2s / interp 0.9s / perm 0.2s，均不超过串行基线）。

- **折模型重拟合与 VIP bootstrap 串行**（`sml/interp.py`、`stage_interp.py`）
  `fit_fold_models` 的 N 个折、VIP 的 100 次按样品重采样都彼此独立，现按任务并行
  （同样先热身判门槛）；bootstrap 的抽签序列仍在主进程预先生成，B 矩阵行序不变。

### 说明（不影响结果）

- 所有并行都在"独立任务级"：候选网格、折、置换、波段、bootstrap 重采样之间互不依赖，
  每个任务的输入与随机数序列逐位不变，指标计算仍由主进程按原顺序拼接后完成。
- `--dry-run` 的估时改用同一套 worker 解析 + 0.55 的实测并行效率系数。
- 未改动：`n_perm` / `n_iter` / `repeats` / `n_splits` / 所有 seed 与 rng 调用顺序、
  超参空间、指标定义与聚合口径。

## [1.2.2] - 2026-11-25

第三次全量复现（同一数据、同一配置，1.1.0 → 1.2.1）在数值上**逐格零差异**，
但实测暴露出若干「会让结论写错」的缺陷。本版按实测修，**不改任何指标计算**：
`cv/` 下的数值产物与 1.2.1 相同，变的只有判定、排名口径、措辞与单位呈现。

### 修复（会改变结论的）

- **D1 适用域有效性检验的方向写反了**（`sml/ad.py`，高）
  回归分支 `mannwhitneyu(r_in, r_out, alternative="greater")` 的备择假设是
  「域内误差 > 域外误差」，与要证的方向相反：域外确实更差时 p 顶到 0.99+，
  `effective` 分支不可达、`reversed` 也永不触发，全部判成 `no_difference`；
  分类分支的 `fisher_exact(..., alternative="less")` 同样写反。
  现改为**秩方向判定**（Mann-Whitney 比的是秩/中位数）+ 方向标签 `effect_direction`，
  同时落盘单侧与双向 p（`mannwhitney_p_greater` / `_less` / `_two`；分类为 `fisher_p_*`）。
  实测影响：目标一由「全无差异」变为 5/5 模型显著有效（域外误差显著高于域内）；
  目标二变为 2/5 有效、2/5 无差异、1/5 口径冲突。T5 的
  「AD 是否有效」改为四态（是 / 否（无显著差异）/ 否（方向相反）/ 无法判定（口径冲突））。

- **D1+ 判定用均值、检验用秩**（同一处，随 D1 一起修）
  旧逻辑拿**均值**判 effective / reversed，而 Mann-Whitney 判的是秩：均值被少数
  极端残差拉动时会与 p 值打架。某份真实数据上，XGB 的秩检验显著且中位数域外更高，
  但均值域外更低，旧代码据此判 `reversed`，
  等于用均值解释一个秩检验。现在两种口径打架时判 `conflict`（`ad_effective=None`），
  报告写「无法判定（秩方向与均值方向相反）」并进 limitations，不再硬给结论；
  `mean_abs_error_*` / `median_abs_error_*` / `mean_rank_direction_conflict` 一并落盘。

- **AD 总结论只取首个模型**（`stage_ad.py`，D1 修复后显形）
  各模型判定不一致时只报第一个会掩盖分歧（且读起来像"AD 有效"）。现在按判决分组
  列出具体模型名，总结论改为「部分模型上适用域能区分出更差的预测，但不足以作为
  统一的域外筛选依据」，并落盘 `verdict_mix`；逐模型明细仍看 T5。

- **D4 排名与选模不同口径**（`sml/compare.py`）
  `rank`/`gap_to_best` 按池化 RMSE 排、选模按折均值 R²，于是同一份报告里
  「rank=1 的是折均值 R²=−0.21 的 KNN，而最优模型写 XGB」。现两者都用选模主指标
  （回归 R²、分类 AUC）；gap 列随之改名 `r2_gap_to_best`（无 `r2` 列时才退回
  `rmse_gap_to_best`）。

- **D9 「最优模型」没有显著性支撑**（`sml/report.py`）
  与次优差异未达显著时仍写「最优模型：xgb」，`compliance.json` 同时报违规。
  现在只有「配对 bootstrap 显著、且 BH 校正后仍显著」才配写「最优模型」，
  其余写「排序第一的模型（未过显著性，不构成「最优」）」。

- **D10 置换检验与性能表两种口径并列却不标**（`sml/report.py`、`report_html.py`）
  抗噪声检验的观测值是**池化 OOF** 指标，与上文折均值不是同一个数（本项目
  折均值 0.083 vs 池化 0.2538）。该句现在显式标注口径。

- **D3 单位缺失与单位误挂**（`sml/report.py`、`sml/present.py`）
  md 报告 T2 表头完全没有单位（html 却有徽标），作者抄 md 就丢量纲。
  表头现按字段规格 + `data.target_unit` 带单位，未填时显式写「单位未指定」。
  另：ΔR² 系列（`delta_mean/sd/min/max`）被当成 `delta` 的 `unit=None` 处理，
  给一个**无量纲量**挂了「单位未指定」徽标；现单独注册 `delta_r2`（`unit=""`）。

### 修复（会误导流程的）

- **D5 `--dry-run` 污染当前目录**：不传 `--out` 时按 `runs/<时间戳>_<数据名>`
  建出整套骨架（`config.resolved.json`/`provenance.json`/`audit/` + 10 个空 stage 目录），
  一条「只看审计」的命令在用户工作目录留下半套 run。现在落系统临时目录并打印路径；
  dry-run 也只建 `audit/`。

- **D7 dry-run 的 stage 列表不按 `enabled` 过滤**：配置里 perm/ad/interp 全
  `false` 仍打印「将执行 6 个」，估时也照算。现按 `*.enabled` 过滤并单列被跳过的 stage。

- **D12 总拟合次数口径不一致**：dry-run 表格合计 8775，报告写「总拟合次数 8900」，
  差 125 = 每折用最优超参的最终重拟合。`estimate_fit_calls` 现把它计入
  （并分列 `search_fit_calls` / `refit_fit_calls`），报告与 html 注明口径。

- **D8 估时偏小 4.1×**：dry-run 把「纯拟合下限」当作预计总时长（本项目
  5.0 min/目标 vs 实测 20.8）。现按「下限 – 4× 上限」区间报，并说明上限含
  折间调度、预处理重建、绘图与报告。

- **D6 死字段 `_n_iter_budget`**：`config.py` 写入、全仓无人读，
  却出现在 `config.resolved.json` 里，让人误以为调参预算被记录。已删除。

### 验证（本项目数据，v1.2.1 → v1.2.2）

- `selfcheck.py` 53/53 通过。
- 用 1.2.2 重跑 B 套 `--stages cv,ad,report`（100+ 行 / 100+ 组，两个目标）：
  `cv/metrics_ci.csv`、`cv/oof_predictions.csv` **逐字节相同**；`cv/metrics_summary.csv`
  的 34 列中除 `rank`（唯一有意变更）与 `rmse_gap_to_best→r2_gap_to_best` 改名外，
  **逐格零差异**。两个目标的最优模型均与 1.2.1 相同。
- `report/compliance.json` 的 `best_claim` 违规由 2 条降为 **0 条**。
- `--dry-run`：拟合合计 8,775 → **8,900**（与报告一致）；估时由单值 5.0 min
  → **区间 5.0–20.1 min**（实测 20.8）；A 配置打印「跳过 interp, perm, ad」；
  当前目录**零残留**。

### 未改（留待方法学裁决）

- 置换检验的范围、AD 投票规则（`ad_voting` 的多数票定义）未动。
- 同一 `--out` 混用两套配置仍报 `E-CFG-030`。

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
