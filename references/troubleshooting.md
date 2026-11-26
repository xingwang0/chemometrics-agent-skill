# 报错与异常速查

---

## 环境 / 编码

### `UnicodeEncodeError: 'gbk' codec can't encode character`

从 cmd.exe 或计划任务里运行时，stdout 不是 UTF-8。

* 本包每个入口脚本都会 `sys.stdout.reconfigure(encoding="utf-8", errors="replace")`，
  正常不会出现；
* 若仍出现，说明你在自己的脚本里 `print` 了中文：加同样的一行，或设置
  `PYTHONUTF8=1`。

### 中文路径下 `joblib`/`loky` 报 pickle 错误

Windows `spawn` 模式下，worker 进程的参数含非 ASCII 偶发失败。

```bash
python run_pipeline.py ... --n-jobs 1        # 先确认是不是并行导致
```

根治：把数据与 `--out` 都放到英文路径。本包启动时会检测并在路径含非 ASCII 且
`n_jobs>1` 时打印提示。

### `ModuleNotFoundError: No module named 'sml'`

stage 脚本需要能找到 `scripts/` 目录。所有 stage 脚本已在开头做了
`sys.path.insert(0, <脚本所在目录>)`，所以直接用绝对路径调用即可：

```bash
python /path/to/research-ml-modeling/scripts/stage_cv.py --run-dir ...
```

不要 `cd` 到别的目录再用相对路径调 `python scripts/xxx.py`（会找不到 `sml`）。

---

## 配置

### `E-CFG-007 JSON 解析失败（配置不支持 // 注释）`

JSON 不允许注释。用 `_comment` 键：

```json
{"_comment": "这是我改过的配置", "task": {...}}
```

### `E-CFG-023 Savitzky-Golay window 必须为奇数`

scipy 1.18 静默接受偶数窗口，但偶数窗口的 SG 中心不唯一。改成 11/15/21 等奇数。

### `E-CFG-017 strategy 需要批次信息`

`strategy=loo_batch` 或 `holdout_batch` 必须有 `data.batch_col`。

### `--set` 不生效

值的类型是推断的：`--set split.repeats=10` 得到 int；
`--set "preprocess.candidates.0.name=raw"` 得到 str；
列表要写成 JSON：`--set 'data.drop_cols=["备注","序号"]'`（外层用单引号包住）。
Windows 下若引号被吞，改用双引号并把内部引号转义，或直接改配置文件。

### 只想从列表里删一个字段（`--set` 会整串替换）

`--set data.drop_cols=[...]` 是**整体覆盖**，漏抄一个字段就等于把它放回特征矩阵——
做"去掉 time 的敏感性运行"时漏掉 `sample_id`，就是当场制造泄漏。用 `--set-remove`：

```bash
--set-remove data.drop_cols=time
--set-remove data.drop_cols=日期 --set-remove data.drop_cols=批次
```

---

## 数据

### `E-IO-007 长表转宽表后有 N 个缺失格`

各样品波长轴不一致。先统一定标/插值，或改用宽表格式。

### `每个 group 只有 1 行`（audit 警告）

数据里没有重复测量，或 `sample_id` 没提供。

* 若确实单次测量 → 报告里不要声称"用分组 CV 避免重复测量泄漏"；
* 若有重复测量 → 补 `sample_id`，这是最关键的一列。

### `E-IO-014 目标列不在表头`

列名有前后空格、或 Excel 多级表头。检查 `list(df.columns)`。

### 特征数比预期多几个

成分列、行号列被当成特征。加到 `data.drop_cols`。audit 会专门抓"列名像 ID 且取值唯一"。

---

## 建模过程

### `ValueError: n_components upper bound is N`

直接调 `PLSRegression` 时会遇到。本包用 `SafePLS` 自动裁剪，若你看到这个错，
说明你绕过了 `models.build_estimator`。

### `TypeError: XGBModel.fit() got an unexpected keyword argument 'early_stopping_rounds'`

xgboost 3.x 的早停只能写在构造器。本包不使用早停，不会触发。

### `IndexError: index N is out of bounds`

最常见原因是"在 A 数据上算的 split 索引，用到 B 数据上"。
本包所有 split 都在对应数据上生成；自己写扩展时注意。

### 某一折的 R² 是 `nan` 或极大负数

* `nan`：测试折里 y 全相同（`SS_tot=0`），R² 无定义 → 正常，聚合时会忽略
* 极大负数：测试折样品极少（1–2 个）或分布极端 → 只看 repeat 级聚合值

### 折均值 R² 是 −0.8 这种数，池化 OOF 却是 +0.19（留一法尤其常见）

每折只有 1 个测试样品时，逐折 R² 在单点上退化（`SS_tot=0` 或只有一个残差），
`nanmean` 又会跳过这些退化折，于是"k 折的平均"可能只来自 1 个折。

* `cv/metrics_summary.csv` 看 `metric_basis`：`pooled` = 该行**折均值不可引用**，
  必须用 `r2`/`rmse`（池化 OOF）列；
* `n_folds_ok` / `n_folds_degenerate` 说明到底几个折真算出了指标；
* 报告里这些行会标 `†` 并并排给出 `r2 (池化)`；同页两个口径不要混引。

样本这么少（n < 15）本来就不该报"最优模型"，只作探索性结论。

### `全部折都失败` 但没看到具体原因

默认 `error_score=np.nan` 会吞掉异常。加 `--fail-fast` 让第一个错误直接抛出。

### `E-CFG-030`：同一个 run 目录被两套配置复用

`prepare()` 每次都会重写 `config.resolved.json`。先跑 `--set models=[pls,pcr]`，
再直接 `--config 预设 --out 同一目录` 跑下游 stage，配置会被覆盖成预设默认值——
`cv/` 里是 pcr 的结果，配置里写着 ridge，报告与结果永久对不上账。

改法三选一：换 `--out`、用 `--run-dir <目录>` 复用旧配置、或显式 `--allow-config-change`。
比较范围只有"换了就是另一个研究问题"的字段（`task` / `data.path` / `models`），
改 `repeats`/`n_perm`/`bootstrap` 不算冲突。

### `E-STAGE-003`：CV 里的最优模型不在当前配置里

通常就是上面那条的后果（配置回退成预设默认值，最优模型 `pcr` 不在 `models` 里）。
按提示用 `--run-dir` 重跑即可。

### `E-STAGE-004`：多目标结果，无法判定"最优模型"指哪个目标

```
[E-STAGE-004] CV 结果含 2 个目标（moisture, protein），无法判定「最优模型」指的是哪个目标。
```

这不是提示，是拒绝出报告：不同目标的指标尺度与难度不同，全表排序选出的是
"跨目标最强"的模型，T2 的 CI 也会被后一个目标覆盖前一个（两者都不报错）。

修法二选一：

```bash
# A. 让 run_pipeline 按目标拆成独立子 run（推荐，各自出完整报告）
$PY scripts/run_pipeline.py --config ... --data ... --out D:/runs/nir_v1

# B. 只跑单个目标
$PY scripts/run_pipeline.py --config ... --data ... --out D:/runs/nir_v1 --set 'task.target=["protein"]'
```

若你**刻意**要把多目标放进同一目录（只跑 CV、自己看表），加 `--no-fanout --stages audit,cv`；
汇总表里 `rank` 与 `gap_to_best` 已按目标分组。但同一目录不要跑 interp/perm/ad/report，
`run_pipeline` 会在开跑前以 `E-CFG-031` 拒绝，避免 CV 跑完才失败。

### `E-CFG-031`：多目标 + `--no-fanout` 还要跑下游 stage

```
[E-CFG-031] 多目标 + --no-fanout 无法正确运行 ad, interp, report
```

一个 run 目录只有一套 `figures/` 与 `report/`，多目标会互相覆盖；interp/perm/ad
只按 `target[0]` 计算，其余目标根本没有结果。修法：

```bash
# 去掉 --no-fanout，让它按目标拆子 run
$PY scripts/run_pipeline.py --config ... --out D:/runs/nir_v1

# 或只跑 CV 对照表
$PY scripts/run_pipeline.py --config ... --out D:/runs/nir_v1 --no-fanout --stages audit,cv
```

### 多目标里"兄弟目标"被当成特征了？

症状：两个子 run 的特征数只差 1 或完全一样；某个目标的 R² 明显高于其光谱信息所能支持的水平；
审计里没有"目标泄漏"告警（因为该列看起来只是个强相关特征）。

成因是两个路径之一：`roles` 只收了 `target[0]`，或 fan-out 子 run 覆写 `task.target` 后
不知道兄弟目标的存在。现在两条都堵上了（`io_data` 收全部目标 + 子 run 传
`data.exclude_from_features`）。核对方式：`audit/audit.md` 应有「目标列未进入特征矩阵」一条，
并列出被剔除的列名。

### 结果表里少了一个模型（不是"效果差"，是它整批失败了）

历史版本里这是最阴的一类 bug：搜索键写错或打分器用错 → 每一折都抛异常 →
被 `error_score` 吞掉 → 模型从 `metrics_summary.csv` 消失，只在 `fold_results.csv`
留下若干行 `status=error:XXX`，日志里一行 warn。

现在的行为：

* 控制台明确打出「模型 X 的 N/N 折全部失败，已从结果表剔除」
* `cv/cv_summary.json` 的 `failed_models` 记录失败折数、`all_failed`、首个错误原文
* 报告 `report.md` **顶部**生成告警块——因为"漏报一个模型"和"该模型很差"在论文里是两件事

排查顺序：先看 `failed_models[模型].first_error` 的 `fit/predict` 前缀（区分是训练期还是预测期），
再用 `--fail-fast` 复现单折。

### `AttributeError: ... none of the following attributes: predict_proba`

模型（典型是 `ridge_clf`）没有概率接口，但打分器用了 `neg_log_loss`。已修复：
现在按**估计器能力**选打分器（`scoring_for`），无 `predict_proba` 时自动用 `balanced_accuracy`。
若你自己加了自定义估计器，同理会自动走这条分支，不需要改配置。

### Brier / ECE / log-loss 全是 `NaN`

不是算错了，是**拒绝算**。这些指标的分母语义是"概率"，而该模型只有
`decision_function`（`proba_kind=decision`）。用 sigmoid 映射出来的分数可以排序但不是概率，
算 Brier 不会报错却会给出错误数字——宁可空着。F5 校准图同理会被跳过（有 warn 说明）。

要概率就换一个有 `predict_proba` 的模型（`logreg`/`pls`(PLS-DA)/`lda`/树模型），
或接受只报 AUC/AP/平衡准确率/MCC。

### `E-MOD-002`：搜索空间里有无法解析的参数

键名和真实估计器对不上，最常见是嵌套层级少写一层
（`model__n_components` vs `model__pca__n_components`）。报错会列出该估计器可调的参数，
并给出最接近的候选名。注意 xgboost/catboost 不会因此报错——它们会静默接受未知 kwarg，
所以这项校验必须发生在配置期。

### `E-MOD-003`：这个 kind 只能做分类

`logreg` `svc` `ridge_clf` `gnb` `lda` 没有回归版本。回归请换 `ridge` / `lasso` / `svr` /
`knn` / `rf` / `gbr` / `pcr`。

### 搜索候选数不等于配置里的组合数

* 预处理候选 >1 时，`prep__which` 会乘进组合数
* `grid` 类型上限 240 组，超出会随机抽样
* budget 会把 `n_iter` 下调（取 `min(配置值, 档位值)`）
* `--dry-run` 会打印每折实际候选数

### 内存暴涨 / 机器卡死

1. 并行只发生在独立任务级（候选 / 折 / 置换 / 波段），不要再在外层自己套一层 `n_jobs=-1`
2. `--n-jobs 2`
3. `permutation.n_perm`、`evaluation.bootstrap.n` 减半
4. 光谱 5000 点以上时考虑 `preprocess` 里加 `band` 裁剪
5. `runtime.save_models="last"`（默认），不要用 `all`

### 训练很久没有输出

`--quiet` 会关掉大部分输出。去掉它，或看 `_stages.json` 里的 `done_at`。

### 完整运行动辄几十分钟

按耗时从大到小排查：

1. **SVC 的 `probability=True`** 内部做 Platt 校准（又一层 CV），在 700+ 维光谱上极慢。
   光谱分类建议删掉 `svc`，用 `logreg` / `plsda` / `lgbm`。
2. **集成树 `n_estimators`**：600 棵树 × 754 特征 × 每折 15 个候选 × 3 内层折 × 15 折，
   乘起来就是几十万次拟合。`--budget fast` 会把 `n_iter` 压到 15。
3. **预处理候选数**：每条候选链都乘进搜索空间，5 条就要 5 倍时间。
4. **`evaluation.bootstrap.n`**：2000 次在 n<300 时只要几秒，不是瓶颈。
5. **`permutation.n_perm`**：fast 模式成本 = n_perm × repeats × 外层折数。

先用 `--dry-run` 看"拟合次数"和"拟合耗时下限"，再决定砍哪一项。`--dry-run` 会在子样本
（≤400×400）上真跑一次冷启动拟合再按行数×列数外推，所以它给的是**下限**：interp 的带宽置换、
报告绘图和 joblib 调度都不在里面，实际通常再高 2–10 倍；如果本 run 目录已经跑过，它会直接回读
`_stages.json` 里上次的实测秒数作为参照。

### 预处理明明只有几条，却占了大部分时间

airPLS 单行成本是 SNV/SG 的几十倍，而旧实现还会在每个候选、每个内层折上重复做预处理。
现在折级预处理结果会缓存（无监督算子，与 y 无关，所以复用不改变任何数值）：

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `SML_PREP_CACHE_MB` | 96 | 折级预处理输出缓存上限（MB） |
| `SML_PREP_CACHE` | 开 | 设 `0`/`off` 关闭缓存（排障时可用来对照"原始路径"的耗时） |
| `SML_PERM_CACHE_MB` | 1500 | 置换阶段每折变换后矩阵的缓存上限（MB），超限自动退回逐步计算 |
| `SML_CACHE_DIR` | `%TEMP%\sml-dfcache` | 数据解析缓存目录（键含文件全内容 sha256 + sheet + 编码 + pandas 版本） |

需要干净对照时：`set SML_PREP_CACHE=0` 再跑，其余参数不动。

---

## 产物

### `stage_report` 报"缺少 OOF 预测"

先跑 `stage_cv`。stage 之间靠文件串联，不能跳步（audit → cv → interp/perm/ad → report）。

注意 `stage_cv.py` 的 `--stages` 参数属于 `run_pipeline.py`；单独跑 stage 脚本时不要传
（会报 `unrecognized arguments`）。单独跑某个 stage 的正确写法：

```bash
python stage_cv.py --config <配置> --out <run目录>          # 重新拟合
python stage_report.py --run-dir <run目录>                  # 只重出报告/图
python rebuild_tables.py --run-dir <run目录>                # 只重算汇总表，不重新拟合
```

### 只想改汇总口径 / bootstrap 次数，不想重跑 CV

```bash
python rebuild_tables.py --run-dir <run目录> --bootstrap 5000
```

它只读 `cv/oof_predictions.csv` 与 `cv/fold_results.csv`，重算
`metrics_summary.csv`、`metrics_by_prep.csv`、`metrics_ci.csv`、`comparison.csv`。
之后再跑 `stage_report.py --run-dir ...` 让报告与图同步。

重算同样会判"折均值 / 池化"口径并发出警告（与 stage_cv 共用 `mark_metric_basis`）。
但**旧版本产出的 run** 里 `fold_results.csv` 可能没有 `status` 列，这时无法判退化折，
会如实退回折均值口径——想要正确的池化数字，得重跑 `stage_cv`。

### 用训练好的模型预测新样品 / 画工艺窗口

```bash
python predict.py --run-dir <run>/targets/01_<目标> --retrain          # 只重建最终模型
python predict.py --run-dir <run>/targets/01_<目标> --data new.csv     # 预测（含判域）
```

模型由 `stage_cv` 自动存在 `model/final_model.joblib`，超参与选模记录在
`model/model_card.json`。`--data` 的列名必须与训练特征一致，缺列报 `E-PRED-002`
并列出前 10 个缺失列；输出带 `prediction` 与 `ad_flag`（`ad_flag=1` 的行在适用域外，
只能作参考）。不要自己 `import sml` 手搓——`model__` 前缀的超参、预处理链、
中位数插补与判域调用都得和这一层对齐。

### 图里中文变成方块

字体缺失或 mathtext 未生效。

* `selfcheck` 的"matplotlib 中文字体"一项会报具体字体名
* 上标/希腊字母用 mathtext（`$R^2$`、`cm$^{-1}$`），不要用 Unicode 上标字符

### `xlsx` 写出失败

`xlsxwriter` 未装或文件被 Excel 占用。CSV 仍然完整可用。

### `--resume` 没有跳过

说明三哈希有一个变了。日志会打印"数据文件已变化 / 配置已变化 / 脚本代码已变化"。
改了任意一个 `.py`（哪怕只是注释）都会让 `code_sha256` 变——这是故意的。

---

## 统计学层面的"异常"

### 域外数字和 `ad_eval.json` 里的 `n_outside` 对不上

不是 bug，是两套口径：

| 字段 | 单位 | 用途 |
|---|---|---|
| `n_outside_groups` / `flag_rate_groups` | **独立样品** | 论文正文 |
| `n_outside` / `flag_rate` | OOF **行**（含重复测量 × 重复次数） | 追溯逐行标记 |

60 样品 × 3 次测量 × 2 次重复 = 360 行，行数是样品数的 6 倍。报告与 Methods
一律用 group 口径，行口径只作备查。

### 正文里出现了"不得宣称…""必须写入 limitations…"

这类是**写给作者的指令**，不是论文句子。新版已把它们分离到 `author_note` 字段，
只出现在报告的"写作提示（给作者，不要抄进论文）"块和 `reviewer_checklist.md` 里。
若你的旧 run 仍混在正文，重跑 `stage_ad` + `stage_report`。

### `metrics_ci.csv` 的数值和图 F1 的注记不一致

表是 repeat 口径（先折内均值、再跨 repeat），图是 OOF 池化口径。
差千分位属正常，同页引用时注明口径即可（见 `reporting-and-figures.md`）。


### Y-randomization 的零分布均值明显大于 0

说明**流程本身有泄漏**（不是模型好）。排查：

1. 预处理是否在 CV 外 fit
2. 是否用了全量数据的统计量（均值、分位数、特征筛选）
3. 分组是否正确

真实数据上，合法的零分布均值应接近 0 或为负。

### 域内外性能没有显著差异

不是 bug，是 AD 没起作用。写进 limitations，不要说"模型可外推"。
可尝试：增大 `ad.pca_max_comp`、改 `quantile`、或换方法组合。

### 两个模型"看起来差很多"但 p 很大

n 小或折间波动大。看 `cv/comparison.csv` 的 CI 宽度；不要只看点估计。
