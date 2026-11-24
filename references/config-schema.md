# 配置字段速查

配置文件是 **JSON**（本机没有 PyYAML）。不支持 `//` 注释；需要注释时用 `_comment` 键，
所有以 `_` 开头的键在解析后会被剔除。

覆盖优先级：**命令行 > `--set` / `--set-remove` > budget 档位 > 配置文件 > 内置默认值**。

配置文件里写死的预算项（`repeats`、`n_perm`、`bootstrap.n`、`n_iter`）一旦被 `--budget`
改写，程序直接报 `E-CFG-032` 并列出被改的键——不会静默换掉你写的值。

---

## 命令行

```bash
python run_pipeline.py --config <json> [--data <表>] [--out <目录>]
    [--budget fast|standard|thorough] [--seed N]
    [--stages audit,cv,interp,perm,ad,report]
    [--set a.b.c=值]... [--set-remove a.b=元素]... [--n-jobs N]
    [--resume] [--dry-run] [--fail-fast] [--quiet]
```

| 参数 | 作用 |
|---|---|
| `--dry-run` | 只做审计 + 打印"将拟合多少次、预计多久"，不训练。**先跑这个** |
| `--resume` | 三哈希（数据/配置/代码）一致且产物完整时跳过已完成 stage |
| `--n-jobs` | 只作用于内层并行；外层永远串行（防嵌套并行爆内存） |
| `--fail-fast` | 任一折出错立即中断（默认记 NaN 继续，并在 `fold_results.csv` 里留 `status`） |
| `--run-dir` | 单独跑某个 stage 时指向已有 run 目录（此时不需要 `--config`） |
| `--set-remove` | 从列表/字典里**删**一个元素（可多次），不用把整串重写一遍 |

`--set` 语法：点号路径，支持列表下标，值自动推断类型（数字/布尔/null/JSON）。

```bash
--set split.repeats=10
--set models.1.n_iter=60
--set "preprocess.candidates.0.name=raw"
--set 'data.drop_cols=["备注","序号"]'
```

**列表列一律用 `--set-remove` 删元素，不要用 `--set` 重写整串**：`--set` 对列表是整体
替换，漏写一个字段就等于把它从"排除清单"里放回特征矩阵——比如做「去掉 time 的敏感性
运行」时漏掉 `sample_id`，就是当场制造泄漏。

```bash
--set-remove data.drop_cols=time                # 只删 time，其余原样保留
--set-remove data.drop_cols=日期 --set-remove data.drop_cols=批次
--set-remove task.target=moisture_copy          # 多目标里剔掉一个目标
```

`--set-remove` 同样做开跑前校验：路径不存在、下标越界、目标不是列表/字典，分别报
`E-CFG-002` / `E-CFG-003` / `E-CFG-004`。

---

## task

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `type` | `regression`\|`classification` | `regression` | 任务类型 |
| `target` | list[str] | `["y"]` | 目标列；回归可多列（逐目标建模），分类只能 1 列 |
| `positive_label` | str\|null | null | 分类的阳性类；null 时取排序后最后一个 |
| `label_order` | list[str]\|null | null | 显式指定类别顺序（避免排序把"合格/不合格"颠倒） |

### 多分类（类别数 > 2）

二分类是默认路径，但三类及以上也支持（已由自检 `multiclass` 组端到端覆盖）：

| 环节 | 多分类行为 |
|---|---|
| OOF 产物 | `cv/oof_predictions.csv` 写出 `p0..pN` 逐类概率；`proba` 列此时是**最大类概率**，只用于展示，不参与指标 |
| 主指标 | AUROC 为 macro 的 one-vs-rest；Brier 为多类平方和 `mean(Σ(p−onehot)²)`；log-loss 用多类交叉熵；敏感度/特异度取各类宏平均 |
| 图 | F1 画逐类 OVR ROC（宏平均 AUROC 标注）；F5 画逐类 OVR 可靠性曲线（宏平均 ECE）；F2 残差图只对回归有意义，分类自动跳过 |
| 概率归一 | 指标计算前逐行归一到 1（避免 float32 概率行和偏离 1 触发 sklearn 警告、并让 Brier/ECE 分母语义正确） |
| `label_order` | 多分类必须给全（少给一个类会报 `E-IO-009`；不给则按排序） |
| 限制 | Youden 阈值调优、`calibration_curve` 的二分类形态、`proba_positive` 输出列都只对二分类有意义；多分类请用 `p0..pN` |

---

## data

| 字段 | 默认 | 说明 |
|---|---|---|
| `format` | `wide` | `wide` 或 `long`（长表三列） |
| `path` | — | 数据文件；相对路径按**配置文件所在目录**解析 |
| `sheet` | `0` | Excel 工作表名或索引 |
| `encoding` | `utf-8` | CSV 编码；中文 Windows 常见 `gb18030` |
| `sep` | null | CSV 分隔符；null 自动嗅探 |
| `id_col` | null | **分组键**（sample_id）。null 时按命名约定猜，再退化为特征指纹 |
| `batch_col` | null | 批次/仪器/日期，用于留一批次外部验证 |
| `replicate_col` | null | 同一样品第几次测量 |
| `feature_mode` | `all_numeric_except_roles` | 特征选择方式 |
| `feature_cols` | null | 显式指定特征列（优先于自动识别） |
| `drop_cols` | `[]` | 要从特征里剔除的列（备注、序号、派生量…） |
| `exclude_from_features` | `[]` | 额外剔除的列。**多目标 fan-out 时由 `run_pipeline` 自动填入全部目标列**（子 run 的 `task.target` 只剩一个，若不带这项，兄弟目标会留在特征里） |
| `long_cols` | `{id,x,y}` | 长表三列的列名 |
| `wavelength_axis` | null | `{start, stop, step, unit}`；列名不是数值时才需要 |
| `na_values` | `["", "NA", "N/A", ...]` | 视为缺失的取值 |
| `min_valid_frac` | `0.5` | 列非缺失比例低于此值则整列丢弃 |

---

## split

| 字段 | 默认 | 说明 |
|---|---|---|
| `strategy` | `nested_group` | `nested_group` / `loo_batch` / `holdout_batch` / `holdout_random` |
| `group_col` | null | null 时用 `data.id_col` |
| `outer.n_splits` | 5 | 外层折数；样品数不足时自动降级并警告 |
| `outer.shuffle` | true | sklearn 1.9 的 GroupKFold 支持打乱 |
| `inner.n_splits` | 3 | 内层折数（调参用） |
| `repeats` | 5 | 重复次数（不同随机种子） |
| `stratify` | true | 分类且每组单类时启用分层分组划分 |
| `holdout_batch` | null | `strategy=holdout_batch` 时必填 |
| `test_size` | 0.2 | 仅 `holdout_random` 用 |

**budget 会覆盖**：`repeats`、`inner.n_splits`、`permutation.n_perm`、
`evaluation.bootstrap.n`、各随机搜索模型的 `n_iter`（取较小者）。

| budget | repeats | inner | n_iter | n_perm | bootstrap |
|---|---|---|---|---|---|
| fast | 3 | 3 | 15 | 100 | 500 |
| standard | 5 | 3 | 30 | 200 | 2000 |
| thorough | 10 | 4 | 60 | 1000 | 5000 |

---

## preprocess

| 字段 | 默认 | 说明 |
|---|---|---|
| `candidates` | `[{name:"raw",steps:[]}]` | 预处理候选链，**每条链是一个离散超参**，建议 3–5 条 |
| `candidates[].name` | — | 出现在结果表的 `prep` 列，取有信息量的名字 |
| `candidates[].steps[].op` | — | `snv` / `msc` / `sg` / `baseline_poly` / `airpls` / `band` / `normalize` / `diff` |
| `scale` | true | 是否在预处理后接 StandardScaler |
| `scale_with_mean` | false | 光谱建议 false（已中心化，且省一次拷贝） |
| `stratify_sampling` | false | **可选开关**：`true` 时 random 搜索按 `prep__which` 分层均分 `n_iter`（每层至少 ⌊n_iter/层数⌋），让"预处理作为超参"的覆盖变均匀。默认关=旧抽样逐位不变；打开会改候选集合（grid 搜索只告警不改）。参见 `E-CFG-041` |

算子参数见 `spectral-preprocessing.md`。

---

## models

```json
{"name": "xgb", "kind": "xgb", "search_type": "random", "n_iter": 40,
 "search": {"max_depth": [3,4,5], "learning_rate": {"dist":"loguniform","low":0.01,"high":0.3}},
 "fixed": {"subsample": 0.8},
 "required": false}
```

| 字段 | 说明 |
|---|---|
| `name` | 结果里的模型名（必须唯一，重名会自动加后缀） |
| `kind` | 见下表 |
| `search_type` | `grid`（全组合，上限 240）/ `random`（抽 `n_iter` 组）/ `none`（不调参） |
| `search` | `参数名: [候选值...]` 或 `{"dist":"int|loguniform|uniform|choice", ...}` |
| `fixed` | 固定参数（不进搜索），优先级高于 `search` |
| `required` | 仅标记用；PLS 无论怎么配都会被强制保留 |

**kind × 任务矩阵**

| kind | 回归 | 分类 | 说明 |
|---|---|---|---|
| `pls` | ✓ | ✓（PLS-DA，有概率） | 强制基线 |
| `pcr` | ✓ | ✓（PCA+LogReg，有概率） | `n_components` 折内自动裁剪 |
| `ridge` `lasso` `enet` `knn` `rf` `et` `gbr` `xgb` `lgbm` `cat` `svr` | ✓ | ✓ | `svr` 分类下为 SVC；`lasso`/`enet` 分类下的超参是 `C` |
| `gp` | ✓ | ✓（Laplace，无 `alpha`） | O(n³)，`n>500` 慎用 |
| `logreg` `svc` `ridge_clf` `gnb` `lda` | ✗ `E-MOD-003` | ✓ | 只有分类语义 |

**搜索键写法**

| 情形 | 正确 | 会被拦下 |
|---|---|---|
| 普通参数 | `{"alpha": [...]}` → `model__alpha` | `{"model__alph": [...]}` |
| 嵌套估计器内部参数 | 写完整路径，如 `model__pca__n_components` | 漏一层：`model__n_components` |
| 已带前缀 | `{"model__C": [...]}` 原样使用 | — |

键在 `resolve_models` 阶段用 `set_params` 逐个试探（报 `E-MOD-002`，附最接近的候选名）。
用 `set_params` 而非查 `get_params()` 的原因：catboost 的 `l2_leaf_reg` 能设置但不出现在
`get_params` 里（已加例外表）；反过来 xgboost 会静默吞掉未知 kwarg，所以两者都要查。

`pls` 在分类任务下自动切换为 PLS-DA（概率接口），回归任务下为 PLS 回归。

---

## evaluation

| 字段 | 默认 | 说明 |
|---|---|---|
| `metrics` | null | null 时按任务取默认集合 |
| `bootstrap.n` | 2000 | 按样品重采样次数 |
| `bootstrap.alpha` | 0.05 | 1-alpha 即 CI 水平 |

回归默认：`r2 rmse mae bias rpd rpiq ccc slope`
分类默认：`auc balanced_accuracy f1_macro mcc brier sensitivity specificity`

---

## permutation

| 字段 | 默认 | 说明 |
|---|---|---|
| `enabled` | true | Y-randomization 开关 |
| `n_perm` | 200 | 置换次数；n_perm=200 时最小可达 p=1/201≈0.005 |
| `mode` | `fast` | `fast`（固定超参，便宜）/ `full`（重跑嵌套，贵） |
| `alpha` | 0.05 | 判定阈值 |

---

## ad

| 字段 | 默认 | 说明 |
|---|---|---|
| `enabled` | true | |
| `methods` | 四法全开 | `pca_t2` `pca_q` `knn` `leverage` |
| `quantile` | 0.95 | 阈值取训练集经验分位 |
| `alpha` | 0.05 | 仅用于 T² 的 F 理论阈值（参考值） |
| `pca_variance` | 0.95 | 主成分累计方差目标 |
| `pca_max_comp` | 10 | 主成分数上限 |
| `knn_k` | 5 | kNN 的 k（在小样本下会自动收缩） |
| `combine` | `any` | `any`（任一越界）或 `majority` |

---

## interp

| 字段 | 默认 | 说明 |
|---|---|---|
| `enabled` | true | |
| `perm_repeats` | 10 | 每个波段的置换重复次数 |
| `band_size` | 25 | 每个波段包含的连续波长数 |
| `top_k` | 20 | 报告里展示前 k 个波段 |
| `vip` | true | 是否算 PLS VIP（含 bootstrap 分位） |
| `coef_stability` | true | 跨折系数符号一致性 |
| `shap` | `auto` | `auto`（仅树模型）/ `on` / `off` |

---

## report

| 字段 | 默认 | 说明 |
|---|---|---|
| `lang` | `["zh","en"]` | Methods 草稿输出语种 |
| `dpi` | 300 | 图片 DPI（PDF 为矢量，不受影响） |
| `formats` | `["png","pdf"]` | 输出格式 |
| `title` | null | 报告副标题 |
| `authors` | null | 预留 |

---

## runtime

| 字段 | 默认 | 说明 |
|---|---|---|
| `n_jobs` | -1 | 内层并行进程数 |
| `seed` | 42 | 主种子（派生规则见 `provenance.py`） |
| `mem_guard_gb` | 0 | >0 时启动前检查可用内存，不足直接报错（比跑到半夜 OOM 好） |
| `save_models` | `last` | `none` / `last`（每 repeat 最后一折）/ `all` |
| `seed_scheme` | `legacy` | **可选开关**：`legacy` 时 fold 种子 = `(seed_r*100+k)*10 + 模型序`，模型数 ≥10 会跨折撞种子；`per_model` 改成 `*100 + 模型序` 并在 `provenance.seeds` 写入 `per_fold_model`。默认 legacy=旧结果逐位不变；打开会改抽样序列。参见 `E-CFG-040` |
| `verbose` | true | |
| `fail_fast` | false | 单折失败是否中断 |

---

## 错误码

| 码 | 含义 |
|---|---|
| `E-CFG-001~009` | `--set` 语法、路径、JSON 解析错误 |
| `E-CFG-010~013` | schema / 任务类型 / 目标列配置错误 |
| `E-CFG-014~015` | models 为空 / 缺 PLS（后者仅警告） |
| `E-CFG-016~019` | split 策略与折数错误 |
| `E-CFG-020~024` | 预处理候选与算子参数错误（含 SG 偶数窗口） |
| `E-CFG-025~026` | AD 方法名 / stage 名非法 |
| `E-CFG-030` | 同一 run 目录被另一套 task/data/models 复用（会覆盖 config.resolved.json） |
| `E-CFG-031` | 多目标 + `--no-fanout` 却要跑 interp/perm/ad/report（单目录无法容纳多个目标） |
| `E-CFG-032` | 配置文件里的预算项（repeats/n_perm/…）被 `--budget` 改写（附被改的键与原值） |
| `E-CFG-040` | `runtime.seed_scheme` 不是 `legacy` / `per_model`（默认 legacy） |
| `E-CFG-041` | `preprocess.stratify_sampling` 不是布尔值（默认 false） |
| `E-MOD-001` | 该 kind 无法在当前任务下构造 |
| `E-MOD-002` | 搜索空间里有无法被估计器解析的参数（附最接近的候选名） |
| `E-MOD-003` | 分类专用 kind 被用于回归任务 |
| `E-IO-001~007` | 文件不存在、格式不支持、长表对齐失败 |
| `E-IO-009~012` | 标签顺序 / 类别数 / 编码错误 |
| `E-IO-013~021` | 列角色、目标缺失、特征缺失 |
| `E-IO-023` | `data.exclude_from_features` 里的列不存在 |
| `E-CV-001~003` | 分组数不足、批次数不足、holdout 批次不存在 |
| `E-STAGE-001~003` | 上游产物缺失、主指标缺失、模型名不在配置中（附"漏了 --set"的排查提示） |
| `E-STAGE-004` | 多目标 CV 结果未指定 target 就要出报告（会跨目标挑最优、CI 串目标） |
| `E-STAGE-005` | 指定的 target 不在 CV 结果里 |
| `E-PRED-001` | 缺 `model/final_model.joblib`（先跑 stage_cv，或 `predict.py --retrain`） |
| `E-PRED-002` | 新样品表缺训练特征（会列出前 10 个缺失列名） |
| `E-PRED-003` | `predict.py` 既没给 `--data` 也没给 `--retrain` |
| `E-PRED-004` | run 目录下没有 `config.resolved.json`，无法重建最终模型 |

---

## 生成一份带注释的默认配置

```bash
python -c "import sys;sys.path.insert(0,'scripts');from sml.config import config_defaults_doc;import json;print(json.dumps(config_defaults_doc(),ensure_ascii=False,indent=2))"
```
