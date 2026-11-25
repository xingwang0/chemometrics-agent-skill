# 本机环境与 API 行为（实测）

换机器、升级包之后，**先跑 `python scripts/selfcheck.py`**，它会逐项验证下面这些结论。

---

## 1. 解释器与依赖

```
python --version   → Python 3.14.7（本包验证所用版本；路径按你本机）
```

| 包 | 版本 | 用途 |
|---|---|---|
| numpy | 2.5.3 | |
| pandas | 3.0.6 | |
| scikit-learn | 1.9.1 | 划分器、模型、搜索 |
| scipy | 1.18.1 | SG、稀疏求解、统计检验 |
| xgboost | 3.4.1 | |
| lightgbm | 4.7.0 | |
| catboost | 1.2.10 | |
| shap | 0.54.1 | 仅补充解释 |
| matplotlib | 3.11.2 | 全部论文图 |
| seaborn | 0.13.2 | 可选 |
| joblib | 1.6.0 | 模型落盘 |
| openpyxl / xlsxwriter | 3.1.5 / 3.2.9 | Excel 读写 |
| tqdm | 0.70.1 | 可选进度条 |

**未安装**（设计上就不依赖）：

| 缺失 | 替代方案 |
|---|---|
| PyYAML | 配置一律 JSON |
| statsmodels | FDR 用 `scipy.stats.false_discovery_control`；校准曲线/Brier 自写 |
| optuna | 用 `GridSearchCV` + 自写随机采样（`tune.expand_space`） |
| imbalanced-learn | 用 `class_weight` + 阈值调优，不做 SMOTE |
| torch | 不支持深度学习 |
| jinja2 | Methods 模板用 `string.Template` |
| kaleido | plotly 只能导 HTML；论文图全走 matplotlib |
| graphviz(二进制) | 流程图用 mermaid 文本 |

---

## 2. 已实测的版本行为（这些是硬约束）

### sklearn 1.9

| 行为 | 结论 |
|---|---|
| `GroupKFold(n_splits, shuffle=True, random_state=...)` | **支持**（旧版没有 shuffle）。重复分组 CV 因此不需要自己造划分 |
| `StratifiedGroupKFold(..., shuffle=True, random_state=...)` | **支持** |
| `GridSearchCV(cv=<显式 split 列表>)` | 可正常 fit + refit + predict；**嵌套 CV 靠这个落地** |
| `cross_validate(..., groups=)` | 可用，但 metadata routing 默认关闭，全项目不要启用 |
| `SVC(probability=True)` | 1.9 起**弃用**（1.11 移除）。本机仍可用；升级时改用 `CalibratedClassifierCV(SVC(), ensemble=False)`；本包已静音该 FutureWarning |
| `LogisticRegression(penalty=)` | 1.9 起**弃用**、1.11 移除。**本包已不使用 penalty**：L1 用 `l1_ratio=1.0`、弹性网用 `l1_ratio=0.5`（配 `solver="saga"`）。若你自己的配置里仍传 `penalty`，会得到 `Inconsistent values: penalty=l1 with l1_ratio=0.0` 告警，删掉即可 |
| `permutation_importance` | 新增 `max_samples`；本包未依赖 |
| `LinearRegression(normalize=)` | 早已移除，不要照抄旧教程 |

### xgboost 3.4.1

```python
# 正确：早停写在构造器
XGBRegressor(n_estimators=500, early_stopping_rounds=50).fit(X, y, eval_set=[(Xv, yv)])
# 错误：TypeError: fit() got an unexpected keyword argument 'early_stopping_rounds'
```

内层搜索不传 `eval_set`，因此本包**不使用早停**，直接固定 `n_estimators` 网格。

### lightgbm 4.7

* `verbose=-1` 才能静默；否则每个模型都刷一行
* 早停走 `fit(callbacks=[lgb.early_stopping(50, verbose=False)])`

### catboost 1.2.10

* **必须** `allow_writing_files=False, verbose=0`，否则会在当前目录建 `catboost_info/`
* 分类用 `auto_class_weights="Balanced"`

### scipy 1.18

| 行为 | 结论 |
|---|---|
| `savgol_filter(window_length=10)`（偶数） | **静默接受**（旧版报错）→ 本包在配置校验阶段直接拒绝偶数 |
| `stats.false_discovery_control(p, method="bh")` | 存在 |
| `stats.bootstrap` | 存在（本包用自己的 cluster bootstrap，不用它） |
| `solveh_banded`（上带三行布局） | airPLS 用它解惩罚最小二乘：`lam·DᵀD` 压成 5 对角带状，单行约 3.8 ms（1200×2000 实测）。注意上带布局 `ab[kd]` 是主对角、`ab[kd-1,1:]` 是一阶上带、`ab[kd-2,2:]` 是二阶上带——记错会报 `leading minor is not positive definite` |
| `sparse.linalg.spsolve` | 保留为兜底：`lam<=0` 或带状求解抛 `LinAlgError` 时回退稀疏版 |

### pandas 3.0

| 行为 | 结论 |
|---|---|
| Copy-on-Write 强制开启，`pd.set_option("mode.copy_on_write", ...)` 抛 `Pandas4Warning` | **代码里禁止设置该选项** |
| 字符串列默认 dtype 是 **`str`**（不再是 `object`） | 禁用 `dtype == object` 判断，改用 `pd.api.types.is_string_dtype` |
| 链式赋值不再生效 | 中间表显式 `.copy()` 或函数式构造 |
| `df.to_numpy(copy=True)` | 取特征矩阵时显式拷贝 |

### numpy 2.5

* `np.float_` 已移除 → 用 `float` / `np.float64`
* `np.add.at` 仍可用于分组累加（本包 bootstrap 用到）

### matplotlib 3.11

* `Microsoft YaHei` / `SimHei` / `Source Han Sans SC` / `DengXian` 均在字体表内
* 中文 `savefig` 无缺字警告
* **上标 `⁻¹` 等特殊字符可能缺字形** → 用 mathtext：`r"cm$^{-1}$"`
* 后端统一 `Agg`（无 GUI 也能跑）

---

## 3. 复现锁定

`provenance.json` 记录：

* `python` / `platform` / `executable`
* `packages`：15 个核心包的版本快照
* `seeds`：master + 每个 repeat 的种子 + 每个 fold 的种子
* `hashes`：`data_sha256` / `config_sha256` / `code_sha256`
* `timing_seconds`：各 stage 耗时
* `total_fit_calls`：实际拟合次数

`--resume` 用三哈希判定产物是否可复用；三者任一变化都会重跑，并打印是哪一项变了。

---

## 4. 换机器时的检查清单

1. `python scripts/selfcheck.py` —— 53 项全绿再动手（含 env / leakage / registry / multitarget / reporting / multiclass / provenance / split / present / figure_qc / standards 十一组）
2. 看 `内存探测可用` 一项；`runtime.mem_guard_gb` 按实际内存调
3. 若 Python < 3.11：`config.py` 的类型注解（`str | Path`）会报错，需改 `Union`
4. 若 sklearn < 1.6：`GroupKFold` 无 `shuffle`，`cv.py` 会走 KD 降级路径（selfcheck 会提示）
5. 若无中文字体：报告里会把文种降到英文，或手动装思源黑体
