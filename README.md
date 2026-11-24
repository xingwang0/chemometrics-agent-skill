# research-ml-modeling

**审稿级机器学习预测建模**：把「原始表格 → 数据审计 → 嵌套分组交叉验证 → 置换检验 → 适用域 → 论文级图表 + Methods 草稿 + 审稿质疑清单」固化成一条命令。

面向材料、化学、光谱（NIR / FTIR / Raman / XRD / XPS / UV-Vis）与配方工艺参数等表格数据的回归与分类。也适用于任何"样本少、维度高、同一批次/同一样品测了多次"的实验数据。

---

## 它解决什么

化学计量学论文最容易死在三个地方，而且都死在方法学，不是死在模型：

1. **同一样品测了多次，随机划分就泄漏**——R² 虚高，审稿人一问"你的分组键是什么"就答不上来
2. **调参和评估用的是同一批折**——R² 是调出来的，不是测出来的
3. **没有不确定度、没有适用域**——"新样品能不能用"无法回答

这套管线把每个漏洞都做成了**会拦你的 Gate**，而不是写在文档里的建议：

| Gate | 触发后果 |
|---|---|
| 审计发现目标泄漏 / ID 当特征 | 报 `critical`，必须先修才能往下走 |
| 置换检验 p ≥ 0.05 | 报告强制写明「不能排除拟合噪声」 |
| 适用域被判定无效 | 禁止宣称模型可外推，写入 limitations |
| 最优模型与次优差异不显著 | 全文只能写「性能相当」，不许写「X 优于 Y」 |

## 快速开始

```bash
# 下载或克隆本仓库后，进入仓库根目录执行以下命令

# 1) 先验环境：34 项自检（换机必跑）
python scripts/selfcheck.py

# 2) 再跑通：30 秒冒烟，用合成数据走完全流程
python scripts/run_pipeline.py --config assets/config/smoke.json \
    --out runs/smoke --budget fast

# 3) 换成你的数据
python scripts/run_pipeline.py --config assets/config/regression_spectra.json \
    --data your_data.xlsx --out runs/v1 --budget standard \
    --stages audit,cv,interp,perm,ad,report
```

`selfcheck` 全绿之前不要碰真实数据——它失败的方式不是报错，而是给你一份看起来很漂亮的错数字。

## 四个预设

| 配置 | 场景 |
|---|---|
| `regression_spectra.json` | 光谱回归（700+ 维，需 SNV/MSC/SG/airPLS） |
| `regression_tabular.json` | 普通表格回归（几十列，DoE / 工艺参数） |
| `classification_tabular.json` | 分类（合格/不合格、相识别），不做 SMOTE |
| `smoke.json` | 合成数据冒烟，验证安装 |

`data.path` 留空时必须用 `--data` 指定文件；也可填相对本文件所在目录的路径。

改参数用 `--set a.b=c`，删列表元素用 `--set-remove`（**不要**用 `--set` 重写整串，漏抄一个字段就泄漏）。字段全表见 `references/config-schema.md`。

## 六个阶段

```
audit → cv → interp → perm → ad → report
```

| 阶段 | 产物 | 关键约束 |
|---|---|---|
| `audit` | `audit/audit.md` | 目标泄漏、ID 当特征 |
| `cv` | `cv/oof_predictions.csv`（唯一真源） | 嵌套分组重复 CV |
| `interp` | VIP / 波段重要性 / 系数稳定性 | — |
| `perm` | `perm/permutation.json` | 样品级置换 y |
| `ad` | `ad/ad_eval.json` | 四法判域 + 有效性检验 |
| `report` | 8 张图 + 10 张表 + Methods 草稿 + 审稿清单 | 每个数字可回溯到 OOF |

## 严谨性红线

**必须做**：按 `sample_id` 分组 CV、嵌套 CV 分离调参与评估、重复 CV 报 mean±sd、按组 bootstrap、Y-randomization、PLS 基线 + RPD/RPIQ、适用域 + 有效性验证、预处理进搜索空间、AD 报独立样品数。

**禁止做**：随机 KFold 报论文结果、用测试折选超参、按行 bootstrap、按行置换 y、只报 GBDT 不报 PLS、剔除域外样本后再报性能、把 OOF 行数当样品数（60 样品 × 3 重复 × 2 repeat = 360 行，会放大 6 倍）。

完整 57 条见 `references/reviewer-checklist.md`；每次运行自动生成带 ✓/⚠ 的 `report/reviewer_checklist.md`。

## 产出物

```
runs/<id>/
├── config.resolved.json  provenance.json      ← 三哈希，改一行代码就失效重跑
├── audit/audit.md                             ← 先看这个
├── cv/     oof_predictions.csv · metrics_summary.csv · metrics_ci.csv · comparison.csv
├── perm/permutation.json
├── ad/     ad_eval.json · ad_model.joblib
├── model/  final_model.joblib · model_card.json
├── figures/ F1..F8  png(300dpi) + pdf(矢量)
├── tables/  T1..T6  csv + xlsx
└── report/  report.md · methods_zh.md · methods_en.md · reviewer_checklist.md
```

多目标自动按目标拆成独立子 run，**兄弟目标列会从特征里剔除**——否则会用「蛋白」预测「水分」，分数虚高且审计难以识别。

## 数据要求

1. 一行一次测量；`sample_id` 能标识"同一份材料/样品"
2. 角色列（`id / batch / replicate / target`）不进特征矩阵，管线自动剥离
3. 光谱列名写成波数（`4000`、`4008`…），画图和波段分组才不用额外配置
4. 目标缺失行剔除；特征缺失 >50% 的列丢弃；其余列中位数插补（审计会警告）
5. 中文路径可用，但 `--out` 建议 ASCII

## 已知边界

- 不做深度学习（依赖矩阵里没有 torch）
- 不做统计假设检验（ANOVA / t 检验）
- 不做纯图像分割、NLP
- 独立样品 < 15 时只做探索，不下结论

## 常见坑

- `--budget` 只接受 `fast` / `standard` / `thorough`；`smoke` 是**配置名**不是预算档
- `--resume` 按三哈希判定：改任何一个 `.py`（哪怕注释）都会让产物失效重跑，这是故意的
- 中文路径下多进程报 pickle 错，加 `--n-jobs 1`
- 报告里的数字对不上，先怀疑自己手改了表——所有数字都能从 `cv/oof_predictions.csv` 重算

## 许可

Apache License 2.0，见 `LICENSE` 与 `NOTICE`。允许商用与修改，须保留版权声明并标注改动；不授予商标权；AS IS，不提供任何担保。

**全部输出（指标表、图表、Methods 草稿、审稿清单）都是计算产物，投稿前必须人工核对**样品来源、仪器型号与测量条件、指标口径。

本项目**未针对**监管申报（ICH / USP / CLSI / GB）、临床诊断、生产实时控制等场景做过验证，不要把管线输出直接当作上述用途的判定依据。上述免责受适用法律约束；中国法下依法不得排除的责任（故意或重大过失致财产损失）不因本声明而免除。

引用时请注明版本号与对应 run 目录下 `provenance.json` 的三哈希。
