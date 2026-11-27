# AGENTS.md

给进入本仓库的编码 agent 的说明。人类用户请看 `README.md`。

## 这是什么

一个 Agent Skill + 可选 CLI 管线。技能入口是根目录 `SKILL.md`（YAML frontmatter 决定何时触发，正文是作业流程）。

| 路径 | 作用 |
|---|---|
| `SKILL.md` | 技能定义。要判断该走哪几个 stage，读这里 |
| `scripts/run_pipeline.py` | 六阶段主入口：`audit → cv → interp → perm → ad → report` |
| `scripts/selfcheck.py` | 环境自检，55 项 |
| `references/` | 按需读取：`reviewer-checklist.md`、`applicability-domain.md`、`config-schema.md`、`validation-protocols.md` |
| `assets/config/` | 四个预设：`regression_spectra` / `regression_tabular` / `classification_tabular` / `smoke` |

## 动手之前

```bash
python scripts/selfcheck.py                                            # 必须全绿
python scripts/run_pipeline.py --config assets/config/smoke.json \
    --out runs/smoke --budget fast                                     # 30 秒冒烟
```

自检未全绿不要碰真实数据——它失败时不报错，会给出一份看起来正常的错数字。

## 红线（Gate 会拦，不要绕过）

- 交叉验证必须按 `sample_id` 分组；不得用随机 KFold 产出论文数字。
- 调参与评估必须分开（嵌套 CV），不得用测试折选超参。
- 置换检验按样品置换 y，不按行；bootstrap 按组。
- 必须保留 PLS 基线并报 RPD / RPIQ，不允许只报树模型。
- 适用域（AD）判定无效时，不得声称模型可外推。
- AD 与置换检验报告独立样品数，不报行数（60 样品 × 3 重复 ≠ 180 个样本）。
- 报告里每个数字只能来自 `cv/oof_predictions.csv`，不要手改表格；改了要重跑。
- 改任何 `.py` 都会让 `--resume` 失效重跑，这是故意的。

## 修改与发布

- 版本号三处同步：`SKILL.md` 的 `version`、`package.json` 的 `version`、`CITATION.cff`。
- 发布流程：打 `vX.Y.Z` tag → GitHub Release（附 `dsh-research-ml-modeling.tgz`）→ 更新 `CHANGELOG.md`。
- 技能名 `research-ml-modeling` 与插件包名 `dsh-research-ml-modeling` 属对外契约，改动需同步改 README / llms.txt 里的安装命令。
