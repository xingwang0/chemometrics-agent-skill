# Copyright 2026 The research-ml-modeling Authors
# SPDX-License-Identifier: Apache-2.0
"""科研级机器学习预测建模内核（research-ml-modeling）。

模块职责一览
------------
utils        路径/编码/日志/计时/内存守卫/JSON 与哈希
config       JSON 配置、budget 档位、--set 覆盖、字段校验
io_data      表格读取、列角色解析、分组兜底、目标编码
audit        数据审计（泄漏/共线/近零方差/EPV/CV 建议）
provenance   复现锁定（版本矩阵 + 三哈希 + 种子）
present      展示规格（字段标签/精度/单位/徽标/措辞守卫）
report_html  单文件响应式 HTML 报告
standards    TRIPOD+AI 清单映射（条目文本由人工转录）
"""

from __future__ import annotations

__version__ = "1.2.11"
__all__ = ["__version__"]
