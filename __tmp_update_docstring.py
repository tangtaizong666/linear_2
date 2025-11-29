# -*- coding: utf-8 -*-
from pathlib import Path
path = Path(''beverage_optimization_model.py'')
text = path.read_text(encoding=''utf-8'')
first = text.find(''"""'')
if first == -1:
    raise SystemExit(''docstring start not found'')
second = text.find(''"""'', first + 3)
if second == -1:
    raise SystemExit(''docstring end not found'')
new_doc = """
饮料生产企业线性规划优化模型 - 解决原料供应与运输双重约束下的利润最大化问题。
1155
核心思想：
1. 将业务中涉及的利润、原料、运输和生产上下限参数转化为标准线性规划矩阵；
2. 使用 SciPy `linprog`（HiGHS 求解器）在严谨的数学模型下寻找最优生产组合；
3. 保持模型层与可视化/交互层解耦，使其能单独用于单元测试、批量仿真或命令行工具。
"""
text = text[:first] + new_doc + text[second + 3:]
path.write_text(text, encoding=''utf-8'')
