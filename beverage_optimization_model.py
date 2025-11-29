"""
饮料生产企业线性规划优化模型
运筹学专家系统 - 解决原料和运输双重约束下的利润最大化问题
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog  # HiGHS solver entry point for large LP problems

# Optional visualization stacks keep the solver usable in batch/headless environments
# 注：模型核心功能与 Streamlit、Plotly 等可视化库。
# 这些依赖在模型求解过程中并不会用到，但如果在其他模块中需要可视化时可单独引入。
# 为避免在无可视化环境下运行测试时引发 ImportError，这里将其设为可选导入。
try:
    import streamlit as st  # type: ignore
except ImportError:
    st = None  # 在模型逻辑中不会用到

try:
    import plotly.graph_objects as go  # type: ignore
    import plotly.express as px  # type: ignore
    from plotly.subplots import make_subplots  # type: ignore
except ImportError:
    go = px = make_subplots = None  # 可视化仅在 Streamlit 应用中使用
import json
from typing import Any, Dict, List, Tuple, Optional
import warnings

# Silence repeated numerical warnings so the notebook/Streamlit console remains readable
warnings.filterwarnings('ignore')

class BeverageOptimizationModel:
    """
    饮料生产企业线性规划优化模型类
    
    该类构建了一个完整的线性规划模型，用于解决饮料生产企业在原料供应和运输能力
    双重约束条件下的利润最大化问题。
    """
    
    def __init__(self):
        """初始化模型参数"""
        # 定义饮料种类和相关参数
        # Production decision variables ordered by SKU importance
        self.beverage_types = ['碳酸饮料', '果汁饮料', '茶饮料', '功能饮料', '矿泉水']
        # Cache counts to reuse when building matrix dimensions
        self.n_beverages = len(self.beverage_types)
        
        # 定义原料种类
        # Raw-material categories whose stock levels constrain production
        self.material_types = ['白砂糖', '浓缩果汁', '茶叶提取物', '功能成分', '包装材料']
        # Track how many material constraints must be generated
        self.n_materials = len(self.material_types)
        
        # 定义运输区域
        # Downstream distribution regions with dedicated transport quotas
        self.transport_regions = ['道里区', '南岗区', '道外区', '香坊区', '松北区']
        # Same idea for transport capacity constraints
        self.n_regions = len(self.transport_regions)
        
        # 初始化默认参数
        # Load a baseline data pack so the model can be solved immediately
        self.setup_default_parameters()
        
    def setup_default_parameters(self):
        """设置默认模型参数"""
        
        # 1. 利润参数 (元/升)
        # Unit profit (per thousand liters) directly feeds the objective vector
        self.profits = np.array([8.5, 12.0, 10.5, 15.0, 6.0])  # 各饮料单位利润
        
        # 2. 原料消耗矩阵 (单位: 千克/升)
        # 每行代表一种原料，每列代表一种饮料
        # Material usage matrix: rows=raw materials, cols=beverage SKUs
        self.material_consumption = np.array([
            [0.15, 0.08, 0.06, 0.10, 0.02],  # 白砂糖
            [0.02, 0.25, 0.03, 0.05, 0.01],  # 浓缩果汁
            [0.01, 0.02, 0.20, 0.08, 0.01],  # 茶叶提取物
            [0.00, 0.00, 0.00, 0.15, 0.00],  # 功能成分
            [0.10, 0.12, 0.11, 0.14, 0.08]   # 包装材料
        ])
        
        # 3. 原料供应限制 (千克)
        # Available inventory per raw material for the planning cycle
        # Regional supply estimates derived from 2025 Harbin industrial planning brief (in tons)
        self.material_limits = np.array([15000, 8000, 6000, 2000, 12000])
        
        # 4. 运输能力限制 (升)
        # Upper bounds for shipping capacity into each sales region (in kl)
        # Freight capacities (general + cold-chain) per district, unit: thousand liters
        self.transport_limits = np.array([3000, 2500, 2000, 1800, 1200])
        
        # 5. 各饮料在各区域的需求量权重
        # Demand allocation ratios telling us how production is split into regions
        self.demand_weights = np.array([
            [0.25, 0.30, 0.20, 0.15, 0.10],  # 碳酸饮料
            [0.20, 0.35, 0.25, 0.15, 0.05],  # 果汁饮料
            [0.30, 0.25, 0.20, 0.20, 0.05],  # 茶饮料
            [0.35, 0.30, 0.20, 0.10, 0.05],  # 功能饮料
            [0.15, 0.25, 0.30, 0.20, 0.10]   # 矿泉水
        ])
        
        # 6. 上期销售情况 (升)
        # Historical sales baseline, used for both min and max production guardrails
        self.previous_sales = np.array([2000, 1500, 1200, 800, 2500])
        
        # 7. 最小生产量要求 (销售量的80%)
        # Enforce at least 80% of last period output to keep shelves supplied
        self.min_production = 0.8 * self.previous_sales
        
        # 8. 最大生产能力限制
        # Cap any SKU at 150% of historical sales to avoid unrealistic spikes
        self.max_production_multiplier = 1.5
        
    def build_matrices(self):
        """
        构建线性规划的标准形式矩阵
        
        目标函数: max c^T x
        约束条件: Ax <= b
        """
        
        # 决策变量: x = [x1, x2, x3, x4, x5] 各饮料生产量
        
        # 目标函数系数 (最大化问题需要转换为最小化)
        # Keep decision variables ordered exactly as self.beverage_types for easier analysis
        c = -self.profits  # 负号因为linprog默认最小化
        # SciPy solves a minimization problem, so flipping the sign yields a max objective
        
        # 约束矩阵 A 和约束向量 b
        constraint_list = []
        constraint_rhs = []
        # Build A and b incrementally so that new business rules are easy to append
        
        # 1. 原料约束
        for i in range(self.n_materials):
            constraint_list.append(self.material_consumption[i, :])
            constraint_rhs.append(self.material_limits[i])
            # 每个原料的消耗行都与其库存上限匹配，形成 n_materials 条约束
        
        # 2. 运输能力约束
        # 总运输量不能超过各区域的运输能力
        for j in range(self.n_regions):
            # 计算各饮料在该区域的运输量
            transport_constraint = np.zeros(self.n_beverages)
            for i in range(self.n_beverages):
                # 假设生产量按需求权重分配到各区域
                transport_constraint[i] = self.demand_weights[i, j]
            constraint_list.append(transport_constraint)
            constraint_rhs.append(self.transport_limits[j])
            # 将产量按需求权重分配到各区域后，与对应运输能力比较
        
        # 3. 最小生产量约束
        for i in range(self.n_beverages):
            min_constraint = np.zeros(self.n_beverages)
            min_constraint[i] = -1  # -x_i <= -min_production_i
            constraint_list.append(min_constraint)
            constraint_rhs.append(-self.min_production[i])
            # 负号将 >= 约束转化成 <= 形式，便于传给 linprog
        
        # 4. 最大生产能力约束
        for i in range(self.n_beverages):
            max_constraint = np.zeros(self.n_beverages)
            max_constraint[i] = 1
            constraint_list.append(max_constraint)
            constraint_rhs.append(self.max_production_multiplier * self.previous_sales[i])
        # 简单的 <= 约束，防止任何 SKU 超过历史销量的 1.5 倍

        A = np.array(constraint_list)
        b = np.array(constraint_rhs)

        return c, A, b  # Standard form consumed directly by SciPy

    def build_constraint_records(self) -> List[Dict[str, np.ndarray]]:
        """构建带有方向信息的约束，供单纯形迭代表可视化使用。"""
        records: List[Dict[str, np.ndarray]] = []

        for idx, material in enumerate(self.material_types):
            records.append({
                'label': f"原料-{material}",
                'sense': '<=',
                'rhs': float(self.material_limits[idx]),
                'coeffs': self.material_consumption[idx, :].astype(float)
            })

        for idx, region in enumerate(self.transport_regions):
            coeffs = self.demand_weights[:, idx].astype(float)
            records.append({
                'label': f"运输-{region}",
                'sense': '<=',
                'rhs': float(self.transport_limits[idx]),
                'coeffs': coeffs
            })

        for idx, beverage in enumerate(self.beverage_types):
            coeffs = np.zeros(self.n_beverages)
            coeffs[idx] = 1.0
            records.append({
                'label': f"最小产量-{beverage}",
                'sense': '>=',
                'rhs': float(self.min_production[idx]),
                'coeffs': coeffs
            })

        for idx, beverage in enumerate(self.beverage_types):
            coeffs = np.zeros(self.n_beverages)
            coeffs[idx] = 1.0
            records.append({
                'label': f"最大产量-{beverage}",
                'sense': '<=',
                'rhs': float(self.max_production_multiplier * self.previous_sales[idx]),
                'coeffs': coeffs
            })

        return records

    def generate_simplex_iterations(self) -> Optional[Dict[str, Any]]:
        """生成单纯形法迭代的详细记录，便于前端逐步展示。"""
        try:
            constraint_records = self.build_constraint_records()
            if not constraint_records:
                return None
            return self._run_logged_simplex(constraint_records)
        except Exception as exc:  # pragma: no cover
            return {'error': str(exc)}

    def _run_logged_simplex(self, constraints: List[Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """执行两阶段单纯形法并记录每一步的单纯形表。"""
        tolerance = 1e-9
        row_labels = [rec['label'] for rec in constraints]
        m = len(constraints)
        n = self.n_beverages

        A = np.vstack([rec['coeffs'] for rec in constraints]).astype(float)
        b = np.array([rec['rhs'] for rec in constraints], dtype=float)

        var_names = [f"x{i + 1}" for i in range(n)]
        var_types = ['decision'] * n
        slack_counter = 0
        surplus_counter = 0
        artificial_counter = 0

        tableau = A.copy()
        if tableau.ndim == 1:
            tableau = tableau.reshape(m, -1)
        if tableau.size == 0:
            return {'iterations': []}

        basis: List[Optional[int]] = [None] * m

        def append_column(values: np.ndarray) -> int:
            nonlocal tableau
            tableau = np.hstack((tableau, values.reshape(m, 1)))
            return tableau.shape[1] - 1

        for row_idx, constraint in enumerate(constraints):
            sense = constraint['sense']
            if sense == '<=':
                slack_counter += 1
                col_values = np.zeros(m)
                col_values[row_idx] = 1.0
                col_idx = append_column(col_values)
                var_names.append(f"s{slack_counter}")
                var_types.append('slack')
                basis[row_idx] = col_idx
            elif sense == '>=':
                surplus_counter += 1
                surplus_col = np.zeros(m)
                surplus_col[row_idx] = -1.0
                append_column(surplus_col)
                var_names.append(f"e{surplus_counter}")
                var_types.append('surplus')
                artificial_counter += 1
                art_col = np.zeros(m)
                art_col[row_idx] = 1.0
                art_idx = append_column(art_col)
                var_names.append(f"a{artificial_counter}")
                var_types.append('artificial')
                basis[row_idx] = art_idx
            else:
                raise ValueError(f"Unsupported constraint sense: {sense}")

        tableau = np.hstack((tableau, b.reshape(m, 1)))

        cj_phase2 = np.zeros(len(var_names))
        for i in range(self.n_beverages):
            cj_phase2[i] = float(self.profits[i])

        cj_phase1 = np.zeros(len(var_names))
        for idx, vtype in enumerate(var_types):
            if vtype == 'artificial':
                cj_phase1[idx] = -1.0

        def compute_cj_minus_zj(cj_vec: np.ndarray) -> np.ndarray:
            zj = np.zeros(len(cj_vec))
            for row_idx, basic in enumerate(basis):
                if basic is None:
                    continue
                zj += cj_vec[basic] * tableau[row_idx, :-1]
            return cj_vec - zj

        def compute_objective_value(cj_vec: np.ndarray) -> float:
            value = 0.0
            for row_idx, basic in enumerate(basis):
                if basic is None:
                    continue
                value += cj_vec[basic] * tableau[row_idx, -1]
            return float(value)

        def pivot(row_idx: int, col_idx: int) -> None:
            pivot_value = tableau[row_idx, col_idx]
            tableau[row_idx, :] = tableau[row_idx, :] / pivot_value
            for r in range(m):
                if r != row_idx:
                    tableau[r, :] -= tableau[r, col_idx] * tableau[row_idx, :]

        def run_phase(phase_name: str, cj_vec: np.ndarray) -> List[Dict[str, Any]]:
            history: List[Dict[str, Any]] = []
            max_iterations = 200
            iteration = 1
            while iteration <= max_iterations:
                tableau_before = tableau.copy()
                column_snapshot = var_names.copy()
                cj_minus_zj = compute_cj_minus_zj(cj_vec)
                entry: Dict[str, Any] = {
                    'phase': phase_name,
                    'iteration': iteration,
                    'status': 'pivot',
                    'column_labels': column_snapshot,
                    'row_labels': row_labels.copy(),
                    'cj_minus_zj': cj_minus_zj.tolist(),
                    'objective_value': compute_objective_value(cj_vec),
                    'tableau_before': tableau_before.tolist(),
                    'entering': None,
                    'leaving': None,
                    'pivot': None,
                    'ratios': []
                }

                valid_indices = list(range(len(cj_vec)))
                if phase_name == 'Phase II':
                    valid_indices = [idx for idx in valid_indices if var_types[idx] != 'artificial']

                entering_idx = None
                entering_value = None
                for idx in valid_indices:
                    value = cj_minus_zj[idx]
                    if entering_value is None or value > entering_value + tolerance:
                        entering_value = value
                        entering_idx = idx

                if entering_idx is None or entering_value is None or entering_value <= tolerance:
                    entry['status'] = 'optimal'
                    entry['tableau_after'] = tableau_before.tolist()
                    history.append(entry)
                    break

                column = tableau[:, entering_idx]
                ratios = []
                leave_row = None
                best_ratio = None
                for row_idx in range(m):
                    if column[row_idx] > tolerance:
                        ratio = tableau[row_idx, -1] / column[row_idx]
                        ratios.append({'constraint': row_labels[row_idx], 'ratio': float(ratio)})
                        if best_ratio is None or ratio < best_ratio - tolerance:
                            best_ratio = ratio
                            leave_row = row_idx
                    else:
                        ratios.append({'constraint': row_labels[row_idx], 'ratio': None})

                entry['ratios'] = ratios
                if leave_row is None:
                    entry['status'] = 'unbounded'
                    entry['tableau_after'] = tableau_before.tolist()
                    history.append(entry)
                    break

                entering_label = column_snapshot[entering_idx]
                leaving_label = column_snapshot[leave_row] if leave_row is not None else row_labels[leave_row]
                entry['entering'] = entering_label
                entry['leaving'] = leaving_label
                entry['pivot'] = {'row': leave_row, 'col': entering_idx, 'value': float(column[leave_row])}
                entry['reason'] = (
                    f"{entering_label} 因 Cj−Zj 最大 {entering_value:.3f} 而入基；"
                    f"{row_labels[leave_row]} 行比值为 {best_ratio:.3f}，因此 {leaving_label} 离基"
                )

                pivot(leave_row, entering_idx)
                basis[leave_row] = entering_idx
                entry['tableau_after'] = tableau.copy().tolist()
                entry['objective_value'] = compute_objective_value(cj_vec)
                history.append(entry)

                iteration += 1
            else:
                history.append({
                    'phase': phase_name,
                    'iteration': max_iterations,
                    'status': 'max_iterations',
                    'column_labels': var_names.copy(),
                    'row_labels': row_labels.copy(),
                    'cj_minus_zj': compute_cj_minus_zj(cj_vec).tolist(),
                    'objective_value': compute_objective_value(cj_vec),
                    'tableau_before': tableau.copy().tolist(),
                    'tableau_after': tableau.copy().tolist(),
                    'entering': None,
                    'leaving': None,
                    'pivot': None,
                    'ratios': []
                })
            return history

        def remove_artificials_from_basis() -> None:
            for row_idx, basic in enumerate(basis):
                if basic is None:
                    continue
                if var_types[basic] != 'artificial':
                    continue
                pivot_done = False
                for candidate_idx, vtype in enumerate(var_types):
                    if vtype == 'artificial':
                        continue
                    if abs(tableau[row_idx, candidate_idx]) > tolerance:
                        pivot(row_idx, candidate_idx)
                        basis[row_idx] = candidate_idx
                        pivot_done = True
                        break
                if not pivot_done:
                    basis[row_idx] = None

        def drop_artificial_columns() -> None:
            nonlocal tableau, var_names, var_types, cj_phase2
            keep_mask = [vtype != 'artificial' for vtype in var_types]
            if all(keep_mask):
                return
            reduced = tableau[:, :-1][:, keep_mask]
            tableau = np.hstack((reduced, tableau[:, -1][:, None]))
            var_names = [name for name, keep in zip(var_names, keep_mask) if keep]
            var_types = [vtype for vtype, keep in zip(var_types, keep_mask) if keep]
            index_mapping = {}
            new_idx = 0
            for old_idx, keep in enumerate(keep_mask):
                if keep:
                    index_mapping[old_idx] = new_idx
                    new_idx += 1
            basis[:] = [index_mapping.get(idx) if idx is not None else None for idx in basis]
            cj_phase2 = cj_phase2[keep_mask]

        history: List[Dict[str, Any]] = []
        if any(vtype == 'artificial' for vtype in var_types):
            history.extend(run_phase('Phase I', cj_phase1.copy()))
            if history:
                final_obj = history[-1]['objective_value']
                if abs(final_obj) > 1e-6:
                    return {'iterations': history, 'status': 'infeasible'}
            remove_artificials_from_basis()
            drop_artificial_columns()
        history.extend(run_phase('Phase II', cj_phase2.copy()))
        return {'iterations': history, 'status': 'feasible'}

    def _diagnose_infeasibility(self) -> Dict[str, Any]:
        """
        在求解器报告模型不可行时，对本业务场景做一些
        基本的一致性检查，帮助用户判断是哪一类约束出问题。

        这里只检查几类“显而易见”的矛盾：
        - 单个产品：最小生产量 > 最大生产量
        - 原料维度：按最小产量计算的原料需求 > 原料供应上限
        - 运输维度：按最小产量计算的各区域运输需求 > 运输能力上限
        """
        diagnostics: Dict[str, Any] = {
            'issues': [],
            'summary': ''
        }

        tol = 1e-6

        # 1) 单个产品维度：最小产量 vs 最大产量
        for i, beverage in enumerate(self.beverage_types):
            min_prod = float(self.min_production[i])
            max_prod = float(self.max_production_multiplier * self.previous_sales[i])
            if min_prod - max_prod > tol:
                diagnostics['issues'].append({
                    'type': 'min_max_conflict',
                    'beverage': beverage,
                    'min_production': min_prod,
                    'max_allowed': max_prod,
                    'message': (
                        f"{beverage} 的最小生产量要求为 {min_prod:.1f}，"
                        f"但最大允许产量只有 {max_prod:.1f}，"
                        "同一产品的“最小生产量”和“最大生产量”约束互相矛盾。"
                    ),
                    'suggestion': "在侧边栏“⚙️ 生产约束参数”中，"
                                  "降低“最小生产比例”或提高“最大生产倍数”，"
                                  "确保最小产量 ≤ 最大产量。"
                })

        # 2) 原料维度：按最小产量计算的需求量是否超过供应上限
        try:
            required_materials = self.material_consumption.dot(self.min_production)
            for j, material in enumerate(self.material_types):
                required = float(required_materials[j])
                limit = float(self.material_limits[j])
                if required - limit > tol:
                    diagnostics['issues'].append({
                        'type': 'material_shortage_at_min',
                        'material': material,
                        'required': required,
                        'limit': limit,
                        'message': (
                            f"在当前“最小生产比例”设置下，"
                            f"{material} 至少需要 {required:.1f} 千克，"
                            f"但原料供应上限只有 {limit:.1f} 千克，"
                            "无论如何调整产品结构都无法满足原料约束。"
                        ),
                        'suggestion': (
                            "可以在侧边栏“📦 原料供应限制”中提高该原料供应上限，"
                            "或在“⚙️ 生产约束参数”中适当降低“最小生产比例”。"
                        )
                    })
        except Exception:
            # 诊断过程失败时不影响整体错误信息返回
            pass

        # 3) 运输维度：按最小产量计算的各区域运输量是否超过运输能力
        try:
            # self.min_production: (n_beverages,)
            # self.demand_weights: (n_beverages, n_regions)
            # 乘积结果为各区域最小运输需求
            required_transport = self.min_production @ self.demand_weights
            for j, region in enumerate(self.transport_regions):
                required = float(required_transport[j])
                limit = float(self.transport_limits[j])
                if required - limit > tol:
                    diagnostics['issues'].append({
                        'type': 'transport_overflow_at_min',
                        'region': region,
                        'required': required,
                        'limit': limit,
                        'message': (
                            f"在当前“最小生产比例”设置下，"
                            f"{region} 区域至少需要运输 {required:.1f} 千升产品，"
                            f"但运输能力上限只有 {limit:.1f} 千升。"
                        ),
                        'suggestion': (
                            "可以在侧边栏“🚛 运输能力限制”中提高该区域运输上限，"
                            "或适当降低“最小生产比例”，减轻该区域的基础运输需求。"
                        )
                    })
        except Exception:
            pass

        if diagnostics['issues']:
            diagnostics['summary'] = (
                f"检测到 {len(diagnostics['issues'])} 处可能导致模型无解的约束设置问题，"
                "请根据下面的具体提示调整参数后重试。"
            )
        else:
            diagnostics['summary'] = (
                "HiGHS 求解器报告“模型不可行（infeasible）”，"
                "但在常见的参数一致性检查中未发现明显互相矛盾的约束。"
                "建议逐步降低“最小生产比例”、放宽部分原料/运输上限，"
                "或检查是否存在极端或错误的参数输入。"
            )

        return diagnostics

    @staticmethod
    def _format_infeasibility_message(solver_message: str,
                                      diagnostics: Dict[str, Any]) -> str:
        """
        将底层求解器返回的信息和业务诊断结果整合为
        一段对用户友好的中文错误提示文本。
        """
        parts: List[str] = []

        base_msg = (solver_message or "").strip()
        if base_msg:
            parts.append(f"底层求解器信息: {base_msg}")

        summary = diagnostics.get('summary')
        if summary:
            parts.append(summary)

        issues = diagnostics.get('issues') or []
        for issue in issues:
            msg = issue.get('message')
            if msg:
                parts.append(msg)
            suggestion = issue.get('suggestion')
            if suggestion:
                parts.append(f"建议：{suggestion}")

        # 使用换行符，便于在 Streamlit 的 st.error 中按行显示
        return "\n".join(parts) if parts else base_msg

    def solve_model(self):
        """
        使用单纯形法求解线性规划模型
        """
        try:
            c, A, b = self.build_matrices()
            result = linprog(c=c, A_ub=A, b_ub=b, method='highs', options={'disp': False})

            if result.success:
                ineqlin = getattr(result, 'ineqlin', None)
                shadow_prices = ineqlin.marginals if ineqlin is not None else None
                reduced_costs = getattr(result, 'reduced_costs', None)
                solution = {
                    'status': '最优解找到',
                    'optimal_value': -result.fun,
                    'decision_variables': result.x,
                    'shadow_prices': shadow_prices,
                    'reduced_costs': reduced_costs,
                    'slack_variables': result.slack,
                    'iterations': result.nit,
                    'success': True
                }
                solution['constraint_analysis'] = self.analyze_constraints(result, A, b)
                solution['simplex_iterations'] = self.generate_simplex_iterations()
                return solution

            diagnostics = self._diagnose_infeasibility()
            message = self._format_infeasibility_message(result.message or "", diagnostics)
            return {
                'status': '求解失败',
                'message': message,
                'success': False,
                'diagnostics': diagnostics
            }
        except Exception as exc:
            diagnostics = self._diagnose_infeasibility()
            message = self._format_infeasibility_message(str(exc), diagnostics)
            return {
                'status': '求解失败',
                'message': message,
                'success': False,
                'diagnostics': diagnostics
            }

    def analyze_constraints(self, result, A, b):
        """
        计算约束利用率、影子价格等信息
        """
        analysis = {
            'material_constraints': {},
            'transport_constraints': {},
            'binding_constraints': [],
            'non_binding_constraints': []
        }
        if not result.success:
            return analysis

        slack = result.slack if getattr(result, 'slack', None) is not None else np.zeros_like(b)
        ineqlin = getattr(result, 'ineqlin', None)
        marginals = ineqlin.marginals if ineqlin is not None else None
        tol = 1e-6
        idx = 0

        for i, material in enumerate(self.material_types):
            usage = float(np.dot(self.material_consumption[i], result.x))
            limit = float(self.material_limits[i])
            slack_val = float(slack[idx]) if slack is not None else 0.0
            shadow_price = float(marginals[idx]) if marginals is not None else 0.0
            binding = abs(usage - limit) <= tol
            analysis['material_constraints'][material] = {
                'usage': usage,
                'limit': limit,
                'slack': slack_val,
                'shadow_price': shadow_price,
                'utilization_rate': usage / limit if limit else 0.0,
                'is_binding': binding
            }
            name = f"原料-{material}"
            (analysis['binding_constraints'] if binding else analysis['non_binding_constraints']).append(name)
            idx += 1

        for j, region in enumerate(self.transport_regions):
            usage = float(np.dot(self.demand_weights[:, j], result.x))
            limit = float(self.transport_limits[j])
            slack_val = float(slack[idx]) if slack is not None else 0.0
            shadow_price = float(marginals[idx]) if marginals is not None else 0.0
            binding = abs(usage - limit) <= tol
            analysis['transport_constraints'][region] = {
                'usage': usage,
                'limit': limit,
                'slack': slack_val,
                'shadow_price': shadow_price,
                'utilization_rate': usage / limit if limit else 0.0,
                'is_binding': binding
            }
            name = f"运输-{region}"
            (analysis['binding_constraints'] if binding else analysis['non_binding_constraints']).append(name)
            idx += 1

        # 跳过最小产量与最大产量约束对应的松弛变量
        idx += self.n_beverages * 2
        return analysis

    def sensitivity_analysis(self, solution):
        """
        进行灵敏度分析，包含详细步骤记录
        """
        if not solution['success']:
            return {'error': '无法对无解模型进行灵敏度分析'}

        analysis = {
            'objective_coefficients': {},
            'rhs_changes': {},
            'recommendations': [],
            'step_logs': []
        }

        base_profits = self.profits.copy()
        base_solution = solution['decision_variables']
        tol = 1e-3
        max_iter_profit = 10
        step_counter = 1

        def log_step_entry(category: str, target: str, direction: str, tested_value: float,
                           sol_result: Optional[Dict[str, Any]] = None, status: str = 'stable',
                           note: Optional[str] = None) -> None:
            nonlocal step_counter
            feasible = bool(sol_result.get('success')) if sol_result else False
            objective_value = float(sol_result['optimal_value']) if sol_result and sol_result.get('success') else None
            snapshot = [round(float(val), 2) for val in sol_result['decision_variables']] if sol_result and sol_result.get('success') else []
            analysis['step_logs'].append({
                'step': step_counter,
                'category': category,
                'target': target,
                'direction': direction,
                'tested_value': float(tested_value),
                'status': status,
                'feasible': feasible,
                'objective_value': objective_value,
                'solution_snapshot': snapshot,
                'note': note
            })
            step_counter += 1

        def clone_base_model() -> 'BeverageOptimizationModel':
            test_model = BeverageOptimizationModel()
            test_model.profits = self.profits.copy()
            test_model.material_limits = self.material_limits.copy()
            test_model.transport_limits = self.transport_limits.copy()
            test_model.demand_weights = self.demand_weights.copy()
            test_model.previous_sales = self.previous_sales.copy()
            test_model.min_production = self.min_production.copy()
            test_model.max_production_multiplier = self.max_production_multiplier
            return test_model

        for i, beverage in enumerate(self.beverage_types):
            info = {
                'current_profit': base_profits[i],
                'optimal_production': base_solution[i],
                'reduced_cost': solution['reduced_costs'][i] if solution['reduced_costs'] is not None else 0
            }
            if solution['reduced_costs'] is not None and solution['reduced_costs'][i] > 1e-6:
                analysis['recommendations'].append(
                    f"{beverage}的当前利润过低，建议提高利润至少{solution['reduced_costs'][i]:.2f}元/升或停止生产"
                )

            orig_profit = base_profits[i]
            step_value = max(abs(orig_profit) * 0.1, 0.5)
            min_profit = orig_profit
            max_profit = orig_profit

            for k in range(1, max_iter_profit + 1):
                new_profit = orig_profit + step_value * k
                test_model = clone_base_model()
                test_model.profits = base_profits.copy()
                test_model.profits[i] = new_profit
                sol = test_model.solve_model()
                stable = sol['success'] and np.allclose(sol['decision_variables'], base_solution, atol=tol, rtol=0)
                status = 'stable' if stable else ('basis_changed' if sol['success'] else 'infeasible')
                log_step_entry('objective', beverage, 'increase', new_profit, sol, status)
                if not sol['success'] or not stable:
                    break
                max_profit = new_profit

            for k in range(1, max_iter_profit + 1):
                new_profit = orig_profit - step_value * k
                if new_profit < 0:
                    log_step_entry('objective', beverage, 'decrease', new_profit, None, 'profit_negative', '利润不能小于0')
                    break
                test_model = clone_base_model()
                test_model.profits = base_profits.copy()
                test_model.profits[i] = new_profit
                sol = test_model.solve_model()
                stable = sol['success'] and np.allclose(sol['decision_variables'], base_solution, atol=tol, rtol=0)
                status = 'stable' if stable else ('basis_changed' if sol['success'] else 'infeasible')
                log_step_entry('objective', beverage, 'decrease', new_profit, sol, status)
                if not sol['success'] or not stable:
                    break
                min_profit = new_profit

            info['range'] = (min_profit, max_profit)
            analysis['objective_coefficients'][beverage] = info

        constraint_analysis = solution['constraint_analysis']
        baseline_solution = solution['decision_variables']

        def analyse_rhs_change(target_label: str, base_value: float, apply_change) -> Tuple[float, float]:
            min_value = base_value
            max_value = base_value
            step_val = max(abs(base_value) * 0.1, 50)
            for direction in ('increase', 'decrease'):
                for k in range(1, 6):
                    delta = step_val * k
                    candidate = base_value + delta if direction == 'increase' else base_value - delta
                    if candidate <= 0 and direction == 'decrease':
                        log_step_entry('rhs', target_label, direction, candidate, None, 'limit_non_positive', '约束右侧需要为正值')
                        break
                    test_model = clone_base_model()
                    apply_change(test_model, candidate)
                    sol = test_model.solve_model()
                    stable = sol['success'] and np.allclose(sol['decision_variables'], baseline_solution, atol=1e-3, rtol=0)
                    status = 'stable' if stable else ('basis_changed' if sol['success'] else 'infeasible')
                    log_step_entry('rhs', target_label, direction, candidate, sol, status)
                    if direction == 'increase' and stable:
                        max_value = candidate
                    if direction == 'decrease' and stable:
                        min_value = candidate
                    if not sol['success'] or not stable:
                        break
            return min_value, max_value

        material_constraints = constraint_analysis.get('material_constraints', {})
        for idx, material in enumerate(self.material_types):
            if material not in material_constraints:
                continue
            info = material_constraints[material]
            if not info['is_binding']:
                continue
            entry = {
                'current_limit': info['limit'],
                'shadow_price': info['shadow_price'],
                'recommendation': f"增加{material}供应可提高利润{info['shadow_price']:.2f}元/千克"
            }

            def apply_material(model: 'BeverageOptimizationModel', value: float) -> None:
                model.material_limits[idx] = value

            entry['range'] = analyse_rhs_change(f"原料-{material}", self.material_limits[idx], apply_material)
            analysis['rhs_changes'][material] = entry

        transport_constraints = constraint_analysis.get('transport_constraints', {})
        for idx, region in enumerate(self.transport_regions):
            if region not in transport_constraints:
                continue
            info = transport_constraints[region]
            if not info['is_binding']:
                continue
            entry = {
                'current_limit': info['limit'],
                'shadow_price': info['shadow_price'],
                'recommendation': f"增加{region}运输能力可提高利润{info['shadow_price']:.2f}元/升"
            }

            def apply_transport(model: 'BeverageOptimizationModel', value: float) -> None:
                model.transport_limits[idx] = value

            entry['range'] = analyse_rhs_change(f"运输-{region}", self.transport_limits[idx], apply_transport)
            analysis['rhs_changes'][region] = entry

        return analysis

    def update_parameters(self, params: Dict):
        """
        更新模型参数
        
        Args:
            params: 参数字典
        """
        # Partial updates keep the solver responsive to UI slider changes
        if 'profits' in params:
            self.profits = np.array(params['profits'])
        
        if 'material_limits' in params:
            self.material_limits = np.array(params['material_limits'])
        
        if 'transport_limits' in params:
            self.transport_limits = np.array(params['transport_limits'])
        
        if 'min_production_ratio' in params:
            ratio = params['min_production_ratio']
            self.min_production = ratio * self.previous_sales
        
        if 'max_production_multiplier' in params:
            self.max_production_multiplier = params['max_production_multiplier']

# 创建全局模型实例
model = BeverageOptimizationModel()
