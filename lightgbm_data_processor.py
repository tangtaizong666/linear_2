"""
LightGBM 参数推荐模型 - 数据处理模块
用于生成训练数据：销售数据 -> 最优参数配置
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import os
from scipy.optimize import linprog
import warnings
warnings.filterwarnings('ignore')


class OptimalParameterGenerator:
    """
    最优参数生成器

    根据销售数据场景，通过优化算法找出最优的参数配置
    用于生成 LightGBM 的训练数据
    """

    def __init__(self):
        # 饮料和原料类型
        self.beverage_types = ['碳酸饮料', '果汁饮料', '茶饮料', '功能饮料', '矿泉水']
        self.material_types = ['白砂糖', '浓缩果汁', '茶叶提取物', '功能成分', '包装材料']
        self.transport_regions = ['道里区', '南岗区', '道外区', '香坊区', '松北区']

        self.n_beverages = 5
        self.n_materials = 5
        self.n_regions = 5

        # 原料消耗矩阵 (单位: 千克/升)
        self.material_consumption = np.array([
            [0.15, 0.08, 0.06, 0.10, 0.02],  # 白砂糖
            [0.02, 0.25, 0.03, 0.05, 0.01],  # 浓缩果汁
            [0.01, 0.02, 0.20, 0.08, 0.01],  # 茶叶提取物
            [0.00, 0.00, 0.00, 0.15, 0.00],  # 功能成分
            [0.10, 0.12, 0.11, 0.14, 0.08]   # 包装材料
        ])

        # 需求权重矩阵
        self.demand_weights = np.array([
            [0.25, 0.30, 0.20, 0.15, 0.10],
            [0.20, 0.35, 0.25, 0.15, 0.05],
            [0.30, 0.25, 0.20, 0.20, 0.05],
            [0.35, 0.30, 0.20, 0.10, 0.05],
            [0.15, 0.25, 0.30, 0.20, 0.10]
        ])

        # 基础利润范围 (元/升)
        self.profit_ranges = {
            '碳酸饮料': (6.0, 12.0),
            '果汁饮料': (8.0, 16.0),
            '茶饮料': (7.0, 14.0),
            '功能饮料': (10.0, 20.0),
            '矿泉水': (4.0, 10.0)
        }

    def calculate_optimal_material_limits(self, predicted_sales: np.ndarray,
                                          buffer_ratio: float = 1.2) -> np.ndarray:
        """
        根据预测销量计算最优原料供应限制

        Args:
            predicted_sales: 预测的销售量 (n_days, n_beverages) 或 (n_beverages,)
            buffer_ratio: 缓冲系数，防止原料不足

        Returns:
            optimal_material_limits: 推荐的原料供应限制
        """
        # 如果是多天数据，取总和或平均
        if predicted_sales.ndim == 2:
            total_sales = np.sum(predicted_sales, axis=0)
        else:
            total_sales = predicted_sales

        # 计算原料需求
        material_needs = self.material_consumption @ total_sales

        # 添加缓冲
        optimal_limits = material_needs * buffer_ratio

        # 确保最小值
        min_limits = np.array([5000, 3000, 2000, 500, 4000])
        optimal_limits = np.maximum(optimal_limits, min_limits)

        # 四舍五入到百位
        optimal_limits = np.round(optimal_limits / 100) * 100

        return optimal_limits

    def calculate_optimal_transport_limits(self, predicted_sales: np.ndarray,
                                           buffer_ratio: float = 1.15) -> np.ndarray:
        """
        根据预测销量计算最优运输能力限制

        Args:
            predicted_sales: 预测的销售量
            buffer_ratio: 缓冲系数

        Returns:
            optimal_transport_limits: 推荐的运输能力限制
        """
        if predicted_sales.ndim == 2:
            total_sales = np.sum(predicted_sales, axis=0)
        else:
            total_sales = predicted_sales

        # 计算各区域运输需求
        transport_needs = total_sales @ self.demand_weights

        # 添加缓冲
        optimal_limits = transport_needs * buffer_ratio

        # 确保最小值
        min_limits = np.array([1000, 800, 600, 500, 400])
        optimal_limits = np.maximum(optimal_limits, min_limits)

        # 四舍五入到五十位
        optimal_limits = np.round(optimal_limits / 50) * 50

        return optimal_limits

    def calculate_optimal_production_constraints(self, predicted_sales: np.ndarray,
                                                  historical_avg: np.ndarray) -> Tuple[float, float]:
        """
        根据预测销量计算最优生产约束参数

        Args:
            predicted_sales: 预测的销售量
            historical_avg: 历史平均销量

        Returns:
            (min_ratio, max_multiplier): 最小生产比例和最大生产倍数
        """
        if predicted_sales.ndim == 2:
            avg_predicted = np.mean(predicted_sales, axis=0)
        else:
            avg_predicted = predicted_sales

        # 计算预测与历史的比值
        ratios = avg_predicted / (historical_avg + 1e-6)

        # 根据比值确定生产约束
        avg_ratio = np.mean(ratios)

        if avg_ratio > 1.3:  # 预测大幅增长
            min_ratio = 0.7
            max_multiplier = 2.0
        elif avg_ratio > 1.1:  # 预测小幅增长
            min_ratio = 0.75
            max_multiplier = 1.8
        elif avg_ratio > 0.9:  # 预测稳定
            min_ratio = 0.8
            max_multiplier = 1.5
        elif avg_ratio > 0.7:  # 预测小幅下降
            min_ratio = 0.85
            max_multiplier = 1.3
        else:  # 预测大幅下降
            min_ratio = 0.9
            max_multiplier = 1.2

        return min_ratio, max_multiplier

    def calculate_optimal_profits(self, predicted_sales: np.ndarray,
                                  market_condition: str = 'normal') -> np.ndarray:
        """
        根据预测销量和市场条件推荐利润设置

        Args:
            predicted_sales: 预测销量
            market_condition: 市场条件 ('high_demand', 'normal', 'low_demand')

        Returns:
            optimal_profits: 推荐的利润设置
        """
        if predicted_sales.ndim == 2:
            total_sales = np.sum(predicted_sales, axis=0)
        else:
            total_sales = predicted_sales

        # 计算各品类销量占比
        sales_ratio = total_sales / (np.sum(total_sales) + 1e-6)

        optimal_profits = []

        for i, beverage in enumerate(self.beverage_types):
            low, high = self.profit_ranges[beverage]

            # 根据销量占比调整利润
            # 高销量品类可以适当降低利润以促销，低销量品类提高利润
            if sales_ratio[i] > 0.25:  # 高销量
                profit = low + (high - low) * 0.4
            elif sales_ratio[i] > 0.15:  # 中等销量
                profit = low + (high - low) * 0.6
            else:  # 低销量
                profit = low + (high - low) * 0.8

            # 根据市场条件调整
            if market_condition == 'high_demand':
                profit *= 1.1
            elif market_condition == 'low_demand':
                profit *= 0.9

            # 确保在范围内
            profit = np.clip(profit, low, high)
            optimal_profits.append(round(profit, 1))

        return np.array(optimal_profits)


class LightGBMDataGenerator:
    """
    LightGBM 训练数据生成器

    生成销售场景和对应的最优参数配置
    """

    def __init__(self):
        self.param_generator = OptimalParameterGenerator()
        self.beverage_types = self.param_generator.beverage_types

        # 基础销量范围
        self.base_sales_ranges = {
            '碳酸饮料': (1500, 3000),
            '果汁饮料': (1000, 2500),
            '茶饮料': (800, 2000),
            '功能饮料': (500, 1500),
            '矿泉水': (2000, 4000)
        }

    def generate_sales_scenarios(self, n_samples: int = 1000,
                                  seq_len: int = 7) -> np.ndarray:
        """
        生成多种销售场景

        Args:
            n_samples: 样本数量
            seq_len: 序列长度（天数）

        Returns:
            sales_scenarios: (n_samples, seq_len, n_beverages)
        """
        scenarios = []

        for _ in range(n_samples):
            # 随机选择场景类型
            scenario_type = np.random.choice([
                'stable', 'growth', 'decline', 'seasonal_high', 'seasonal_low',
                'weekend_boost', 'holiday', 'random'
            ])

            scenario = self._generate_single_scenario(seq_len, scenario_type)
            scenarios.append(scenario)

        return np.array(scenarios)

    def _generate_single_scenario(self, seq_len: int, scenario_type: str) -> np.ndarray:
        """生成单个销售场景"""
        scenario = np.zeros((seq_len, 5))

        # 基础销量
        base = np.array([
            np.random.uniform(*self.base_sales_ranges[bev])
            for bev in self.beverage_types
        ])

        for day in range(seq_len):
            daily = base.copy()

            if scenario_type == 'stable':
                # 稳定销量，小幅波动
                daily *= np.random.uniform(0.95, 1.05, 5)

            elif scenario_type == 'growth':
                # 增长趋势
                growth_factor = 1 + 0.03 * day
                daily *= growth_factor * np.random.uniform(0.95, 1.05, 5)

            elif scenario_type == 'decline':
                # 下降趋势
                decline_factor = 1 - 0.02 * day
                daily *= decline_factor * np.random.uniform(0.95, 1.05, 5)

            elif scenario_type == 'seasonal_high':
                # 旺季（夏季）
                daily *= np.random.uniform(1.2, 1.5, 5)

            elif scenario_type == 'seasonal_low':
                # 淡季（冬季）
                daily *= np.random.uniform(0.6, 0.8, 5)

            elif scenario_type == 'weekend_boost':
                # 周末增长
                if day % 7 >= 5:
                    daily *= np.random.uniform(1.2, 1.4, 5)
                else:
                    daily *= np.random.uniform(0.9, 1.0, 5)

            elif scenario_type == 'holiday':
                # 节假日
                daily *= np.random.uniform(1.3, 1.8, 5)

            else:  # random
                daily *= np.random.uniform(0.7, 1.3, 5)

            scenario[day] = daily

        return scenario

    def generate_training_data(self, n_samples: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成完整的训练数据

        Args:
            n_samples: 样本数量

        Returns:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签矩阵 (n_samples, n_targets)
        """
        # 生成销售场景
        scenarios = self.generate_sales_scenarios(n_samples, seq_len=7)

        X_list = []
        y_list = []

        for scenario in scenarios:
            # 提取特征
            features = self._extract_features(scenario)

            # 计算最优参数
            labels = self._calculate_optimal_params(scenario)

            X_list.append(features)
            y_list.append(labels)

        X = np.array(X_list)
        y = np.array(y_list)

        return X, y

    def _extract_features(self, scenario: np.ndarray) -> np.ndarray:
        """
        从销售场景中提取特征

        Args:
            scenario: (seq_len, n_beverages)

        Returns:
            features: 特征向量
        """
        features = []

        # 1. 各饮料总销量
        total_sales = np.sum(scenario, axis=0)
        features.extend(total_sales)

        # 2. 各饮料平均销量
        avg_sales = np.mean(scenario, axis=0)
        features.extend(avg_sales)

        # 3. 各饮料销量标准差（波动性）
        std_sales = np.std(scenario, axis=0)
        features.extend(std_sales)

        # 4. 各饮料销量趋势（最后一天 vs 第一天）
        trend = scenario[-1] / (scenario[0] + 1e-6)
        features.extend(trend)

        # 5. 总体统计
        features.append(np.sum(total_sales))  # 总销量
        features.append(np.max(total_sales))  # 最大单品销量
        features.append(np.min(total_sales))  # 最小单品销量

        # 6. 销量占比
        sales_ratio = total_sales / (np.sum(total_sales) + 1e-6)
        features.extend(sales_ratio)

        # 7. 日销量波动
        daily_totals = np.sum(scenario, axis=1)
        features.append(np.mean(daily_totals))
        features.append(np.std(daily_totals))
        features.append(np.max(daily_totals))
        features.append(np.min(daily_totals))

        return np.array(features)

    def _calculate_optimal_params(self, scenario: np.ndarray) -> np.ndarray:
        """
        计算场景对应的最优参数

        Args:
            scenario: 销售场景

        Returns:
            params: 参数向量
        """
        gen = self.param_generator

        # 历史平均（用于计算生产约束）
        historical_avg = np.array([2000, 1500, 1200, 800, 2500])

        # 计算各类参数
        profits = gen.calculate_optimal_profits(scenario)
        material_limits = gen.calculate_optimal_material_limits(scenario)
        transport_limits = gen.calculate_optimal_transport_limits(scenario)
        min_ratio, max_multiplier = gen.calculate_optimal_production_constraints(
            scenario, historical_avg
        )

        # 合并所有参数
        params = np.concatenate([
            profits,           # 5个利润参数
            material_limits,   # 5个原料限制
            transport_limits,  # 5个运输限制
            [min_ratio, max_multiplier]  # 2个生产约束
        ])

        return params

    def get_feature_names(self) -> List[str]:
        """获取特征名称列表"""
        names = []

        # 总销量特征
        for bev in self.beverage_types:
            names.append(f'{bev}_总销量')

        # 平均销量特征
        for bev in self.beverage_types:
            names.append(f'{bev}_平均销量')

        # 标准差特征
        for bev in self.beverage_types:
            names.append(f'{bev}_销量波动')

        # 趋势特征
        for bev in self.beverage_types:
            names.append(f'{bev}_趋势')

        # 总体统计
        names.extend(['总销量', '最大单品销量', '最小单品销量'])

        # 销量占比
        for bev in self.beverage_types:
            names.append(f'{bev}_占比')

        # 日销量波动
        names.extend(['日均总销量', '日销量波动', '日最大销量', '日最小销量'])

        return names

    def get_target_names(self) -> List[str]:
        """获取目标变量名称列表"""
        names = []

        # 利润参数
        for bev in self.beverage_types:
            names.append(f'{bev}_利润')

        # 原料限制
        material_types = ['白砂糖', '浓缩果汁', '茶叶提取物', '功能成分', '包装材料']
        for mat in material_types:
            names.append(f'{mat}_供应限制')

        # 运输限制
        regions = ['道里区', '南岗区', '道外区', '香坊区', '松北区']
        for reg in regions:
            names.append(f'{reg}_运输限制')

        # 生产约束
        names.extend(['最小生产比例', '最大生产倍数'])

        return names


def prepare_lightgbm_data(n_samples: int = 3000,
                          test_ratio: float = 0.2,
                          save_dir: str = './data') -> Tuple:
    """
    准备 LightGBM 训练数据

    Args:
        n_samples: 样本数量
        test_ratio: 测试集比例
        save_dir: 保存目录

    Returns:
        X_train, X_test, y_train, y_test, feature_names, target_names
    """
    os.makedirs(save_dir, exist_ok=True)

    # 生成数据
    generator = LightGBMDataGenerator()
    X, y = generator.generate_training_data(n_samples)

    feature_names = generator.get_feature_names()
    target_names = generator.get_target_names()

    # 划分训练集和测试集
    n_test = int(n_samples * test_ratio)
    indices = np.random.permutation(n_samples)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    # 保存数据
    np.savez(os.path.join(save_dir, 'lightgbm_data.npz'),
             X_train=X_train, X_test=X_test,
             y_train=y_train, y_test=y_test)

    # 保存特征和目标名称
    pd.DataFrame({'feature_names': feature_names}).to_csv(
        os.path.join(save_dir, 'feature_names.csv'), index=False
    )
    pd.DataFrame({'target_names': target_names}).to_csv(
        os.path.join(save_dir, 'target_names.csv'), index=False
    )

    print(f"数据准备完成:")
    print(f"  - 训练样本数: {len(X_train)}")
    print(f"  - 测试样本数: {len(X_test)}")
    print(f"  - 特征数量: {X.shape[1]}")
    print(f"  - 目标数量: {y.shape[1]}")
    print(f"  - 数据已保存至: {save_dir}")

    return X_train, X_test, y_train, y_test, feature_names, target_names


if __name__ == '__main__':
    print("=" * 60)
    print("LightGBM 参数推荐模型 - 数据生成测试")
    print("=" * 60)

    # 测试数据生成
    X_train, X_test, y_train, y_test, feature_names, target_names = prepare_lightgbm_data(
        n_samples=1000, test_ratio=0.2
    )

    print(f"\n特征名称 ({len(feature_names)}个):")
    for i, name in enumerate(feature_names):
        print(f"  {i+1}. {name}")

    print(f"\n目标名称 ({len(target_names)}个):")
    for i, name in enumerate(target_names):
        print(f"  {i+1}. {name}")

    print("\n训练数据样例:")
    print(f"X_train[0]: {X_train[0][:10]}...")
    print(f"y_train[0]: {y_train[0]}")

    print("\n数据生成测试完成！")
