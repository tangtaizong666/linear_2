"""
LightGBM 参数推荐模型
用于根据销售预测数据推荐最优的生产优化参数
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import os
import joblib

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("警告: LightGBM 未安装，请运行 'pip install lightgbm' 安装")


class ParameterRecommender:
    """
    参数推荐模型

    使用多个 LightGBM 回归器分别预测各类参数
    """

    def __init__(self):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM 未安装，请运行 'pip install lightgbm' 安装")

        # 模型字典：每个目标一个模型
        self.models: Dict[str, lgb.LGBMRegressor] = {}

        # 参数分组
        self.param_groups = {
            'profits': ['碳酸饮料_利润', '果汁饮料_利润', '茶饮料_利润', '功能饮料_利润', '矿泉水_利润'],
            'material_limits': ['白砂糖_供应限制', '浓缩果汁_供应限制', '茶叶提取物_供应限制',
                               '功能成分_供应限制', '包装材料_供应限制'],
            'transport_limits': ['道里区_运输限制', '南岗区_运输限制', '道外区_运输限制',
                                '香坊区_运输限制', '松北区_运输限制'],
            'production_constraints': ['最小生产比例', '最大生产倍数']
        }

        # 目标名称列表
        self.target_names: List[str] = []
        for group in self.param_groups.values():
            self.target_names.extend(group)

        # 特征名称
        self.feature_names: List[str] = []

        # 是否已训练
        self.is_fitted = False

        # 模型参数
        self.lgb_params = {
            'n_estimators': 200,
            'max_depth': 8,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'verbose': -1
        }

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: List[str] = None,
            target_names: List[str] = None,
            eval_set: Tuple[np.ndarray, np.ndarray] = None) -> Dict[str, float]:
        """
        训练模型

        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签矩阵 (n_samples, n_targets)
            feature_names: 特征名称列表
            target_names: 目标名称列表
            eval_set: 验证集 (X_val, y_val)

        Returns:
            metrics: 各目标的训练指标
        """
        if feature_names:
            self.feature_names = feature_names
        if target_names:
            self.target_names = target_names

        metrics = {}

        print(f"开始训练 {len(self.target_names)} 个 LightGBM 模型...")

        for i, target_name in enumerate(self.target_names):
            print(f"  训练模型 [{i+1}/{len(self.target_names)}]: {target_name}")

            # 创建模型
            model = lgb.LGBMRegressor(**self.lgb_params)

            # 准备回调
            callbacks = []
            if eval_set is not None:
                X_val, y_val = eval_set
                callbacks.append(lgb.early_stopping(stopping_rounds=20, verbose=False))

            # 训练
            if eval_set is not None:
                model.fit(
                    X, y[:, i],
                    eval_set=[(X_val, y_val[:, i])],
                    callbacks=callbacks
                )
            else:
                model.fit(X, y[:, i])

            self.models[target_name] = model

            # 计算训练指标
            train_pred = model.predict(X)
            mae = np.mean(np.abs(train_pred - y[:, i]))
            metrics[target_name] = mae

        self.is_fitted = True
        print("所有模型训练完成！")

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测参数

        Args:
            X: 特征矩阵

        Returns:
            predictions: 预测的参数矩阵
        """
        if not self.is_fitted:
            raise RuntimeError("模型尚未训练，请先调用 fit() 方法")

        predictions = np.zeros((len(X), len(self.target_names)))

        for i, target_name in enumerate(self.target_names):
            predictions[:, i] = self.models[target_name].predict(X)

        return predictions

    def predict_dict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        预测参数并返回字典格式

        Args:
            X: 特征矩阵

        Returns:
            predictions_dict: 按参数组分类的预测结果
        """
        predictions = self.predict(X)

        result = {}
        idx = 0

        # 利润参数
        result['profits'] = predictions[:, idx:idx+5]
        idx += 5

        # 原料限制
        result['material_limits'] = predictions[:, idx:idx+5]
        idx += 5

        # 运输限制
        result['transport_limits'] = predictions[:, idx:idx+5]
        idx += 5

        # 生产约束
        result['min_production_ratio'] = predictions[:, idx]
        result['max_production_multiplier'] = predictions[:, idx+1]

        return result

    def get_feature_importance(self, top_n: int = 10) -> Dict[str, pd.DataFrame]:
        """
        获取特征重要性

        Args:
            top_n: 返回前N个重要特征

        Returns:
            importance_dict: 各目标的特征重要性
        """
        if not self.is_fitted:
            raise RuntimeError("模型尚未训练")

        importance_dict = {}

        for target_name, model in self.models.items():
            importance = model.feature_importances_

            if self.feature_names:
                df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': importance
                })
            else:
                df = pd.DataFrame({
                    'feature': [f'feature_{i}' for i in range(len(importance))],
                    'importance': importance
                })

            df = df.sort_values('importance', ascending=False).head(top_n)
            importance_dict[target_name] = df

        return importance_dict

    def save_model(self, save_dir: str = './data'):
        """保存模型"""
        os.makedirs(save_dir, exist_ok=True)

        # 保存所有子模型
        for target_name, model in self.models.items():
            safe_name = target_name.replace('/', '_')
            model_path = os.path.join(save_dir, f'lgb_{safe_name}.joblib')
            joblib.dump(model, model_path)

        # 保存元信息
        meta = {
            'target_names': self.target_names,
            'feature_names': self.feature_names,
            'param_groups': self.param_groups
        }
        joblib.dump(meta, os.path.join(save_dir, 'lgb_meta.joblib'))

        print(f"模型已保存至: {save_dir}")

    def load_model(self, load_dir: str = './data'):
        """加载模型"""
        # 加载元信息
        meta = joblib.load(os.path.join(load_dir, 'lgb_meta.joblib'))
        self.target_names = meta['target_names']
        self.feature_names = meta['feature_names']
        self.param_groups = meta['param_groups']

        # 加载所有子模型
        self.models = {}
        for target_name in self.target_names:
            safe_name = target_name.replace('/', '_')
            model_path = os.path.join(load_dir, f'lgb_{safe_name}.joblib')
            self.models[target_name] = joblib.load(model_path)

        self.is_fitted = True
        print(f"模型已从 {load_dir} 加载")


class SalesFeatureExtractor:
    """
    销售数据特征提取器

    将 Transformer 预测的销售数据转换为 LightGBM 输入特征
    """

    def __init__(self):
        self.beverage_types = ['碳酸饮料', '果汁饮料', '茶饮料', '功能饮料', '矿泉水']

    def extract_features(self, sales_data: np.ndarray) -> np.ndarray:
        """
        从销售数据中提取特征

        Args:
            sales_data: 销售数据，形状为 (seq_len, n_beverages) 或 (batch, seq_len, n_beverages)

        Returns:
            features: 特征向量或矩阵
        """
        if sales_data.ndim == 2:
            return self._extract_single(sales_data)
        elif sales_data.ndim == 3:
            return np.array([self._extract_single(s) for s in sales_data])
        else:
            raise ValueError(f"不支持的数据维度: {sales_data.ndim}")

    def _extract_single(self, scenario: np.ndarray) -> np.ndarray:
        """提取单个场景的特征"""
        features = []

        # 1. 各饮料总销量
        total_sales = np.sum(scenario, axis=0)
        features.extend(total_sales)

        # 2. 各饮料平均销量
        avg_sales = np.mean(scenario, axis=0)
        features.extend(avg_sales)

        # 3. 各饮料销量标准差
        std_sales = np.std(scenario, axis=0)
        features.extend(std_sales)

        # 4. 各饮料销量趋势
        trend = scenario[-1] / (scenario[0] + 1e-6)
        features.extend(trend)

        # 5. 总体统计
        features.append(np.sum(total_sales))
        features.append(np.max(total_sales))
        features.append(np.min(total_sales))

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

    def get_feature_names(self) -> List[str]:
        """获取特征名称"""
        names = []

        for bev in self.beverage_types:
            names.append(f'{bev}_总销量')
        for bev in self.beverage_types:
            names.append(f'{bev}_平均销量')
        for bev in self.beverage_types:
            names.append(f'{bev}_销量波动')
        for bev in self.beverage_types:
            names.append(f'{bev}_趋势')

        names.extend(['总销量', '最大单品销量', '最小单品销量'])

        for bev in self.beverage_types:
            names.append(f'{bev}_占比')

        names.extend(['日均总销量', '日销量波动', '日最大销量', '日最小销量'])

        return names


def recommend_parameters_from_prediction(transformer_predictions: np.ndarray,
                                          model_dir: str = './data') -> Dict:
    """
    根据 Transformer 预测结果推荐参数

    Args:
        transformer_predictions: Transformer 预测的未来销售数据 (seq_len, n_beverages)
        model_dir: LightGBM 模型目录

    Returns:
        recommendations: 推荐的参数字典
    """
    # 加载模型
    recommender = ParameterRecommender()
    recommender.load_model(model_dir)

    # 提取特征
    extractor = SalesFeatureExtractor()
    features = extractor.extract_features(transformer_predictions)

    if features.ndim == 1:
        features = features.reshape(1, -1)

    # 预测参数
    predictions = recommender.predict_dict(features)

    # 整理结果
    beverage_types = ['碳酸饮料', '果汁饮料', '茶饮料', '功能饮料', '矿泉水']
    material_types = ['白砂糖', '浓缩果汁', '茶叶提取物', '功能成分', '包装材料']
    regions = ['道里区', '南岗区', '道外区', '香坊区', '松北区']

    recommendations = {
        'profits': {bev: round(float(predictions['profits'][0, i]), 1)
                   for i, bev in enumerate(beverage_types)},
        'material_limits': {mat: round(float(predictions['material_limits'][0, i]), 0)
                           for i, mat in enumerate(material_types)},
        'transport_limits': {reg: round(float(predictions['transport_limits'][0, i]), 0)
                            for i, reg in enumerate(regions)},
        'min_production_ratio': round(float(predictions['min_production_ratio'][0]), 2),
        'max_production_multiplier': round(float(predictions['max_production_multiplier'][0]), 1)
    }

    return recommendations


if __name__ == '__main__':
    print("=" * 60)
    print("LightGBM 参数推荐模型测试")
    print("=" * 60)

    # 测试特征提取器
    extractor = SalesFeatureExtractor()

    # 模拟销售数据 (7天, 5种饮料)
    test_sales = np.random.uniform(1000, 3000, (7, 5))

    features = extractor.extract_features(test_sales)
    print(f"\n输入销售数据形状: {test_sales.shape}")
    print(f"提取的特征形状: {features.shape}")
    print(f"特征数量: {len(extractor.get_feature_names())}")

    # 测试模型（如果已安装 LightGBM）
    if LIGHTGBM_AVAILABLE:
        print("\nLightGBM 已安装，可以进行模型训练")

        # 创建简单测试数据
        X_test = np.random.randn(100, len(extractor.get_feature_names()))
        y_test = np.random.randn(100, 17)  # 17个目标

        # 创建并训练模型
        recommender = ParameterRecommender()
        recommender.lgb_params['n_estimators'] = 10  # 减少迭代次数用于测试

        feature_names = extractor.get_feature_names()
        target_names = recommender.target_names

        metrics = recommender.fit(X_test, y_test, feature_names, target_names)
        print(f"\n训练完成，MAE: {np.mean(list(metrics.values())):.4f}")

        # 测试预测
        predictions = recommender.predict(X_test[:5])
        print(f"预测结果形状: {predictions.shape}")

    print("\n模型测试完成！")
