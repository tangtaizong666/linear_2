"""
LightGBM 参数推荐模型 - 测试与推荐脚本

核心功能：根据 Transformer 预测的销售数据，推荐最优的生产优化参数
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("错误: LightGBM 未安装")

from lightgbm_model import ParameterRecommender, SalesFeatureExtractor, recommend_parameters_from_prediction
from lightgbm_data_processor import LightGBMDataGenerator


def load_transformer_model():
    """
    加载训练好的 Transformer 模型

    Returns:
        model: Transformer 模型
        processor: 数据处理器
    """
    from transformer_model import SalesForecasterEncoderOnly
    from sales_data_processor import SalesDataProcessor

    # 检查模型文件
    model_path = './data/best_transformer_model.pth'
    scaler_path = './data/scaler_params.npz'

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Transformer 模型文件不存在: {model_path}\n请先运行 transformer_train.py")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"数据标准化参数不存在: {scaler_path}\n请先运行 transformer_train.py")

    # 创建模型
    model = SalesForecasterEncoderOnly(
        input_dim=5,
        d_model=128,
        num_heads=8,
        num_layers=4,
        d_ff=512,
        input_seq_len=30,
        output_seq_len=7,
        dropout=0.1
    )

    # 加载权重
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # 加载数据处理器
    processor = SalesDataProcessor()
    processor.load_scaler(scaler_path)

    return model, processor


def get_transformer_predictions(model, processor, historical_data=None):
    """
    使用 Transformer 模型获取销售预测

    Args:
        model: Transformer 模型
        processor: 数据处理器
        historical_data: 历史销售数据 (30天, 5种饮料)，如果为None则生成模拟数据

    Returns:
        predictions: 预测的未来7天销售数据
    """
    from sales_data_processor import SalesDataGenerator

    # 如果没有提供历史数据，生成模拟数据
    if historical_data is None:
        generator = SalesDataGenerator()
        df = generator.generate_sales_data(num_days=30, start_date='2025-01-01')
        historical_data = df[generator.beverage_types].values

    # 标准化
    normalized_input = processor.transform(historical_data)

    # 转换为tensor
    x = torch.FloatTensor(normalized_input).unsqueeze(0)

    # 预测
    with torch.no_grad():
        pred = model(x)
        pred = pred.squeeze(0).numpy()

    # 反标准化
    predictions = processor.inverse_transform(pred)

    return predictions, historical_data


def recommend_parameters(transformer_predictions, model_dir='./data'):
    """
    根据 Transformer 预测推荐参数

    Args:
        transformer_predictions: Transformer 预测的销售数据 (7天, 5种饮料)
        model_dir: LightGBM 模型目录

    Returns:
        recommendations: 推荐的参数字典
    """
    # 检查模型文件
    meta_path = os.path.join(model_dir, 'lgb_meta.joblib')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"LightGBM 模型不存在: {meta_path}\n请先运行 lightgbm_train.py")

    # 使用封装的函数
    recommendations = recommend_parameters_from_prediction(transformer_predictions, model_dir)

    return recommendations


def display_recommendations(recommendations, transformer_predictions):
    """
    展示推荐的参数

    Args:
        recommendations: 推荐的参数字典
        transformer_predictions: Transformer 预测数据
    """
    beverage_types = ['碳酸饮料', '果汁饮料', '茶饮料', '功能饮料', '矿泉水']

    print("\n" + "=" * 70)
    print("📊 参数推荐结果")
    print("=" * 70)

    # 1. 预测销售汇总
    print("\n📈 Transformer 预测销售数据 (未来7天):")
    print("-" * 50)
    total_sales = np.sum(transformer_predictions, axis=0)
    avg_sales = np.mean(transformer_predictions, axis=0)

    for i, bev in enumerate(beverage_types):
        print(f"  {bev}: 总计 {total_sales[i]:.0f} 升, 日均 {avg_sales[i]:.0f} 升")

    print(f"\n  所有饮料总计: {np.sum(total_sales):.0f} 升")

    # 2. 利润参数推荐
    print("\n💰 推荐利润参数 (元/升):")
    print("-" * 50)
    for bev, profit in recommendations['profits'].items():
        print(f"  {bev}: {profit} 元/升")

    # 3. 原料供应限制推荐
    print("\n📦 推荐原料供应限制 (千克):")
    print("-" * 50)
    for material, limit in recommendations['material_limits'].items():
        print(f"  {material}: {limit:.0f} 千克")

    # 4. 运输能力限制推荐
    print("\n🚛 推荐运输能力限制 (升):")
    print("-" * 50)
    for region, limit in recommendations['transport_limits'].items():
        print(f"  {region}: {limit:.0f} 升")

    # 5. 生产约束参数推荐
    print("\n⚙️ 推荐生产约束参数:")
    print("-" * 50)
    print(f"  最小生产比例: {recommendations['min_production_ratio']}")
    print(f"  最大生产倍数: {recommendations['max_production_multiplier']}")

    print("\n" + "=" * 70)


def visualize_recommendations(recommendations, transformer_predictions):
    """
    可视化推荐结果

    Args:
        recommendations: 推荐的参数字典
        transformer_predictions: Transformer 预测数据
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    beverage_types = ['碳酸饮料', '果汁饮料', '茶饮料', '功能饮料', '矿泉水']
    colors = ['#2E8B57', '#4682B4', '#DAA520', '#CD853F', '#708090']

    # 1. 预测销售趋势
    ax1 = axes[0, 0]
    days = range(1, len(transformer_predictions) + 1)
    for i, (bev, color) in enumerate(zip(beverage_types, colors)):
        ax1.plot(days, transformer_predictions[:, i], '-o', label=bev, color=color, markersize=6)
    ax1.set_xlabel('预测天数')
    ax1.set_ylabel('销量 (升)')
    ax1.set_title('Transformer 预测的未来7天销量')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 2. 推荐利润参数
    ax2 = axes[0, 1]
    profits = list(recommendations['profits'].values())
    ax2.bar(beverage_types, profits, color=colors)
    ax2.set_ylabel('利润 (元/升)')
    ax2.set_title('推荐的利润参数')
    ax2.tick_params(axis='x', rotation=45)
    for i, (bev, profit) in enumerate(zip(beverage_types, profits)):
        ax2.text(i, profit + 0.2, f'{profit}', ha='center', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. 推荐原料供应限制
    ax3 = axes[1, 0]
    materials = list(recommendations['material_limits'].keys())
    limits = list(recommendations['material_limits'].values())
    bars = ax3.bar(range(len(materials)), limits, color='forestgreen')
    ax3.set_xticks(range(len(materials)))
    ax3.set_xticklabels(materials, rotation=45, ha='right')
    ax3.set_ylabel('供应限制 (千克)')
    ax3.set_title('推荐的原料供应限制')
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. 推荐运输能力限制
    ax4 = axes[1, 1]
    regions = list(recommendations['transport_limits'].keys())
    transport = list(recommendations['transport_limits'].values())
    ax4.bar(regions, transport, color='steelblue')
    ax4.set_ylabel('运输限制 (升)')
    ax4.set_title('推荐的运输能力限制')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('./data/parameter_recommendations.png', dpi=150)
    plt.show()

    print("\n推荐结果图已保存至: ./data/parameter_recommendations.png")


def generate_optimization_config(recommendations) -> dict:
    """
    生成可直接用于优化模型的配置

    Args:
        recommendations: 推荐的参数

    Returns:
        config: 优化模型配置字典
    """
    config = {
        'profits': list(recommendations['profits'].values()),
        'material_limits': list(recommendations['material_limits'].values()),
        'transport_limits': list(recommendations['transport_limits'].values()),
        'min_production_ratio': recommendations['min_production_ratio'],
        'max_production_multiplier': recommendations['max_production_multiplier']
    }

    return config


def save_recommendations_to_file(recommendations, transformer_predictions, filepath='./data/recommended_params.json'):
    """
    保存推荐参数到文件

    Args:
        recommendations: 推荐的参数
        transformer_predictions: 预测数据
        filepath: 保存路径
    """
    import json

    output = {
        'transformer_predictions': {
            'daily_sales': transformer_predictions.tolist(),
            'total_sales': np.sum(transformer_predictions, axis=0).tolist(),
            'avg_sales': np.mean(transformer_predictions, axis=0).tolist()
        },
        'recommendations': recommendations,
        'optimization_config': generate_optimization_config(recommendations)
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n推荐参数已保存至: {filepath}")


def test_with_sample_data():
    """
    使用样本数据测试（不依赖 Transformer 模型）
    """
    print("\n" + "=" * 70)
    print("使用模拟数据测试 LightGBM 参数推荐")
    print("=" * 70)

    # 生成模拟的 Transformer 预测数据
    np.random.seed(42)
    simulated_predictions = np.array([
        [2100, 1600, 1300, 900, 2600],
        [2150, 1550, 1350, 850, 2700],
        [2200, 1700, 1400, 950, 2550],
        [2300, 1800, 1500, 1000, 2800],
        [2250, 1750, 1450, 920, 2650],
        [2180, 1650, 1380, 880, 2720],
        [2220, 1720, 1420, 940, 2680]
    ])

    print("\n模拟的 Transformer 预测数据:")
    beverage_types = ['碳酸饮料', '果汁饮料', '茶饮料', '功能饮料', '矿泉水']
    df = pd.DataFrame(simulated_predictions, columns=beverage_types)
    df.index = [f'第{i+1}天' for i in range(7)]
    print(df)

    # 推荐参数
    recommendations = recommend_parameters(simulated_predictions)

    # 展示结果
    display_recommendations(recommendations, simulated_predictions)

    # 可视化
    visualize_recommendations(recommendations, simulated_predictions)

    # 保存结果
    save_recommendations_to_file(recommendations, simulated_predictions)

    return recommendations, simulated_predictions


def test_with_transformer():
    """
    使用真实的 Transformer 模型预测并推荐参数
    """
    print("\n" + "=" * 70)
    print("使用 Transformer 预测进行参数推荐")
    print("=" * 70)

    # 加载 Transformer 模型
    print("\n[1/3] 加载 Transformer 模型...")
    transformer_model, processor = load_transformer_model()
    print("Transformer 模型加载成功!")

    # 获取预测
    print("\n[2/3] 生成销售预测...")
    predictions, historical_data = get_transformer_predictions(transformer_model, processor)

    print(f"历史数据形状: {historical_data.shape}")
    print(f"预测数据形状: {predictions.shape}")

    # 推荐参数
    print("\n[3/3] 根据预测推荐参数...")
    recommendations = recommend_parameters(predictions)

    # 展示结果
    display_recommendations(recommendations, predictions)

    # 可视化
    visualize_recommendations(recommendations, predictions)

    # 保存结果
    save_recommendations_to_file(recommendations, predictions)

    return recommendations, predictions


if __name__ == '__main__':
    if not LIGHTGBM_AVAILABLE:
        print("请先安装 LightGBM: pip install lightgbm")
        exit(1)

    print("=" * 70)
    print("LightGBM 参数推荐模型 - 测试与推荐")
    print("=" * 70)

    # 检查 LightGBM 模型
    lgb_model_path = './data/lgb_meta.joblib'
    if not os.path.exists(lgb_model_path):
        print(f"\n错误: LightGBM 模型不存在: {lgb_model_path}")
        print("请先运行 lightgbm_train.py 训练模型")
        exit(1)

    # 检查 Transformer 模型
    transformer_model_path = './data/best_transformer_model.pth'

    if os.path.exists(transformer_model_path):
        print("\n检测到 Transformer 模型，使用真实预测进行推荐")
        recommendations, predictions = test_with_transformer()
    else:
        print("\n未检测到 Transformer 模型，使用模拟数据进行测试")
        print("提示: 运行 transformer_train.py 训练 Transformer 模型后可获得更准确的推荐")
        recommendations, predictions = test_with_sample_data()

    # 生成优化配置
    config = generate_optimization_config(recommendations)

    print("\n" + "=" * 70)
    print("可直接用于优化模型的配置:")
    print("=" * 70)
    print(f"""
from beverage_optimization_model import model

# 应用推荐的参数
model.update_parameters({{
    'profits': {config['profits']},
    'material_limits': {config['material_limits']},
    'transport_limits': {config['transport_limits']},
    'min_production_ratio': {config['min_production_ratio']},
    'max_production_multiplier': {config['max_production_multiplier']}
}})

# 求解优化模型
solution = model.solve_model()
print(f"最大利润: {{solution['optimal_value']:.2f}} 元")
""")

    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)
