"""
饮料销售预测 - Transformer模型测试与预测脚本
用于加载训练好的模型进行销售预测
"""

import torch
import torch.utils.data as Data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

from transformer_model import SalesForecasterEncoderOnly
from sales_data_processor import SalesDataProcessor, SalesDataGenerator


def test_data_process(test_days=60, input_seq_len=30):
    """
    准备测试数据

    Args:
        test_days: 测试数据天数
        input_seq_len: 输入序列长度

    Returns:
        test_data: 测试数据 (normalized)
        processor: 数据处理器
        raw_df: 原始数据DataFrame
    """
    # 生成测试数据
    generator = SalesDataGenerator()
    df = generator.generate_sales_data(num_days=test_days, start_date='2025-01-01')

    # 加载标准化参数
    processor = SalesDataProcessor()
    processor.load_scaler('./data/scaler_params.npz')

    # 标准化数据
    sales_values = df[generator.beverage_types].values
    normalized_data = processor.transform(sales_values)

    return normalized_data, processor, df


def test_model_process(model, test_data, processor, output_seq_len=7):
    """
    测试模型性能

    Args:
        model: 训练好的模型
        test_data: 测试数据 (normalized)
        processor: 数据处理器
        output_seq_len: 预测序列长度

    Returns:
        metrics: 评估指标字典
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    input_seq_len = model.input_seq_len
    total_samples = len(test_data) - input_seq_len - output_seq_len + 1

    all_predictions = []
    all_targets = []

    print(f"测试样本数: {total_samples}")

    with torch.no_grad():
        for i in range(total_samples):
            # 准备输入数据
            x = test_data[i:i + input_seq_len]
            y = test_data[i + input_seq_len:i + input_seq_len + output_seq_len]

            x_tensor = torch.FloatTensor(x).unsqueeze(0).to(device)

            # 预测
            pred = model(x_tensor)
            pred = pred.squeeze(0).cpu().numpy()

            all_predictions.append(pred)
            all_targets.append(y)

    # 转换为numpy数组
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # 反标准化
    predictions_real = []
    targets_real = []

    for pred, target in zip(all_predictions, all_targets):
        predictions_real.append(processor.inverse_transform(pred))
        targets_real.append(processor.inverse_transform(target))

    predictions_real = np.array(predictions_real)
    targets_real = np.array(targets_real)

    # 计算评估指标
    mae = np.mean(np.abs(predictions_real - targets_real))
    mse = np.mean((predictions_real - targets_real) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((predictions_real - targets_real) / (targets_real + 1e-8))) * 100

    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape
    }

    print("\n" + "=" * 50)
    print("模型评估指标:")
    print("=" * 50)
    print(f"MAE (平均绝对误差): {mae:.2f} 升")
    print(f"MSE (均方误差): {mse:.2f}")
    print(f"RMSE (均方根误差): {rmse:.2f} 升")
    print(f"MAPE (平均绝对百分比误差): {mape:.2f}%")

    return metrics, predictions_real, targets_real


def predict_future_sales(model, recent_data, processor, predict_days=7):
    """
    预测未来销售数据

    Args:
        model: 训练好的模型
        recent_data: 最近的销售数据 (raw, 未标准化)
        processor: 数据处理器
        predict_days: 预测天数

    Returns:
        predictions: 预测结果DataFrame
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    # 标准化输入数据
    normalized_input = processor.transform(recent_data)

    # 转换为tensor
    x = torch.FloatTensor(normalized_input).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x)
        pred = pred.squeeze(0).cpu().numpy()

    # 反标准化
    predictions_real = processor.inverse_transform(pred)

    # 创建结果DataFrame
    beverage_types = ['碳酸饮料', '果汁饮料', '茶饮料', '功能饮料', '矿泉水']

    # 生成预测日期
    start_date = datetime.now() + timedelta(days=1)
    dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(predict_days)]

    result_df = pd.DataFrame(predictions_real[:predict_days], columns=beverage_types)
    result_df.insert(0, '日期', dates[:len(result_df)])

    return result_df


def visualize_test_results(predictions, targets, beverage_types, num_samples=5):
    """
    可视化测试结果

    Args:
        predictions: 预测值数组
        targets: 真实值数组
        beverage_types: 饮料类型列表
        num_samples: 展示的样本数
    """
    for sample_idx in range(min(num_samples, len(predictions))):
        pred = predictions[sample_idx]
        target = targets[sample_idx]

        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        fig.suptitle(f'测试样本 {sample_idx + 1}: 预测 vs 真实值', fontsize=14)

        for j, (ax, name) in enumerate(zip(axes, beverage_types)):
            days = range(1, len(target) + 1)
            ax.plot(days, target[:, j], 'b-o', label='真实值', markersize=6)
            ax.plot(days, pred[:, j], 'r--s', label='预测值', markersize=6)
            ax.set_title(name)
            ax.set_xlabel('预测天数')
            ax.set_ylabel('销量 (升)')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # 添加误差标注
            mae = np.mean(np.abs(pred[:, j] - target[:, j]))
            ax.text(0.02, 0.98, f'MAE: {mae:.1f}',
                    transform=ax.transAxes, fontsize=8,
                    verticalalignment='top')

        plt.tight_layout()
        plt.savefig(f'./data/test_result_{sample_idx + 1}.png', dpi=150)
        plt.show()


def display_prediction_results(predictions_df):
    """
    展示预测结果

    Args:
        predictions_df: 预测结果DataFrame
    """
    beverage_types = ['碳酸饮料', '果汁饮料', '茶饮料', '功能饮料', '矿泉水']

    print("\n" + "=" * 70)
    print("未来销售预测结果")
    print("=" * 70)
    print(predictions_df.to_string(index=False))

    # 计算汇总统计
    print("\n" + "-" * 70)
    print("预测销量汇总:")
    print("-" * 70)

    for beverage in beverage_types:
        total = predictions_df[beverage].sum()
        avg = predictions_df[beverage].mean()
        print(f"{beverage}: 总计 {total:.0f} 升, 日均 {avg:.0f} 升")

    print(f"\n所有饮料总预测销量: {predictions_df[beverage_types].values.sum():.0f} 升")

    # 可视化预测结果
    plt.figure(figsize=(12, 6))

    x = range(len(predictions_df))
    width = 0.15
    colors = ['#2E8B57', '#4682B4', '#DAA520', '#CD853F', '#708090']

    for i, (beverage, color) in enumerate(zip(beverage_types, colors)):
        plt.bar([xi + i * width for xi in x], predictions_df[beverage],
                width=width, label=beverage, color=color)

    plt.xlabel('预测日期')
    plt.ylabel('销量 (升)')
    plt.title('未来7天各饮料销售预测')
    plt.xticks([xi + 2 * width for xi in x], predictions_df['日期'], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('./data/future_prediction.png', dpi=150)
    plt.show()

    print("\n预测图表已保存至: ./data/future_prediction.png")


if __name__ == '__main__':
    print("=" * 60)
    print("饮料销售预测 - Transformer模型测试")
    print("=" * 60)

    # 检查模型文件是否存在
    model_path = './data/best_transformer_model.pth'
    if not os.path.exists(model_path):
        print(f"\n错误: 未找到模型文件 {model_path}")
        print("请先运行 transformer_train.py 训练模型")
        exit(1)

    # 模型参数（需要与训练时一致）
    INPUT_DIM = 5
    INPUT_SEQ_LEN = 30
    OUTPUT_SEQ_LEN = 7

    beverage_types = ['碳酸饮料', '果汁饮料', '茶饮料', '功能饮料', '矿泉水']

    # 1. 创建模型并加载权重
    print("\n[1/4] 加载模型...")
    model = SalesForecasterEncoderOnly(
        input_dim=INPUT_DIM,
        d_model=128,
        num_heads=8,
        num_layers=4,
        d_ff=512,
        input_seq_len=INPUT_SEQ_LEN,
        output_seq_len=OUTPUT_SEQ_LEN,
        dropout=0.1
    )

    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    print("模型加载成功!")

    # 2. 准备测试数据
    print("\n[2/4] 准备测试数据...")
    test_data, processor, raw_df = test_data_process(
        test_days=60,
        input_seq_len=INPUT_SEQ_LEN
    )
    print(f"测试数据形状: {test_data.shape}")

    # 3. 模型测试
    print("\n[3/4] 测试模型性能...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    metrics, predictions, targets = test_model_process(
        model, test_data, processor, OUTPUT_SEQ_LEN
    )

    # 可视化测试结果
    visualize_test_results(predictions, targets, beverage_types, num_samples=3)

    # 4. 预测未来销售
    print("\n[4/4] 预测未来销售...")

    # 使用最近30天的数据进行预测
    recent_sales = raw_df[beverage_types].values[-INPUT_SEQ_LEN:]
    predictions_df = predict_future_sales(model, recent_sales, processor, predict_days=OUTPUT_SEQ_LEN)

    # 展示预测结果
    display_prediction_results(predictions_df)

    # 保存预测结果
    predictions_df.to_csv('./data/future_predictions.csv', index=False, encoding='utf-8-sig')
    print("\n预测结果已保存至: ./data/future_predictions.csv")

    # 交互式预测示例
    print("\n" + "=" * 60)
    print("交互式预测示例:")
    print("=" * 60)
    print("\n您可以使用以下代码进行自定义预测:")
    print("""
from transformer_test import predict_future_sales
from transformer_model import SalesForecasterEncoderOnly
from sales_data_processor import SalesDataProcessor
import torch
import numpy as np

# 加载模型
model = SalesForecasterEncoderOnly(input_dim=5, d_model=128, num_heads=8,
                                   num_layers=4, d_ff=512, input_seq_len=30,
                                   output_seq_len=7, dropout=0.1)
model.load_state_dict(torch.load('./data/best_transformer_model.pth'))

# 加载数据处理器
processor = SalesDataProcessor()
processor.load_scaler('./data/scaler_params.npz')

# 准备您的历史销售数据（最近30天，5种饮料）
# recent_data形状: (30, 5)
recent_data = np.array([...])  # 您的数据

# 预测未来7天销量
predictions = predict_future_sales(model, recent_data, processor, predict_days=7)
print(predictions)
""")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
