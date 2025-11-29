"""
饮料销售预测 - Transformer模型训练脚本
参考ResNet18项目结构设计
"""

import copy
import time
import os

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from transformer_model import SalesForecasterTransformer, SalesForecasterEncoderOnly
from sales_data_processor import (
    SalesDataGenerator, SalesDataProcessor,
    create_data_loaders, prepare_training_data
)


def train_val_data_process(num_days=730, input_seq_len=30, output_seq_len=7, batch_size=32):
    """
    数据处理函数：生成并处理训练和验证数据

    Args:
        num_days: 生成数据天数
        input_seq_len: 输入序列长度
        output_seq_len: 输出序列长度
        batch_size: 批次大小

    Returns:
        train_dataloader, val_dataloader, processor
    """
    # 生成销售数据
    generator = SalesDataGenerator()
    df = generator.generate_sales_data(num_days=num_days)

    # 保存原始数据
    os.makedirs('./data', exist_ok=True)
    df.to_csv('./data/sales_data.csv', index=False, encoding='utf-8-sig')

    # 提取销售数据
    sales_values = df[generator.beverage_types].values

    # 数据标准化
    processor = SalesDataProcessor()
    normalized_data = processor.fit_transform(sales_values)

    # 保存标准化参数
    processor.save_scaler('./data/scaler_params.npz')

    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        normalized_data,
        input_seq_len=input_seq_len,
        output_seq_len=output_seq_len,
        batch_size=batch_size
    )

    return train_loader, val_loader, processor


def train_model_process(model, train_dataloader, val_dataloader, num_epochs, model_type='encoder_only'):
    """
    模型训练函数

    Args:
        model: Transformer模型
        train_dataloader: 训练数据加载器
        val_dataloader: 验证数据加载器
        num_epochs: 训练轮数
        model_type: 模型类型 ('encoder_only' 或 'full')

    Returns:
        train_process: 训练过程记录DataFrame
    """
    # 设定训练设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # 损失函数：MSE用于回归任务
    criterion = nn.MSELoss()

    # 将模型放入训练设备
    model = model.to(device)

    # 复制当前模型参数
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化参数
    best_loss = float('inf')
    train_loss_all = []
    val_loss_all = []
    train_mae_all = []
    val_mae_all = []

    # 记录开始时间
    since = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs-1}")
        print("-" * 50)

        # 初始化每个epoch的统计量
        train_loss = 0.0
        train_mae = 0.0
        train_num = 0
        val_loss = 0.0
        val_mae = 0.0
        val_num = 0

        # ========== 训练阶段 ==========
        model.train()
        for step, (b_x, b_y) in enumerate(train_dataloader):
            # 将数据放入设备
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            # 前向传播
            if model_type == 'encoder_only':
                output = model(b_x)
            else:
                # 完整Transformer使用teacher forcing
                output = model(b_x, b_y)

            # 计算损失
            loss = criterion(output, b_y)

            # 梯度清零
            optimizer.zero_grad()

            # 反向传播
            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 更新参数
            optimizer.step()

            # 累计损失
            train_loss += loss.item() * b_x.size(0)
            train_mae += torch.mean(torch.abs(output - b_y)).item() * b_x.size(0)
            train_num += b_x.size(0)

        # ========== 验证阶段 ==========
        model.eval()
        with torch.no_grad():
            for step, (b_x, b_y) in enumerate(val_dataloader):
                b_x = b_x.to(device)
                b_y = b_y.to(device)

                # 前向传播
                if model_type == 'encoder_only':
                    output = model(b_x)
                else:
                    output = model(b_x, b_y)

                # 计算损失
                loss = criterion(output, b_y)

                val_loss += loss.item() * b_x.size(0)
                val_mae += torch.mean(torch.abs(output - b_y)).item() * b_x.size(0)
                val_num += b_x.size(0)

        # 计算平均损失
        epoch_train_loss = train_loss / train_num
        epoch_train_mae = train_mae / train_num
        epoch_val_loss = val_loss / val_num
        epoch_val_mae = val_mae / val_num

        # 记录损失
        train_loss_all.append(epoch_train_loss)
        val_loss_all.append(epoch_val_loss)
        train_mae_all.append(epoch_train_mae)
        val_mae_all.append(epoch_val_mae)

        # 打印训练信息
        print(f"{epoch} Train Loss: {epoch_train_loss:.6f} | Train MAE: {epoch_train_mae:.6f}")
        print(f"{epoch} Val Loss: {epoch_val_loss:.6f} | Val MAE: {epoch_val_mae:.6f}")

        # 更新学习率
        scheduler.step(epoch_val_loss)

        # 保存最佳模型
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"  -> 保存最佳模型 (Val Loss: {best_loss:.6f})")

        # 计算耗时
        time_use = time.time() - since
        print(f"训练耗时: {time_use//60:.0f}m {time_use%60:.0f}s")
        print()

    # 保存最佳模型
    os.makedirs('./data', exist_ok=True)
    torch.save(best_model_wts, './data/best_transformer_model.pth')
    print(f"\n最佳模型已保存至: ./data/best_transformer_model.pth")
    print(f"最佳验证损失: {best_loss:.6f}")

    # 返回训练过程记录
    train_process = pd.DataFrame({
        'epoch': range(num_epochs),
        'train_loss': train_loss_all,
        'val_loss': val_loss_all,
        'train_mae': train_mae_all,
        'val_mae': val_mae_all
    })

    return train_process


def matplot_loss_mae(train_process):
    """
    可视化训练过程中的损失和MAE

    Args:
        train_process: 训练过程记录DataFrame
    """
    plt.figure(figsize=(14, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_process['epoch'], train_process['train_loss'], 'ro-', label='Train Loss')
    plt.plot(train_process['epoch'], train_process['val_loss'], 'bs-', label='Val Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('训练和验证损失曲线')
    plt.grid(True, alpha=0.3)

    # MAE曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_process['epoch'], train_process['train_mae'], 'ro-', label='Train MAE')
    plt.plot(train_process['epoch'], train_process['val_mae'], 'bs-', label='Val MAE')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('训练和验证MAE曲线')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./data/training_curves.png', dpi=150)
    plt.show()
    print("训练曲线已保存至: ./data/training_curves.png")


def visualize_predictions(model, val_dataloader, processor, num_samples=3, model_type='encoder_only'):
    """
    可视化模型预测结果

    Args:
        model: 训练好的模型
        val_dataloader: 验证数据加载器
        processor: 数据处理器（用于反标准化）
        num_samples: 展示的样本数
        model_type: 模型类型
    """
    device = next(model.parameters()).device
    model.eval()

    beverage_types = ['碳酸饮料', '果汁饮料', '茶饮料', '功能饮料', '矿泉水']

    # 获取一些样本
    for i, (x, y) in enumerate(val_dataloader):
        if i >= num_samples:
            break

        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            if model_type == 'encoder_only':
                pred = model(x)
            else:
                pred = model.predict(x)

        # 取第一个样本进行可视化
        x_sample = x[0].cpu().numpy()
        y_sample = y[0].cpu().numpy()
        pred_sample = pred[0].cpu().numpy()

        # 反标准化
        y_real = processor.inverse_transform(y_sample)
        pred_real = processor.inverse_transform(pred_sample)

        # 绘图
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        fig.suptitle(f'样本 {i+1}: 预测 vs 真实值', fontsize=14)

        for j, (ax, name) in enumerate(zip(axes, beverage_types)):
            days = range(1, len(y_real) + 1)
            ax.plot(days, y_real[:, j], 'b-o', label='真实值', markersize=4)
            ax.plot(days, pred_real[:, j], 'r--s', label='预测值', markersize=4)
            ax.set_title(name)
            ax.set_xlabel('天数')
            ax.set_ylabel('销量 (升)')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'./data/prediction_sample_{i+1}.png', dpi=150)
        plt.show()

    print("预测可视化图已保存至 ./data/ 目录")


if __name__ == '__main__':
    print("=" * 60)
    print("饮料销售预测 - Transformer模型训练")
    print("=" * 60)

    # 模型参数配置
    INPUT_SEQ_LEN = 30   # 使用过去30天数据
    OUTPUT_SEQ_LEN = 7   # 预测未来7天
    NUM_EPOCHS = 50      # 训练轮数
    BATCH_SIZE = 32      # 批次大小
    NUM_DAYS = 730       # 生成2年数据

    # 饮料种类数
    INPUT_DIM = 5  # 碳酸饮料、果汁饮料、茶饮料、功能饮料、矿泉水

    # 1. 数据准备
    print("\n[1/4] 准备训练数据...")
    train_dataloader, val_dataloader, processor = train_val_data_process(
        num_days=NUM_DAYS,
        input_seq_len=INPUT_SEQ_LEN,
        output_seq_len=OUTPUT_SEQ_LEN,
        batch_size=BATCH_SIZE
    )

    print(f"训练集批次数: {len(train_dataloader)}")
    print(f"验证集批次数: {len(val_dataloader)}")

    # 2. 创建模型
    print("\n[2/4] 创建Transformer模型...")

    # 使用简化版编码器模型（训练更快）
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

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 3. 训练模型
    print("\n[3/4] 开始训练模型...")
    train_process = train_model_process(
        model, train_dataloader, val_dataloader,
        num_epochs=NUM_EPOCHS, model_type='encoder_only'
    )

    # 保存训练记录
    train_process.to_csv('./data/training_history.csv', index=False)

    # 4. 可视化训练过程
    print("\n[4/4] 可视化训练结果...")
    matplot_loss_mae(train_process)

    # 加载最佳模型并可视化预测结果
    model.load_state_dict(torch.load('./data/best_transformer_model.pth'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    visualize_predictions(model, val_dataloader, processor, num_samples=3, model_type='encoder_only')

    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print("\n保存的文件:")
    print("  - ./data/sales_data.csv         (原始销售数据)")
    print("  - ./data/scaler_params.npz      (标准化参数)")
    print("  - ./data/best_transformer_model.pth (最佳模型)")
    print("  - ./data/training_history.csv   (训练历史)")
    print("  - ./data/training_curves.png    (训练曲线)")
    print("  - ./data/prediction_sample_*.png (预测可视化)")
