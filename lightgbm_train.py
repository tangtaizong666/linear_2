"""
LightGBM 参数推荐模型 - 训练脚本
参考 ResNet18 和 Transformer 项目结构设计

注意：训练时仅使用生成的数据集，不使用 Transformer 的预测数据
"""

import time
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("错误: LightGBM 未安装，请运行 'pip install lightgbm' 安装")

from lightgbm_model import ParameterRecommender, SalesFeatureExtractor
from lightgbm_data_processor import prepare_lightgbm_data, LightGBMDataGenerator


def train_val_data_process(n_samples=3000, val_ratio=0.2):
    """
    数据处理函数：生成并处理训练和验证数据

    注意：此函数仅使用生成的模拟数据集，不使用 Transformer 预测数据

    Args:
        n_samples: 生成样本数量
        val_ratio: 验证集比例

    Returns:
        X_train, X_val, y_train, y_val, feature_names, target_names
    """
    print("生成训练数据集...")
    print("注意：训练数据完全由数据生成器生成，不包含 Transformer 预测数据")

    X_train, X_test, y_train, y_test, feature_names, target_names = prepare_lightgbm_data(
        n_samples=n_samples,
        test_ratio=val_ratio
    )

    return X_train, X_test, y_train, y_test, feature_names, target_names


def train_model_process(X_train, X_val, y_train, y_val,
                        feature_names, target_names, params=None):
    """
    模型训练函数

    Args:
        X_train: 训练特征
        X_val: 验证特征
        y_train: 训练标签
        y_val: 验证标签
        feature_names: 特征名称
        target_names: 目标名称
        params: LightGBM 参数

    Returns:
        recommender: 训练好的模型
        train_process: 训练过程记录
    """
    print("\n" + "=" * 60)
    print("开始训练 LightGBM 参数推荐模型")
    print("=" * 60)

    # 开始计时
    since = time.time()

    # 创建模型
    recommender = ParameterRecommender()

    # 更新参数（如果提供）
    if params:
        recommender.lgb_params.update(params)

    # 训练模型
    train_metrics = recommender.fit(
        X_train, y_train,
        feature_names=feature_names,
        target_names=target_names,
        eval_set=(X_val, y_val)
    )

    # 验证集评估
    val_predictions = recommender.predict(X_val)
    val_mae = np.mean(np.abs(val_predictions - y_val), axis=0)

    # 计算总体指标
    train_mae_mean = np.mean(list(train_metrics.values()))
    val_mae_mean = np.mean(val_mae)

    # 计算耗时
    time_use = time.time() - since

    print("\n" + "-" * 60)
    print("训练完成!")
    print("-" * 60)
    print(f"训练耗时: {time_use//60:.0f}m {time_use%60:.0f}s")
    print(f"训练集平均 MAE: {train_mae_mean:.4f}")
    print(f"验证集平均 MAE: {val_mae_mean:.4f}")

    # 保存模型
    os.makedirs('./data', exist_ok=True)
    recommender.save_model('./data')

    # 创建训练记录
    train_process = pd.DataFrame({
        'target': target_names,
        'train_mae': list(train_metrics.values()),
        'val_mae': val_mae
    })

    train_process.to_csv('./data/lightgbm_training_history.csv', index=False)

    return recommender, train_process


def evaluate_model(recommender, X_test, y_test, target_names):
    """
    详细评估模型性能

    Args:
        recommender: 训练好的模型
        X_test: 测试特征
        y_test: 测试标签
        target_names: 目标名称

    Returns:
        evaluation_results: 评估结果字典
    """
    print("\n" + "=" * 60)
    print("模型评估")
    print("=" * 60)

    predictions = recommender.predict(X_test)

    results = {}

    # 按参数组评估
    param_groups = {
        '利润参数': (0, 5),
        '原料供应限制': (5, 10),
        '运输能力限制': (10, 15),
        '生产约束参数': (15, 17)
    }

    for group_name, (start, end) in param_groups.items():
        group_pred = predictions[:, start:end]
        group_true = y_test[:, start:end]

        mae = np.mean(np.abs(group_pred - group_true))
        rmse = np.sqrt(np.mean((group_pred - group_true) ** 2))
        mape = np.mean(np.abs((group_pred - group_true) / (group_true + 1e-6))) * 100

        results[group_name] = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }

        print(f"\n{group_name}:")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAPE: {mape:.2f}%")

    # 总体评估
    total_mae = np.mean(np.abs(predictions - y_test))
    total_rmse = np.sqrt(np.mean((predictions - y_test) ** 2))

    print(f"\n总体评估:")
    print(f"  MAE: {total_mae:.4f}")
    print(f"  RMSE: {total_rmse:.4f}")

    results['overall'] = {'MAE': total_mae, 'RMSE': total_rmse}

    return results


def matplot_training_results(train_process, recommender):
    """
    可视化训练结果

    Args:
        train_process: 训练过程记录
        recommender: 训练好的模型
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 各目标的MAE对比
    ax1 = axes[0, 0]
    x = range(len(train_process))
    width = 0.35

    ax1.bar([i - width/2 for i in x], train_process['train_mae'],
            width, label='训练集', color='steelblue')
    ax1.bar([i + width/2 for i in x], train_process['val_mae'],
            width, label='验证集', color='darkorange')

    ax1.set_xlabel('目标变量')
    ax1.set_ylabel('MAE')
    ax1.set_title('各目标变量的MAE对比')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'T{i+1}' for i in x], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 按参数组的MAE汇总
    ax2 = axes[0, 1]
    groups = ['利润参数\n(5个)', '原料限制\n(5个)', '运输限制\n(5个)', '生产约束\n(2个)']
    group_indices = [(0, 5), (5, 10), (10, 15), (15, 17)]

    train_group_mae = [train_process['train_mae'].iloc[s:e].mean() for s, e in group_indices]
    val_group_mae = [train_process['val_mae'].iloc[s:e].mean() for s, e in group_indices]

    x_groups = range(len(groups))
    ax2.bar([i - width/2 for i in x_groups], train_group_mae,
            width, label='训练集', color='steelblue')
    ax2.bar([i + width/2 for i in x_groups], val_group_mae,
            width, label='验证集', color='darkorange')

    ax2.set_xlabel('参数组')
    ax2.set_ylabel('平均 MAE')
    ax2.set_title('各参数组的平均MAE')
    ax2.set_xticks(x_groups)
    ax2.set_xticklabels(groups)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 特征重要性（利润参数模型）
    ax3 = axes[1, 0]
    importance_dict = recommender.get_feature_importance(top_n=10)
    first_target = list(importance_dict.keys())[0]
    importance_df = importance_dict[first_target]

    ax3.barh(importance_df['feature'], importance_df['importance'], color='forestgreen')
    ax3.set_xlabel('重要性')
    ax3.set_title(f'特征重要性 ({first_target}模型)')
    ax3.grid(True, alpha=0.3)

    # 4. 训练/验证MAE分布
    ax4 = axes[1, 1]
    ax4.boxplot([train_process['train_mae'], train_process['val_mae']],
                labels=['训练集', '验证集'])
    ax4.set_ylabel('MAE')
    ax4.set_title('MAE分布')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./data/lightgbm_training_results.png', dpi=150)
    plt.show()

    print("\n训练结果图已保存至: ./data/lightgbm_training_results.png")


def visualize_prediction_comparison(recommender, X_test, y_test, target_names, num_samples=5):
    """
    可视化预测与真实值的对比

    Args:
        recommender: 训练好的模型
        X_test: 测试特征
        y_test: 测试标签
        target_names: 目标名称
        num_samples: 展示样本数
    """
    predictions = recommender.predict(X_test[:num_samples])

    # 分组展示
    param_groups = {
        '利润参数': (0, 5, ['碳酸饮料', '果汁饮料', '茶饮料', '功能饮料', '矿泉水']),
        '原料供应限制': (5, 10, ['白砂糖', '浓缩果汁', '茶叶提取物', '功能成分', '包装材料']),
        '运输能力限制': (10, 15, ['道里区', '南岗区', '道外区', '香坊区', '松北区'])
    }

    for group_name, (start, end, labels) in param_groups.items():
        fig, axes = plt.subplots(1, num_samples, figsize=(4*num_samples, 4))
        fig.suptitle(f'{group_name} - 预测 vs 真实值', fontsize=14)

        for sample_idx in range(num_samples):
            ax = axes[sample_idx] if num_samples > 1 else axes

            pred_values = predictions[sample_idx, start:end]
            true_values = y_test[sample_idx, start:end]

            x = range(len(labels))
            width = 0.35

            ax.bar([i - width/2 for i in x], true_values, width, label='真实值', color='steelblue')
            ax.bar([i + width/2 for i in x], pred_values, width, label='预测值', color='coral')

            ax.set_title(f'样本 {sample_idx + 1}')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        safe_name = group_name.replace(' ', '_')
        plt.savefig(f'./data/lgb_comparison_{safe_name}.png', dpi=150)
        plt.show()

    print("\n预测对比图已保存至 ./data/ 目录")


if __name__ == '__main__':
    if not LIGHTGBM_AVAILABLE:
        print("请先安装 LightGBM: pip install lightgbm")
        exit(1)

    print("=" * 70)
    print("LightGBM 参数推荐模型 - 训练")
    print("=" * 70)
    print("\n重要说明：")
    print("  - 训练数据完全由数据生成器生成")
    print("  - 不使用 Transformer 的预测数据进行训练")
    print("  - 训练后，模型可以根据 Transformer 预测来推荐参数")
    print("=" * 70)

    # 训练参数配置
    N_SAMPLES = 3000    # 训练样本数
    VAL_RATIO = 0.2     # 验证集比例

    # LightGBM 参数
    LGB_PARAMS = {
        'n_estimators': 200,
        'max_depth': 8,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42
    }

    # 1. 数据准备
    print("\n[1/4] 准备训练数据...")
    X_train, X_val, y_train, y_val, feature_names, target_names = train_val_data_process(
        n_samples=N_SAMPLES,
        val_ratio=VAL_RATIO
    )

    print(f"\n数据集信息:")
    print(f"  训练集: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征")
    print(f"  验证集: {X_val.shape[0]} 样本")
    print(f"  目标数: {y_train.shape[1]} 个参数")

    # 2. 训练模型
    print("\n[2/4] 训练模型...")
    recommender, train_process = train_model_process(
        X_train, X_val, y_train, y_val,
        feature_names, target_names,
        params=LGB_PARAMS
    )

    # 3. 评估模型
    print("\n[3/4] 评估模型...")
    evaluation_results = evaluate_model(recommender, X_val, y_val, target_names)

    # 4. 可视化结果
    print("\n[4/4] 可视化训练结果...")
    matplot_training_results(train_process, recommender)
    visualize_prediction_comparison(recommender, X_val, y_val, target_names, num_samples=3)

    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)
    print("\n保存的文件:")
    print("  - ./data/lightgbm_data.npz              (训练数据)")
    print("  - ./data/lgb_*.joblib                   (模型文件)")
    print("  - ./data/lgb_meta.joblib                (模型元信息)")
    print("  - ./data/lightgbm_training_history.csv  (训练历史)")
    print("  - ./data/lightgbm_training_results.png  (训练结果图)")
    print("  - ./data/lgb_comparison_*.png           (预测对比图)")

    print("\n下一步:")
    print("  1. 运行 transformer_train.py 训练 Transformer 模型")
    print("  2. 运行 lightgbm_test.py 使用 Transformer 预测来推荐参数")
