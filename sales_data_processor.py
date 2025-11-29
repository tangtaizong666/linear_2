"""
饮料销售数据处理模块
用于生成模拟销售数据、数据预处理、创建训练数据集
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta
from typing import Tuple, List, Optional
import os


class BeverageSalesDataset(Dataset):
    """
    饮料销售数据集类

    将时间序列数据转换为监督学习格式：
    - 输入：过去N天的销售数据
    - 输出：未来M天的销售数据
    """
    def __init__(self, data: np.ndarray, input_seq_len: int = 30, output_seq_len: int = 7):
        """
        Args:
            data: 销售数据 (total_days, num_products)
            input_seq_len: 输入序列长度（历史天数）
            output_seq_len: 输出序列长度（预测天数）
        """
        self.data = torch.FloatTensor(data)
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len

        # 计算有效样本数
        self.total_samples = len(data) - input_seq_len - output_seq_len + 1

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # 输入序列
        x = self.data[idx:idx + self.input_seq_len]
        # 输出序列
        y = self.data[idx + self.input_seq_len:idx + self.input_seq_len + self.output_seq_len]
        return x, y


class SalesDataGenerator:
    """
    饮料销售数据生成器

    生成具有真实特征的模拟销售数据：
    - 趋势：长期增长/下降趋势
    - 季节性：周期性波动（周、月、季节）
    - 噪声：随机波动
    - 节假日效应：特殊日期销量变化
    """
    def __init__(self, beverage_types: List[str] = None):
        if beverage_types is None:
            self.beverage_types = ['碳酸饮料', '果汁饮料', '茶饮料', '功能饮料', '矿泉水']
        else:
            self.beverage_types = beverage_types

        self.num_products = len(self.beverage_types)

        # 各饮料的基础销量（升/天）
        self.base_sales = np.array([2000, 1500, 1200, 800, 2500])

        # 各饮料的季节性系数（夏季高、冬季低等）
        self.seasonal_amplitude = np.array([0.3, 0.4, 0.25, 0.2, 0.35])

        # 各饮料的周销售模式（周末vs工作日）
        self.weekend_factor = np.array([1.3, 1.2, 1.1, 1.4, 1.25])

    def generate_sales_data(self, num_days: int = 365, start_date: str = '2024-01-01',
                           trend_rate: float = 0.0001, noise_level: float = 0.1) -> pd.DataFrame:
        """
        生成模拟销售数据

        Args:
            num_days: 生成数据的天数
            start_date: 起始日期
            trend_rate: 每日趋势增长率
            noise_level: 噪声水平

        Returns:
            DataFrame: 包含日期和各饮料销量的数据框
        """
        dates = pd.date_range(start=start_date, periods=num_days, freq='D')
        sales_data = np.zeros((num_days, self.num_products))

        for day in range(num_days):
            date = dates[day]

            # 1. 基础销量
            daily_sales = self.base_sales.copy().astype(float)

            # 2. 长期趋势
            trend = 1 + trend_rate * day
            daily_sales *= trend

            # 3. 年度季节性（使用正弦函数模拟）
            day_of_year = date.dayofyear
            seasonal_factor = 1 + self.seasonal_amplitude * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            daily_sales *= seasonal_factor

            # 4. 周销售模式（周末效应）
            if date.weekday() >= 5:  # 周六、周日
                daily_sales *= self.weekend_factor

            # 5. 月度变化
            month_factor = 1 + 0.05 * np.sin(2 * np.pi * date.month / 12)
            daily_sales *= month_factor

            # 6. 节假日效应
            if self._is_holiday(date):
                daily_sales *= np.random.uniform(1.2, 1.5, self.num_products)

            # 7. 随机噪声
            noise = np.random.normal(1, noise_level, self.num_products)
            daily_sales *= np.maximum(noise, 0.5)  # 确保不会出现负值

            sales_data[day] = daily_sales

        # 创建DataFrame
        df = pd.DataFrame(sales_data, columns=self.beverage_types)
        df['日期'] = dates
        df = df[['日期'] + self.beverage_types]

        return df

    def _is_holiday(self, date) -> bool:
        """判断是否为节假日（简化版）"""
        # 元旦
        if date.month == 1 and date.day <= 3:
            return True
        # 春节（假设在2月初）
        if date.month == 2 and 1 <= date.day <= 7:
            return True
        # 五一
        if date.month == 5 and 1 <= date.day <= 5:
            return True
        # 国庆
        if date.month == 10 and 1 <= date.day <= 7:
            return True
        return False


class SalesDataProcessor:
    """
    销售数据预处理器

    提供数据标准化、反标准化、数据集划分等功能
    """
    def __init__(self):
        self.scaler_mean = None
        self.scaler_std = None
        self.is_fitted = False

    def fit(self, data: np.ndarray):
        """
        拟合标准化参数

        Args:
            data: 原始数据 (num_samples, num_features)
        """
        self.scaler_mean = np.mean(data, axis=0)
        self.scaler_std = np.std(data, axis=0)
        self.scaler_std[self.scaler_std == 0] = 1  # 避免除零
        self.is_fitted = True

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        标准化数据

        Args:
            data: 原始数据

        Returns:
            标准化后的数据
        """
        if not self.is_fitted:
            raise RuntimeError("请先调用fit()方法拟合标准化参数")
        return (data - self.scaler_mean) / self.scaler_std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        反标准化数据

        Args:
            data: 标准化后的数据

        Returns:
            原始尺度的数据
        """
        if not self.is_fitted:
            raise RuntimeError("请先调用fit()方法拟合标准化参数")
        return data * self.scaler_std + self.scaler_mean

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """拟合并转换"""
        self.fit(data)
        return self.transform(data)

    def save_scaler(self, filepath: str):
        """保存标准化参数"""
        np.savez(filepath, mean=self.scaler_mean, std=self.scaler_std)

    def load_scaler(self, filepath: str):
        """加载标准化参数"""
        data = np.load(filepath)
        self.scaler_mean = data['mean']
        self.scaler_std = data['std']
        self.is_fitted = True


def create_data_loaders(data: np.ndarray, input_seq_len: int = 30, output_seq_len: int = 7,
                        train_ratio: float = 0.7, val_ratio: float = 0.15,
                        batch_size: int = 32, num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练、验证、测试数据加载器

    Args:
        data: 销售数据 (total_days, num_products)
        input_seq_len: 输入序列长度
        output_seq_len: 输出序列长度
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        batch_size: 批次大小
        num_workers: 数据加载线程数

    Returns:
        train_loader, val_loader, test_loader
    """
    total_len = len(data)

    # 计算划分点
    train_end = int(total_len * train_ratio)
    val_end = int(total_len * (train_ratio + val_ratio))

    # 划分数据
    train_data = data[:train_end]
    val_data = data[train_end - input_seq_len:val_end]  # 包含一些重叠以保证连续性
    test_data = data[val_end - input_seq_len:]

    # 创建数据集
    train_dataset = BeverageSalesDataset(train_data, input_seq_len, output_seq_len)
    val_dataset = BeverageSalesDataset(val_data, input_seq_len, output_seq_len)
    test_dataset = BeverageSalesDataset(test_data, input_seq_len, output_seq_len)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def prepare_training_data(num_days: int = 730, input_seq_len: int = 30, output_seq_len: int = 7,
                          batch_size: int = 32, save_dir: str = './data') -> Tuple:
    """
    准备完整的训练数据

    Args:
        num_days: 生成数据的天数
        input_seq_len: 输入序列长度
        output_seq_len: 输出序列长度
        batch_size: 批次大小
        save_dir: 数据保存目录

    Returns:
        train_loader, val_loader, test_loader, processor, raw_df
    """
    # 创建数据目录
    os.makedirs(save_dir, exist_ok=True)

    # 生成数据
    generator = SalesDataGenerator()
    df = generator.generate_sales_data(num_days=num_days)

    # 保存原始数据
    df.to_csv(os.path.join(save_dir, 'sales_data.csv'), index=False, encoding='utf-8-sig')

    # 提取销售数据（排除日期列）
    sales_values = df[generator.beverage_types].values

    # 数据预处理
    processor = SalesDataProcessor()
    normalized_data = processor.fit_transform(sales_values)

    # 保存标准化参数
    processor.save_scaler(os.path.join(save_dir, 'scaler_params.npz'))

    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        normalized_data, input_seq_len, output_seq_len, batch_size=batch_size
    )

    print(f"数据准备完成:")
    print(f"  - 总天数: {num_days}")
    print(f"  - 训练样本数: {len(train_loader.dataset)}")
    print(f"  - 验证样本数: {len(val_loader.dataset)}")
    print(f"  - 测试样本数: {len(test_loader.dataset)}")
    print(f"  - 输入序列长度: {input_seq_len}")
    print(f"  - 输出序列长度: {output_seq_len}")
    print(f"  - 数据已保存至: {save_dir}")

    return train_loader, val_loader, test_loader, processor, df


if __name__ == '__main__':
    # 测试数据生成和处理
    print("=== 饮料销售数据生成测试 ===\n")

    # 生成数据
    generator = SalesDataGenerator()
    df = generator.generate_sales_data(num_days=365)

    print("生成的销售数据预览:")
    print(df.head(10))
    print(f"\n数据形状: {df.shape}")
    print(f"\n各饮料销量统计:")
    print(df.describe())

    # 测试数据预处理
    print("\n=== 数据预处理测试 ===\n")
    sales_values = df[generator.beverage_types].values

    processor = SalesDataProcessor()
    normalized_data = processor.fit_transform(sales_values)

    print(f"标准化后数据均值: {normalized_data.mean(axis=0)}")
    print(f"标准化后数据标准差: {normalized_data.std(axis=0)}")

    # 测试数据加载器
    print("\n=== 数据加载器测试 ===\n")
    train_loader, val_loader, test_loader = create_data_loaders(
        normalized_data, input_seq_len=30, output_seq_len=7, batch_size=16
    )

    # 获取一个batch
    for x, y in train_loader:
        print(f"输入形状: {x.shape}")  # (batch_size, 30, 5)
        print(f"输出形状: {y.shape}")  # (batch_size, 7, 5)
        break

    print("\n数据处理模块测试通过！")
