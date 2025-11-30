"""
Streamlit 机器学习功能模块
提供模型训练和智能参数优化的完整页面功能
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go

# 检查依赖是否可用
TORCH_AVAILABLE = False
LIGHTGBM_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    import lightgbm
    LIGHTGBM_AVAILABLE = True
except ImportError:
    pass


# ==================== 参数限制常量（与模型保持一致）====================

PARAM_LIMITS = {
    # 利润参数限制 (元/升)
    'profits': {
        '碳酸饮料': (5.0, 15.0),
        '果汁饮料': (8.0, 18.0),
        '茶饮料': (6.0, 16.0),
        '功能饮料': (10.0, 25.0),
        '矿泉水': (3.0, 10.0),
    },
    # 原料供应限制 (千克)
    'material_limits': {
        '白砂糖': (8000, 25000),
        '浓缩果汁': (4000, 15000),
        '茶叶提取物': (3000, 12000),
        '功能成分': (1000, 5000),
        '包装材料': (8000, 20000),
    },
    # 运输能力限制 (升)
    'transport_limits': {
        '道里区': (2000, 5000),
        '南岗区': (1500, 4000),
        '道外区': (1200, 3500),
        '香坊区': (1000, 3000),
        '松北区': (600, 2000),
    },
    # 生产约束参数
    'min_production_ratio': (0.5, 0.95),
    'max_production_multiplier': (1.2, 2.5),
}


# ==================== 数据模板生成 ====================

def generate_transformer_template():
    """生成 Transformer 训练数据模板 - 包含各地区销售数据"""
    # 生成示例数据（30天，5种饮料，5个地区）
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    np.random.seed(42)

    # 地区列表
    regions = ['道里区', '南岗区', '道外区', '香坊区', '松北区']
    beverages = ['碳酸饮料', '果汁饮料', '茶饮料', '功能饮料', '矿泉水']

    # 各饮料在各地区的基础销量范围
    base_sales = {
        '碳酸饮料': {'道里区': (400, 550), '南岗区': (500, 650), '道外区': (350, 480), '香坊区': (280, 400), '松北区': (180, 280)},
        '果汁饮料': {'道里区': (280, 380), '南岗区': (420, 550), '道外区': (320, 420), '香坊区': (200, 300), '松北区': (80, 150)},
        '茶饮料': {'道里区': (320, 420), '南岗区': (280, 380), '道外区': (220, 320), '香坊区': (200, 300), '松北区': (60, 120)},
        '功能饮料': {'道里区': (200, 300), '南岗区': (180, 280), '道外区': (120, 200), '香坊区': (60, 120), '松北区': (30, 70)},
        '矿泉水': {'道里区': (350, 480), '南岗区': (550, 700), '道外区': (600, 780), '香坊区': (400, 550), '松北区': (200, 350)},
    }

    data = {'日期': dates.strftime('%Y-%m-%d')}

    # 为每个饮料的每个地区生成销售数据
    for bev in beverages:
        for region in regions:
            col_name = f'{bev}_{region}'
            low, high = base_sales[bev][region]
            # 添加一些随机波动和趋势
            base = np.random.uniform(low, high, 30)
            # 添加周末效应（周六日销量增加10-20%）
            weekend_mask = np.array([d.weekday() >= 5 for d in dates])
            base[weekend_mask] *= np.random.uniform(1.1, 1.2, weekend_mask.sum())
            data[col_name] = base.astype(int)

    # 添加各饮料的总销量列（可选，便于查看）
    df = pd.DataFrame(data)
    for bev in beverages:
        bev_cols = [f'{bev}_{r}' for r in regions]
        df[f'{bev}_总计'] = df[bev_cols].sum(axis=1)

    return df


def generate_lightgbm_template():
    """生成 LightGBM 训练数据模板 - 包含详细场景索引，参数严格在模型限制范围内"""
    np.random.seed(42)
    n_samples = 10  # 示例10条

    # 生成有意义的场景索引
    scenarios = [
        '2024年1月_春节前旺季',
        '2024年2月_春节期间',
        '2024年3月_节后淡季',
        '2024年4月_春季促销',
        '2024年5月_劳动节档期',
        '2024年6月_夏季开始',
        '2024年7月_暑期旺季',
        '2024年8月_高温期',
        '2024年9月_开学季',
        '2024年10月_国庆档期',
    ]

    # 地区列表
    regions = ['道里区', '南岗区', '道外区', '香坊区', '松北区']
    beverages = ['碳酸饮料', '果汁饮料', '茶饮料', '功能饮料', '矿泉水']
    materials = ['白砂糖', '浓缩果汁', '茶叶提取物', '功能成分', '包装材料']

    data = {'场景': scenarios}

    # 输入特征：各饮料各地区的7天预测销量汇总
    sales_ranges = {
        '碳酸饮料': (12000, 18000),
        '果汁饮料': (9000, 14000),
        '茶饮料': (7000, 11000),
        '功能饮料': (4000, 8000),
        '矿泉水': (15000, 22000),
    }

    for bev in beverages:
        low, high = sales_ranges[bev]
        data[f'{bev}_7天总销量'] = np.random.uniform(low, high, n_samples).astype(int)

        # 各地区销量占比特征
        for region in regions:
            data[f'{bev}_{region}_占比'] = np.random.uniform(0.1, 0.35, n_samples).round(2)

    # 添加趋势特征
    data['整体销量趋势'] = np.random.choice(['上升', '平稳', '下降'], n_samples)
    data['季节性因子'] = np.random.uniform(0.8, 1.3, n_samples).round(2)

    # 输出标签：推荐参数（严格使用 PARAM_LIMITS 中的范围）
    # 利润参数
    for bev in beverages:
        low, high = PARAM_LIMITS['profits'][bev]
        data[f'{bev}_推荐利润'] = np.random.uniform(low, high, n_samples).round(1)

    # 原料供应限制
    for mat in materials:
        low, high = PARAM_LIMITS['material_limits'][mat]
        data[f'{mat}_供应限制'] = np.random.uniform(low, high, n_samples).astype(int)

    # 运输能力限制
    for region in regions:
        low, high = PARAM_LIMITS['transport_limits'][region]
        data[f'{region}_运输限制'] = np.random.uniform(low, high, n_samples).astype(int)

    # 生产约束参数（使用 PARAM_LIMITS 中的范围）
    min_low, min_high = PARAM_LIMITS['min_production_ratio']
    max_low, max_high = PARAM_LIMITS['max_production_multiplier']
    data['最小生产比例'] = np.random.uniform(min_low, min_high, n_samples).round(2)
    data['最大生产倍数'] = np.random.uniform(max_low, max_high, n_samples).round(1)

    df = pd.DataFrame(data)
    df = df.set_index('场景')

    return df


def convert_df_to_csv(df, include_index=False):
    """将 DataFrame 转换为 CSV 字节流"""
    return df.to_csv(index=include_index, encoding='utf-8-sig').encode('utf-8-sig')


def clip_params_to_limits(params_dict):
    """将预测的参数裁剪到模型允许的范围内"""
    clipped = {}

    beverages = ['碳酸饮料', '果汁饮料', '茶饮料', '功能饮料', '矿泉水']
    materials = ['白砂糖', '浓缩果汁', '茶叶提取物', '功能成分', '包装材料']
    regions = ['道里区', '南岗区', '道外区', '香坊区', '松北区']

    # 裁剪利润参数
    clipped['profits'] = {}
    for bev in beverages:
        low, high = PARAM_LIMITS['profits'][bev]
        val = params_dict.get('profits', {}).get(bev, (low + high) / 2)
        clipped['profits'][bev] = round(max(low, min(high, val)), 1)

    # 裁剪原料供应限制
    clipped['material_limits'] = {}
    for mat in materials:
        low, high = PARAM_LIMITS['material_limits'][mat]
        val = params_dict.get('material_limits', {}).get(mat, (low + high) / 2)
        clipped['material_limits'][mat] = int(max(low, min(high, val)))

    # 裁剪运输能力限制
    clipped['transport_limits'] = {}
    for region in regions:
        low, high = PARAM_LIMITS['transport_limits'][region]
        val = params_dict.get('transport_limits', {}).get(region, (low + high) / 2)
        clipped['transport_limits'][region] = int(max(low, min(high, val)))

    # 裁剪生产约束参数
    min_low, min_high = PARAM_LIMITS['min_production_ratio']
    max_low, max_high = PARAM_LIMITS['max_production_multiplier']

    min_ratio = params_dict.get('min_production_ratio', (min_low + min_high) / 2)
    max_mult = params_dict.get('max_production_multiplier', (max_low + max_high) / 2)

    clipped['min_production_ratio'] = round(max(min_low, min(min_high, min_ratio)), 2)
    clipped['max_production_multiplier'] = round(max(max_low, min(max_high, max_mult)), 1)

    return clipped


# ==================== 页面状态管理 ====================

def init_session_state():
    """初始化 session state"""
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'main'
    if 'training_model_expanded' not in st.session_state:
        st.session_state.training_model_expanded = False
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    if 'sync_params' not in st.session_state:
        st.session_state.sync_params = False


def check_model_status():
    """检查模型文件状态"""
    try:
        # 获取当前脚本所在目录
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, 'data')

        return {
            'transformer_model': os.path.exists(os.path.join(data_dir, 'best_transformer_model.pth')),
            'transformer_scaler': os.path.exists(os.path.join(data_dir, 'scaler_params.npz')),
            'lightgbm_model': os.path.exists(os.path.join(data_dir, 'lgb_meta.joblib')),
        }
    except Exception:
        # 如果出现任何错误，返回全部为 False
        return {
            'transformer_model': False,
            'transformer_scaler': False,
            'lightgbm_model': False,
        }


# ==================== 侧边栏导航 ====================

def sidebar_navigation():
    """侧边栏导航菜单"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🚀 高级功能")

    # 模型训练 - 可展开的子选项
    with st.sidebar.expander("🎓 模型训练", expanded=st.session_state.get('training_model_expanded', False)):
        st.caption("训练深度学习和机器学习模型")

        if st.button("🤖 Transformer 训练", key="nav_transformer", use_container_width=True):
            st.session_state.current_page = 'transformer_training'
            st.session_state.training_model_expanded = True
            st.rerun()

        if st.button("📊 LightGBM 训练", key="nav_lightgbm", use_container_width=True):
            st.session_state.current_page = 'lightgbm_training'
            st.session_state.training_model_expanded = True
            st.rerun()

        # 显示模型状态
        status = check_model_status()
        st.markdown("---")
        st.markdown("**模型状态:**")
        transformer_status = "✅ 已训练" if status['transformer_model'] else "⚪ 未训练"
        lightgbm_status = "✅ 已训练" if status['lightgbm_model'] else "⚪ 未训练"
        st.caption(f"Transformer: {transformer_status}")
        st.caption(f"LightGBM: {lightgbm_status}")

    # 智能参数优化
    if st.sidebar.button("🧠 智能参数优化", key="nav_smart_opt", use_container_width=True):
        st.session_state.current_page = 'smart_optimization'
        st.rerun()

    # 返回主页
    if st.session_state.current_page != 'main':
        st.sidebar.markdown("---")
        if st.sidebar.button("🏠 返回主页", key="nav_home", use_container_width=True):
            st.session_state.current_page = 'main'
            st.rerun()


# ==================== Transformer 训练页面 ====================

def page_transformer_training():
    """Transformer 模型训练页面"""
    st.markdown("# 🤖 Transformer 销售预测模型训练")
    st.markdown("---")

    # 检查 PyTorch 是否可用
    if not TORCH_AVAILABLE:
        st.warning("⚠️ PyTorch 未安装，无法训练 Transformer 模型")
        st.info("请运行以下命令安装：`pip install torch`")
        st.markdown("---")
        st.markdown("### 💡 提示")
        st.markdown("安装 PyTorch 后刷新页面即可使用训练功能。")
        return

    # 模型介绍
    with st.container():
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px;">
            <h3>📌 模型说明</h3>
            <p><strong>用途：</strong>基于历史销售数据，预测未来7天各饮料的销量</p>
            <p><strong>输入：</strong>过去30天的5种饮料销售数据</p>
            <p><strong>输出：</strong>未来7天的销售预测</p>
            <p><strong>应用场景：</strong>生产计划制定、原料采购规划、运输调度安排</p>
        </div>
        """, unsafe_allow_html=True)

    # 数据来源选择
    st.markdown("### 📊 选择数据来源")
    data_source = st.radio(
        "请选择训练数据来源：",
        ["🎲 使用模拟数据", "📁 上传自定义数据"],
        key="transformer_data_source",
        horizontal=True
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        if data_source == "📁 上传自定义数据":
            # 数据上传区域
            st.markdown("### 📁 上传训练数据")
            st.info("""
            **数据格式要求：**
            - CSV 文件格式
            - 必须包含列：日期 + 各饮料各地区销量
            - 列名格式：`饮料名_地区名`（如：碳酸饮料_道里区）
            - 地区：道里区、南岗区、道外区、香坊区、松北区
            - 饮料：碳酸饮料、果汁饮料、茶饮料、功能饮料、矿泉水
            - 销量单位：升
            - 建议至少 365 天数据
            """)

            uploaded_file = st.file_uploader(
                "选择 CSV 文件上传",
                type=['csv'],
                key="transformer_upload",
                help="上传包含历史销售数据的 CSV 文件"
            )

            # 显示上传的数据预览
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
                    st.success(f"✅ 文件上传成功！共 {len(df)} 条记录")
                    st.markdown("**数据预览：**")
                    st.dataframe(df.head(10), use_container_width=True)
                    st.session_state['transformer_data'] = df
                    st.session_state['transformer_data_ready'] = True
                except Exception as e:
                    st.error(f"❌ 文件读取失败: {str(e)}")
                    st.session_state['transformer_data_ready'] = False
            else:
                st.warning("⚠️ 请上传训练数据文件")
                st.session_state['transformer_data_ready'] = False

        else:  # 使用模拟数据
            st.markdown("### 🎲 模拟数据设置")
            st.info("系统将自动生成符合模型要求的模拟销售数据用于训练。")

            data_days = st.slider("生成数据天数", 365, 1095, 730, 365, key="t_data_days")

            # 显示数据预览（如果已生成）
            if 'transformer_data' in st.session_state and st.session_state.get('transformer_data_source_type') == 'simulated':
                st.success(f"✅ 已准备 {len(st.session_state['transformer_data'])} 天的模拟数据")
                st.markdown("**数据预览：**")
                st.dataframe(st.session_state['transformer_data'].head(10), use_container_width=True)

            if st.button("🔄 预览/刷新模拟数据", key="preview_transformer_data"):
                with st.spinner("正在生成模拟数据..."):
                    from sales_data_processor import SalesDataGenerator
                    generator = SalesDataGenerator()
                    df = generator.generate_sales_data(num_days=data_days)
                    st.session_state['transformer_data'] = df
                    st.session_state['transformer_data_source_type'] = 'simulated'
                    st.session_state['transformer_data_days'] = data_days
                    st.rerun()

            # 模拟数据始终可用
            st.session_state['transformer_data_ready'] = True

    with col2:
        # 下载模板
        st.markdown("### 📥 数据模板下载")
        template_df = generate_transformer_template()
        csv_data = convert_df_to_csv(template_df)

        st.download_button(
            label="⬇️ 下载数据模板",
            data=csv_data,
            file_name="transformer_data_template.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.markdown("**模板预览：**")
        st.dataframe(template_df.head(5), use_container_width=True)

        # 训练参数
        st.markdown("### ⚙️ 训练参数")
        epochs = st.number_input("训练轮数", 10, 200, 50, 10, key="t_epochs")
        batch_size = st.selectbox("批次大小", [16, 32, 64], index=1, key="t_batch")
        learning_rate = st.select_slider("学习率", [0.0001, 0.0005, 0.001, 0.005], value=0.001, key="t_lr")

    # 开始训练按钮
    st.markdown("---")
    st.markdown("### 🚀 开始训练")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🎯 开始训练 Transformer 模型", key="start_transformer_training",
                     use_container_width=True, type="primary"):
            if not TORCH_AVAILABLE:
                st.error("❌ PyTorch 未安装，请运行: pip install torch")
            elif data_source == "📁 上传自定义数据":
                # 使用上传数据模式
                if not st.session_state.get('transformer_data_ready', False) or 'transformer_data' not in st.session_state:
                    st.error("❌ 请先上传训练数据文件")
                else:
                    train_transformer_with_progress(
                        st.session_state['transformer_data'],
                        epochs, batch_size, learning_rate
                    )
            else:
                # 使用模拟数据模式 - 自动生成数据
                with st.spinner("正在准备模拟数据..."):
                    from sales_data_processor import SalesDataGenerator
                    generator = SalesDataGenerator()
                    data_days = st.session_state.get('transformer_data_days', 730)
                    df = generator.generate_sales_data(num_days=data_days)
                    st.session_state['transformer_data'] = df
                    st.session_state['transformer_data_source_type'] = 'simulated'
                train_transformer_with_progress(df, epochs, batch_size, learning_rate)


def train_transformer_with_progress(df, epochs, batch_size, learning_rate):
    """带进度条的 Transformer 训练"""
    import torch
    import torch.nn as nn
    import copy

    from transformer_model import SalesForecasterEncoderOnly
    from sales_data_processor import SalesDataProcessor, create_data_loaders

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建进度显示区域
    st.markdown("### 📈 训练进度")
    progress_container = st.container()

    with progress_container:
        status_text = st.empty()
        progress_bar = st.progress(0)
        metrics_placeholder = st.empty()
        chart_placeholder = st.empty()

    try:
        status_text.info("🔄 准备数据中...")

        # 保存原始数据
        os.makedirs('./data', exist_ok=True)
        df.to_csv('./data/sales_data.csv', index=False, encoding='utf-8-sig')

        # 提取销售数据
        beverage_cols = ['碳酸饮料', '果汁饮料', '茶饮料', '功能饮料', '矿泉水']
        sales_values = df[beverage_cols].values

        # 数据预处理
        processor = SalesDataProcessor()
        normalized_data = processor.fit_transform(sales_values)
        processor.save_scaler('./data/scaler_params.npz')

        # 创建数据加载器
        train_loader, val_loader, _ = create_data_loaders(
            normalized_data, input_seq_len=30, output_seq_len=7, batch_size=batch_size
        )

        status_text.info("🔄 初始化模型...")

        # 创建模型
        model = SalesForecasterEncoderOnly(
            input_dim=5, d_model=128, num_heads=8, num_layers=4,
            d_ff=512, input_seq_len=30, output_seq_len=7, dropout=0.1
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        best_loss = float('inf')
        best_model_wts = copy.deepcopy(model.state_dict())

        # 记录训练历史
        train_losses = []
        val_losses = []

        status_text.info("🚀 开始训练...")

        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            for b_x, b_y in train_loader:
                b_x, b_y = b_x.to(device), b_y.to(device)
                output = model(b_x)
                loss = criterion(output, b_y)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            # 验证阶段
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for b_x, b_y in val_loader:
                    b_x, b_y = b_x.to(device), b_y.to(device)
                    output = model(b_x)
                    val_loss += criterion(output, b_y).item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())

            # 更新进度
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)

            # 更新指标显示
            metrics_placeholder.markdown(f"""
            **当前进度：** Epoch {epoch + 1}/{epochs}

            | 指标 | 训练集 | 验证集 |
            |------|--------|--------|
            | Loss | {train_loss:.6f} | {val_loss:.6f} |
            | 最佳 Val Loss | {best_loss:.6f} | - |
            """)

            # 更新损失曲线
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=train_losses, mode='lines', name='训练损失'))
                fig.add_trace(go.Scatter(y=val_losses, mode='lines', name='验证损失'))
                fig.update_layout(
                    title="训练损失曲线",
                    xaxis_title="Epoch",
                    yaxis_title="Loss",
                    height=300
                )
                chart_placeholder.plotly_chart(fig, use_container_width=True)

        # 保存最佳模型
        torch.save(best_model_wts, './data/best_transformer_model.pth')

        status_text.empty()
        progress_bar.empty()

        st.success(f"""
        ✅ **训练完成！**
        - 最佳验证损失: {best_loss:.6f}
        - 模型已保存至: ./data/best_transformer_model.pth
        """)
        st.balloons()

    except Exception as e:
        st.error(f"❌ 训练失败: {str(e)}")


# ==================== LightGBM 训练页面 ====================

def page_lightgbm_training():
    """LightGBM 模型训练页面"""
    st.markdown("# 📊 LightGBM 参数推荐模型训练")
    st.markdown("---")

    # 检查 LightGBM 是否可用
    if not LIGHTGBM_AVAILABLE:
        st.warning("⚠️ LightGBM 未安装，无法训练参数推荐模型")
        st.info("请运行以下命令安装：`pip install lightgbm`")
        st.markdown("---")
        st.markdown("### 💡 提示")
        st.markdown("安装 LightGBM 后刷新页面即可使用训练功能。")
        return

    # 模型介绍
    with st.container():
        st.markdown("""
        <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
                    padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px;">
            <h3>📌 模型说明</h3>
            <p><strong>用途：</strong>根据销售预测数据，智能推荐最优的生产优化参数</p>
            <p><strong>输入：</strong>销售数据特征（总销量、平均销量、趋势等）</p>
            <p><strong>输出：</strong>推荐的利润参数、原料限制、运输限制、生产约束</p>
            <p><strong>应用场景：</strong>自动化参数调优、减少人工决策时间、提高优化效率</p>
        </div>
        """, unsafe_allow_html=True)

    # 数据来源选择
    st.markdown("### 📊 选择数据来源")
    data_source = st.radio(
        "请选择训练数据来源：",
        ["🎲 使用模拟数据", "📁 上传自定义数据"],
        key="lightgbm_data_source",
        horizontal=True
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        if data_source == "📁 上传自定义数据":
            # 数据上传区域
            st.markdown("### 📁 上传训练数据")
            st.info("""
            **数据格式要求：**
            - CSV 文件格式，首列为场景索引
            - **输入特征列：**
              - 各饮料7天总销量（如：碳酸饮料_7天总销量）
              - 各地区销量占比（如：碳酸饮料_道里区_占比）
              - 趋势和季节性因子
            - **输出标签列：**
              - 推荐利润参数、原料供应限制
              - 运输限制、生产约束参数
            - 详见数据模板
            """)

            uploaded_file = st.file_uploader(
                "选择 CSV 文件上传",
                type=['csv'],
                key="lightgbm_upload",
                help="上传包含特征和标签的训练数据"
            )

            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
                    st.success(f"✅ 文件上传成功！共 {len(df)} 条记录")
                    st.markdown("**数据预览：**")
                    st.dataframe(df.head(10), use_container_width=True)
                    st.session_state['lightgbm_data'] = df
                    st.session_state['lightgbm_data_ready'] = True
                except Exception as e:
                    st.error(f"❌ 文件读取失败: {str(e)}")
                    st.session_state['lightgbm_data_ready'] = False
            else:
                st.warning("⚠️ 请上传训练数据文件")
                st.session_state['lightgbm_data_ready'] = False

        else:  # 使用模拟数据
            st.markdown("### 🎲 模拟数据设置")
            st.info("系统将自动生成符合模型要求的模拟特征和标签数据用于训练。")

            n_samples = st.slider("生成样本数量", 1000, 10000, 3000, 500, key="l_samples")

            # 显示数据预览（如果已生成）
            if 'lightgbm_data' in st.session_state and st.session_state.get('lightgbm_data_source_type') == 'simulated':
                st.success(f"✅ 已准备 {len(st.session_state['lightgbm_data'])} 条模拟数据")
                st.markdown("**数据预览：**")
                st.dataframe(st.session_state['lightgbm_data'].head(10), use_container_width=True)

            if st.button("🔄 预览/刷新模拟数据", key="preview_lightgbm_data"):
                with st.spinner("正在生成模拟数据..."):
                    from lightgbm_data_processor import LightGBMDataGenerator
                    generator = LightGBMDataGenerator()
                    X, y = generator.generate_training_data(n_samples)

                    # 合并为 DataFrame
                    feature_names = generator.get_feature_names()
                    target_names = generator.get_target_names()

                    df_features = pd.DataFrame(X, columns=feature_names)
                    df_targets = pd.DataFrame(y, columns=target_names)
                    df = pd.concat([df_features, df_targets], axis=1)

                    st.session_state['lightgbm_data'] = df
                    st.session_state['lightgbm_X'] = X
                    st.session_state['lightgbm_y'] = y
                    st.session_state['lightgbm_feature_names'] = feature_names
                    st.session_state['lightgbm_target_names'] = target_names
                    st.session_state['lightgbm_data_source_type'] = 'simulated'
                    st.session_state['lightgbm_n_samples'] = n_samples
                    st.rerun()

            # 模拟数据始终可用
            st.session_state['lightgbm_data_ready'] = True

    with col2:
        # 下载模板
        st.markdown("### 📥 数据模板下载")
        template_df = generate_lightgbm_template()
        csv_data = convert_df_to_csv(template_df, include_index=True)  # 包含场景索引

        st.download_button(
            label="⬇️ 下载数据模板",
            data=csv_data,
            file_name="lightgbm_data_template.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.markdown("**模板预览：**")
        st.dataframe(template_df.head(5), use_container_width=True)

        # 训练参数
        st.markdown("### ⚙️ 训练参数")
        n_estimators = st.number_input("树的数量", 50, 500, 200, 50, key="l_estimators")
        max_depth = st.number_input("最大深度", 3, 15, 8, 1, key="l_depth")
        learning_rate = st.select_slider("学习率", [0.01, 0.03, 0.05, 0.1], value=0.05, key="l_lr")

    # 开始训练按钮
    st.markdown("---")
    st.markdown("### 🚀 开始训练")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🎯 开始训练 LightGBM 模型", key="start_lightgbm_training",
                     use_container_width=True, type="primary"):
            if not LIGHTGBM_AVAILABLE:
                st.error("❌ LightGBM 未安装，请运行: pip install lightgbm")
            elif data_source == "📁 上传自定义数据":
                # 使用上传数据模式
                if not st.session_state.get('lightgbm_data_ready', False) or 'lightgbm_data' not in st.session_state:
                    st.error("❌ 请先上传训练数据文件")
                else:
                    train_lightgbm_with_progress(n_estimators, max_depth, learning_rate)
            else:
                # 使用模拟数据模式 - 自动生成数据
                with st.spinner("正在准备模拟数据..."):
                    from lightgbm_data_processor import LightGBMDataGenerator
                    generator = LightGBMDataGenerator()
                    n_samples_val = st.session_state.get('lightgbm_n_samples', 3000)
                    X, y = generator.generate_training_data(n_samples_val)

                    feature_names = generator.get_feature_names()
                    target_names = generator.get_target_names()

                    df_features = pd.DataFrame(X, columns=feature_names)
                    df_targets = pd.DataFrame(y, columns=target_names)
                    df = pd.concat([df_features, df_targets], axis=1)

                    st.session_state['lightgbm_data'] = df
                    st.session_state['lightgbm_X'] = X
                    st.session_state['lightgbm_y'] = y
                    st.session_state['lightgbm_feature_names'] = feature_names
                    st.session_state['lightgbm_target_names'] = target_names
                    st.session_state['lightgbm_data_source_type'] = 'simulated'

                train_lightgbm_with_progress(n_estimators, max_depth, learning_rate)


def train_lightgbm_with_progress(n_estimators, max_depth, learning_rate):
    """带进度条的 LightGBM 训练"""
    from lightgbm_model import ParameterRecommender
    from lightgbm_data_processor import LightGBMDataGenerator

    # 创建进度显示区域
    st.markdown("### 📈 训练进度")
    progress_container = st.container()

    with progress_container:
        status_text = st.empty()
        progress_bar = st.progress(0)
        metrics_placeholder = st.empty()

    try:
        status_text.info("🔄 准备数据中...")

        # 检查是否有预处理的数据
        if 'lightgbm_X' in st.session_state:
            X = st.session_state['lightgbm_X']
            y = st.session_state['lightgbm_y']
            feature_names = st.session_state['lightgbm_feature_names']
            target_names = st.session_state['lightgbm_target_names']
        else:
            # 从上传的数据中提取
            df = st.session_state['lightgbm_data']
            generator = LightGBMDataGenerator()
            feature_names = generator.get_feature_names()
            target_names = generator.get_target_names()

            X = df[feature_names].values
            y = df[target_names].values

        # 划分训练集和测试集
        n_test = int(len(X) * 0.2)
        indices = np.random.permutation(len(X))
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        progress_bar.progress(0.1)
        status_text.info("🔄 初始化模型...")

        # 创建模型
        recommender = ParameterRecommender()
        recommender.lgb_params.update({
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate
        })

        progress_bar.progress(0.2)
        status_text.info("🚀 开始训练...")

        # 训练模型（带进度更新）
        total_targets = len(target_names)
        metrics_list = []

        for i, target_name in enumerate(target_names):
            import lightgbm as lgb

            model = lgb.LGBMRegressor(**recommender.lgb_params)
            model.fit(X_train, y_train[:, i])
            recommender.models[target_name] = model

            # 计算 MAE
            pred = model.predict(X_test)
            mae = np.mean(np.abs(pred - y_test[:, i]))
            metrics_list.append({'目标': target_name, 'MAE': mae})

            # 更新进度
            progress = 0.2 + 0.7 * (i + 1) / total_targets
            progress_bar.progress(progress)
            status_text.info(f"🚀 训练中... ({i+1}/{total_targets}) {target_name}")

            # 更新指标显示
            metrics_df = pd.DataFrame(metrics_list)
            metrics_placeholder.dataframe(metrics_df, use_container_width=True)

        recommender.feature_names = feature_names
        recommender.target_names = target_names
        recommender.is_fitted = True

        progress_bar.progress(0.95)
        status_text.info("🔄 保存模型...")

        # 保存模型
        os.makedirs('./data', exist_ok=True)
        recommender.save_model('./data')

        progress_bar.progress(1.0)
        status_text.empty()
        progress_bar.empty()

        avg_mae = np.mean([m['MAE'] for m in metrics_list])
        st.success(f"""
        ✅ **训练完成！**
        - 平均 MAE: {avg_mae:.4f}
        - 模型已保存至: ./data/
        """)
        st.balloons()

    except Exception as e:
        st.error(f"❌ 训练失败: {str(e)}")


# ==================== 智能参数优化页面 ====================

def page_smart_optimization(optimization_model):
    """智能参数优化页面"""
    st.markdown("# 🧠 智能参数优化")
    st.markdown("---")

    # 检查依赖是否可用
    if not TORCH_AVAILABLE:
        st.warning("⚠️ PyTorch 未安装，智能优化功能不可用")
        st.info("请运行以下命令安装：`pip install torch`")
        st.markdown("---")
        st.markdown("### 💡 替代方案")
        st.markdown("您可以使用侧边栏的 **参数设置** 手动调整优化参数，然后在主页点击「开始求解」。")
        return

    if not LIGHTGBM_AVAILABLE:
        st.warning("⚠️ LightGBM 未安装，智能优化功能不可用")
        st.info("请运行以下命令安装：`pip install lightgbm`")
        st.markdown("---")
        st.markdown("### 💡 替代方案")
        st.markdown("您可以使用侧边栏的 **参数设置** 手动调整优化参数，然后在主页点击「开始求解」。")
        return

    # 功能介绍
    with st.container():
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px;">
            <h3>📌 功能说明</h3>
            <p><strong>工作原理：</strong></p>
            <ol>
                <li>Transformer 模型预测未来7天的销售数据</li>
                <li>LightGBM 模型根据预测数据推荐最优参数</li>
                <li>用户可选择将推荐参数同步到优化模型</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    # 检查模型状态
    status = check_model_status()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📋 模型状态检查")
        if status['transformer_model']:
            st.success("✅ Transformer 模型已就绪")
        else:
            st.info("ℹ️ Transformer 模型未训练")
            st.caption("请先在「模型训练」中训练 Transformer 模型")

    with col2:
        st.markdown("###  ")  # 占位
        if status['lightgbm_model']:
            st.success("✅ LightGBM 模型已就绪")
        else:
            st.info("ℹ️ LightGBM 模型未训练")
            st.caption("请先在「模型训练」中训练 LightGBM 模型")

    models_ready = status['transformer_model'] and status['lightgbm_model']

    if not models_ready:
        st.markdown("---")
        st.markdown("### 📝 训练指南")
        st.markdown("""
        要使用智能参数优化，请按以下顺序训练模型：

        1. **训练 Transformer 模型**：用于预测未来销售数据
        2. **训练 LightGBM 模型**：用于根据预测推荐最优参数

        训练完成后，即可使用智能优化功能自动推荐参数。
        """)

        # 清除可能存在的旧推荐结果
        if 'recommendations' in st.session_state:
            del st.session_state['recommendations']

        # 提供快速导航按钮
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🤖 训练 Transformer", key="goto_transformer", use_container_width=True):
                st.session_state.current_page = 'transformer_training'
                st.session_state.training_model_expanded = True
                st.rerun()
        with col2:
            if st.button("📊 训练 LightGBM", key="goto_lightgbm", use_container_width=True):
                st.session_state.current_page = 'lightgbm_training'
                st.session_state.training_model_expanded = True
                st.rerun()

        st.markdown("---")
        st.markdown("### 💡 替代方案")
        st.markdown("在模型训练完成前，您可以使用侧边栏的 **参数设置** 手动调整参数。")
        return

    st.markdown("---")

    # 优化按钮
    st.markdown("### 🚀 开始智能优化")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🎯 开始智能参数优化", key="start_smart_opt",
                     use_container_width=True, type="primary"):
            run_smart_optimization()

    # 显示优化结果（只有在模型准备好的情况下才显示）
    if st.session_state.get('recommendations'):
        display_optimization_results(optimization_model)


def run_smart_optimization():
    """执行智能优化"""
    import torch
    from transformer_model import SalesForecasterEncoderOnly
    from sales_data_processor import SalesDataProcessor, SalesDataGenerator
    from lightgbm_model import ParameterRecommender, SalesFeatureExtractor

    st.markdown("### 📈 优化进度")
    progress_container = st.container()

    with progress_container:
        status_text = st.empty()
        progress_bar = st.progress(0)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 步骤1: 加载 Transformer 模型
        status_text.info("🔄 加载 Transformer 模型...")
        progress_bar.progress(0.1)

        transformer = SalesForecasterEncoderOnly(
            input_dim=5, d_model=128, num_heads=8, num_layers=4,
            d_ff=512, input_seq_len=30, output_seq_len=7, dropout=0.1
        )
        transformer.load_state_dict(torch.load('./data/best_transformer_model.pth', map_location=device))
        transformer = transformer.to(device)
        transformer.eval()

        # 步骤2: 加载数据处理器
        status_text.info("🔄 加载数据处理器...")
        progress_bar.progress(0.2)

        processor = SalesDataProcessor()
        processor.load_scaler('./data/scaler_params.npz')

        # 步骤3: 生成输入数据
        status_text.info("🔄 准备输入数据...")
        progress_bar.progress(0.3)

        generator = SalesDataGenerator()
        df = generator.generate_sales_data(num_days=30, start_date='2025-01-01')
        recent_data = df[generator.beverage_types].values

        # 步骤4: Transformer 预测
        status_text.info("🔮 执行销售预测...")
        progress_bar.progress(0.5)

        normalized_input = processor.transform(recent_data)
        x = torch.FloatTensor(normalized_input).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = transformer(x)
            pred = pred.squeeze(0).cpu().numpy()

        predictions = processor.inverse_transform(pred)

        # 步骤5: 加载 LightGBM 模型
        status_text.info("🔄 加载 LightGBM 模型...")
        progress_bar.progress(0.7)

        recommender = ParameterRecommender()
        recommender.load_model('./data')

        # 步骤6: 推荐参数
        status_text.info("🧠 生成参数推荐...")
        progress_bar.progress(0.9)

        extractor = SalesFeatureExtractor()
        features = extractor.extract_features(predictions).reshape(1, -1)
        pred_params = recommender.predict_dict(features)

        # 整理原始预测结果
        beverage_types = ['碳酸饮料', '果汁饮料', '茶饮料', '功能饮料', '矿泉水']
        material_types = ['白砂糖', '浓缩果汁', '茶叶提取物', '功能成分', '包装材料']
        regions = ['道里区', '南岗区', '道外区', '香坊区', '松北区']

        raw_result = {
            'profits': {bev: float(pred_params['profits'][0, i])
                       for i, bev in enumerate(beverage_types)},
            'material_limits': {mat: float(pred_params['material_limits'][0, i])
                               for i, mat in enumerate(material_types)},
            'transport_limits': {reg: float(pred_params['transport_limits'][0, i])
                                for i, reg in enumerate(regions)},
            'min_production_ratio': float(pred_params['min_production_ratio'][0]),
            'max_production_multiplier': float(pred_params['max_production_multiplier'][0])
        }

        # 裁剪参数到模型允许的范围内
        status_text.info("🔧 校验参数范围...")
        clipped_params = clip_params_to_limits(raw_result)

        result = {
            'predictions': predictions,
            'profits': clipped_params['profits'],
            'material_limits': clipped_params['material_limits'],
            'transport_limits': clipped_params['transport_limits'],
            'min_production_ratio': clipped_params['min_production_ratio'],
            'max_production_multiplier': clipped_params['max_production_multiplier']
        }

        st.session_state['recommendations'] = result

        progress_bar.progress(1.0)
        status_text.empty()
        progress_bar.empty()

        st.success("✅ 智能优化完成！")
        st.rerun()

    except Exception as e:
        status_text.empty()
        progress_bar.empty()
        st.error(f"❌ 优化失败: {str(e)}")


def display_optimization_results(optimization_model):
    """显示优化结果并提供同步选项"""
    result = st.session_state['recommendations']
    predictions = result['predictions']
    beverage_types = ['碳酸饮料', '果汁饮料', '茶饮料', '功能饮料', '矿泉水']

    st.markdown("---")
    st.markdown("## 📊 优化结果")

    # 销售预测
    st.markdown("### 📈 未来7天销售预测")

    col1, col2 = st.columns([2, 1])

    with col1:
        # 预测图表
        fig = go.Figure()
        colors = ['#2E8B57', '#4682B4', '#DAA520', '#CD853F', '#708090']

        for i, (bev, color) in enumerate(zip(beverage_types, colors)):
            fig.add_trace(go.Scatter(
                x=list(range(1, 8)),
                y=predictions[:, i],
                mode='lines+markers',
                name=bev,
                line=dict(color=color, width=2),
                marker=dict(size=8)
            ))

        fig.update_layout(
            xaxis_title="天数",
            yaxis_title="销量 (升)",
            height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**预测汇总**")
        for i, bev in enumerate(beverage_types):
            total = np.sum(predictions[:, i])
            st.metric(bev, f"{total:,.0f} 升", f"日均 {total/7:,.0f}")

    # 推荐参数
    st.markdown("### 🎯 推荐参数")

    tab1, tab2, tab3, tab4 = st.tabs(["💰 利润参数", "📦 原料限制", "🚛 运输限制", "⚙️ 生产约束"])

    with tab1:
        col1, col2 = st.columns(2)
        profits = result['profits']
        items = list(profits.items())
        for i, (bev, profit) in enumerate(items):
            with (col1 if i < 3 else col2):
                st.metric(bev, f"{profit} 元/升")

    with tab2:
        col1, col2 = st.columns(2)
        materials = result['material_limits']
        items = list(materials.items())
        for i, (mat, limit) in enumerate(items):
            with (col1 if i < 3 else col2):
                st.metric(mat, f"{limit:,.0f} 千克")

    with tab3:
        col1, col2 = st.columns(2)
        transport = result['transport_limits']
        items = list(transport.items())
        for i, (reg, limit) in enumerate(items):
            with (col1 if i < 3 else col2):
                st.metric(reg, f"{limit:,.0f} 升")

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("最小生产比例", result['min_production_ratio'])
        with col2:
            st.metric("最大生产倍数", result['max_production_multiplier'])

    # 同步选项
    st.markdown("---")
    st.markdown("### 🔄 参数同步")

    sync_option = st.radio(
        "是否将推荐参数同步到优化模型？",
        ["否，仅查看推荐结果", "是，同步到参数设置"],
        key="sync_radio",
        horizontal=True
    )

    if sync_option == "是，同步到参数设置":
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("✅ 确认同步参数", key="confirm_sync", use_container_width=True, type="primary"):
                # 同步参数到模型
                profits_list = list(result['profits'].values())
                material_limits_list = list(result['material_limits'].values())
                transport_limits_list = list(result['transport_limits'].values())

                params = {
                    'profits': profits_list,
                    'material_limits': material_limits_list,
                    'transport_limits': transport_limits_list,
                    'min_production_ratio': result['min_production_ratio'],
                    'max_production_multiplier': result['max_production_multiplier']
                }
                optimization_model.update_parameters(params)

                # 同步更新侧边栏的 session_state，使 UI 控件显示新值
                st.session_state.sidebar_profits = profits_list
                st.session_state.sidebar_material_limits = material_limits_list
                st.session_state.sidebar_transport_limits = transport_limits_list
                st.session_state.sidebar_min_ratio = result['min_production_ratio']
                st.session_state.sidebar_max_multiplier = result['max_production_multiplier']

                # 删除控件的 key，让下次渲染时从 sidebar_* 变量重新读取
                # （Streamlit 不允许直接修改已实例化控件的 key 值）
                keys_to_delete = []
                for i in range(5):
                    keys_to_delete.extend([f"profit_{i}", f"material_{i}", f"transport_{i}"])
                keys_to_delete.extend(["min_ratio", "max_multiplier"])

                for key in keys_to_delete:
                    if key in st.session_state:
                        del st.session_state[key]

                st.session_state['sync_params'] = True
                st.session_state['sync_success'] = True  # 标记同步成功，用于显示提示
                st.rerun()

    # 在页面重新渲染后显示同步成功提示
    if st.session_state.get('sync_success', False):
        st.success("✅ 参数已同步成功！侧边栏参数已更新，请返回主页点击「开始求解」查看优化结果。")
        st.balloons()
        st.session_state['sync_success'] = False  # 重置状态，只显示一次


# ==================== 主路由函数 ====================

def render_ml_page(optimization_model):
    """根据当前页面状态渲染对应页面"""
    init_session_state()

    current_page = st.session_state.get('current_page', 'main')

    if current_page == 'transformer_training':
        page_transformer_training()
        return True
    elif current_page == 'lightgbm_training':
        page_lightgbm_training()
        return True
    elif current_page == 'smart_optimization':
        page_smart_optimization(optimization_model)
        return True

    return False
