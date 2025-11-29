"""
饮料生产企业线性规划优化系统 - Streamlit应用
运筹学专家系统 - 交互式界面
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime
import os


def inject_coze_chatbot():
    """注入 Coze 聊天机器人到页面 - 使用 Coze 官方 WebSDK"""
    # 使用 Coze 官方 WebSDK 嵌���
    coze_chatbot_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            html, body {
                margin: 0;
                padding: 0;
                background: transparent !important;
                overflow: visible !important;
                width: 100%;
                height: 100%;
            }
        </style>
    </head>
    <body>
        <!-- Coze 官方 WebSDK -->
        <script src="https://lf-cdn.coze.cn/obj/unpkg/flow-platform/chat-app-sdk/1.2.0-beta.19/libs/cn/index.js"></script>
        <script>
            new CozeWebSDK.WebChatClient({
                config: {
                    bot_id: '7578098968145100834',
                },
                componentProps: {
                    title: '智能助手',
                },
                auth: {
                    type: 'token',
                    token: 'pat_1SoLFxXchERCiFAktfLsybEwHUUrz6OtZVWlJemZawCDCIC0vI6BkFruhrKKQEC1',
                    onRefreshToken: function () {
                        return 'pat_1SoLFxXchERCiFAktfLsybEwHUUrz6OtZVWlJemZawCDCIC0vI6BkFruhrKKQEC1'
                    }
                }
            });
        </script>
    </body>
    </html>
    """

    # CSS让iframe覆盖页面
    st.markdown("""
    <style>
        div[data-testid="stHtml"]:last-of-type {
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            width: 100vw !important;
            height: 100vh !important;
            z-index: 99998 !important;
            pointer-events: none !important;
            overflow: visible !important;
        }
        div[data-testid="stHtml"]:last-of-type > div {
            width: 100% !important;
            height: 100% !important;
            overflow: visible !important;
        }
        div[data-testid="stHtml"]:last-of-type iframe {
            width: 100vw !important;
            height: 100vh !important;
            border: none !important;
            background: transparent !important;
            pointer-events: auto !important;
            overflow: visible !important;
        }
    </style>
    """, unsafe_allow_html=True)

    components.html(coze_chatbot_html, height=800, width=800, scrolling=False)

SIMPLEX_TABLEAU_HTML = """
<div style="margin-top:0.5rem;">
<figure style="margin:1rem auto;max-width:720px;text-align:center;">
<svg width="720" height="240" viewBox="0 0 720 240" xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="st-simplex-iter0">
  <title id="st-simplex-iter0">单纯形法迭代 0 - 初始单纯形表</title>
  <style>
    .title { font: 600 18px 'Segoe UI','Microsoft YaHei',sans-serif; fill:#0f172a; }
    .subtitle { font: 500 14px 'Segoe UI','Microsoft YaHei',sans-serif; fill:#475569; }
    .header-cell { fill:#2563eb; stroke:#1d4ed8; }
    .header-text { font: 600 13px 'Segoe UI','Microsoft YaHei',sans-serif; fill:#ffffff; text-anchor:middle; dominant-baseline:middle; }
    .data-cell { fill:#ffffff; stroke:#cbd5f5; }
    .pivot-cell { fill:#fde68a; stroke:#f59e0b; }
    .cell-text { font: 500 13px 'Segoe UI','Microsoft YaHei',sans-serif; fill:#0f172a; text-anchor:middle; dominant-baseline:middle; }
    .note { font: 500 12px 'Segoe UI','Microsoft YaHei',sans-serif; fill:#475569; }
  </style>
  <rect x="0" y="0" width="720" height="240" rx="18" fill="#f8fafc" stroke="#e2e8f0"/>
  <text x="30" y="35" class="title">迭代 0 · 初始单纯形表</text>
  <text x="520" y="35" class="subtitle">入基: x₁ | 出基: s₁</text>
  <rect x="20" y="70" width="130" height="34" class="header-cell"/>
  <text x="85" y="87" class="header-text">基变量</text>
  <rect x="150" y="70" width="90" height="34" class="header-cell"/>
  <text x="195" y="87" class="header-text">x₁</text>
  <rect x="240" y="70" width="90" height="34" class="header-cell"/>
  <text x="285" y="87" class="header-text">x₂</text>
  <rect x="330" y="70" width="90" height="34" class="header-cell"/>
  <text x="375" y="87" class="header-text">x₃</text>
  <rect x="420" y="70" width="90" height="34" class="header-cell"/>
  <text x="465" y="87" class="header-text">s₁</text>
  <rect x="510" y="70" width="90" height="34" class="header-cell"/>
  <text x="555" y="87" class="header-text">s₂</text>
  <rect x="600" y="70" width="100" height="34" class="header-cell"/>
  <text x="650" y="87" class="header-text">RHS</text>
  <rect x="20" y="110" width="130" height="32" class="data-cell"/>
  <text x="85" y="126" class="cell-text">s₁</text>
  <rect x="150" y="110" width="90" height="32" class="pivot-cell"/>
  <text x="195" y="126" class="cell-text">2</text>
  <rect x="240" y="110" width="90" height="32" class="data-cell"/>
  <text x="285" y="126" class="cell-text">1</text>
  <rect x="330" y="110" width="90" height="32" class="data-cell"/>
  <text x="375" y="126" class="cell-text">0</text>
  <rect x="420" y="110" width="90" height="32" class="data-cell"/>
  <text x="465" y="126" class="cell-text">1</text>
  <rect x="510" y="110" width="90" height="32" class="data-cell"/>
  <text x="555" y="126" class="cell-text">0</text>
  <rect x="600" y="110" width="100" height="32" class="data-cell"/>
  <text x="650" y="126" class="cell-text">240</text>
  <rect x="20" y="146" width="130" height="32" class="data-cell"/>
  <text x="85" y="162" class="cell-text">s₂</text>
  <rect x="150" y="146" width="90" height="32" class='data-cell'/>
  <text x="195" y="162" class='cell-text'>1</text>
  <rect x="240" y="146" width="90" height="32" class="data-cell"/>
  <text x="285" y="162" class="cell-text">3</text>
  <rect x="330" y="146" width="90" height="32" class="data-cell"/>
  <text x="375" y="162" class="cell-text">1</text>
  <rect x="420" y="146" width="90" height="32" class="data-cell"/>
  <text x="465" y="162" class="cell-text">0</text>
  <rect x="510" y="146" width="90" height="32" class="data-cell"/>
  <text x="555" y="162" class="cell-text">1</text>
  <rect x="600" y="146" width="100" height="32" class="data-cell"/>
  <text x="650" y="162" class="cell-text">360</text>
  <rect x="20" y="182" width="130" height="32" class="data-cell"/>
  <text x="85" y="198" class="cell-text">Z</text>
  <rect x="150" y="182" width="90" height="32" class="data-cell"/>
  <text x="195" y="198" class="cell-text">-5</text>
  <rect x="240" y="182" width="90" height="32" class="data-cell"/>
  <text x="285" y="198" class="cell-text">-4</text>
  <rect x="330" y="182" width="90" height="32" class="data-cell"/>
  <text x="375" y="198" class="cell-text">-3</text>
  <rect x="420" y="182" width="90" height="32" class="data-cell"/>
  <text x="465" y="198" class="cell-text">0</text>
  <rect x="510" y="182" width="90" height="32" class="data-cell"/>
  <text x="555" y="198" class="cell-text">0</text>
  <rect x="600" y="182" width="100" height="32" class="data-cell"/>
  <text x="650" y="198" class="cell-text">0</text>
  <text x="30" y="222" class="note">最小比值检验：s₁ 行 240 ÷ 2 = 120，s₂ 行 360 ÷ 1 = 360 → 选 s₁ 离基</text>
</svg>
<figcaption style="font-size:0.9rem;color:#475569;">初始基为 s₁、s₂，x₁ 列的 reduced cost 最负，通过最小比值选择 s₁ 离基。</figcaption>
</figure>

<figure style="margin:1.5rem auto;max-width:720px;text-align:center;">
<svg width="720" height="240" viewBox="0 0 720 240" xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="st-simplex-iter1">
  <title id="st-simplex-iter1">单纯形法迭代 1 - 枢轴完成后</title>
  <style>
    .title { font: 600 18px 'Segoe UI','Microsoft YaHei',sans-serif; fill:#0f172a; }
    .subtitle { font: 500 14px 'Segoe UI','Microsoft YaHei',sans-serif; fill:#475569; }
    .header-cell { fill:#2563eb; stroke:#1d4ed8; }
    .header-text { font: 600 13px 'Segoe UI','Microsoft YaHei',sans-serif; fill:#ffffff; text-anchor:middle; dominant-baseline:middle; }
    .data-cell { fill:#ffffff; stroke:#cbd5f5; }
    .pivot-cell { fill:#fde68a; stroke:#f59e0b; }
    .cell-text { font: 500 13px 'Segoe UI','Microsoft YaHei',sans-serif; fill:#0f172a; text-anchor:middle; dominant-baseline:middle; }
    .note { font: 500 12px 'Segoe UI','Microsoft YaHei',sans-serif; fill:#475569; }
  </style>
  <rect x="0" y="0" width="720" height="240" rx="18" fill="#f8fafc" stroke="#e2e8f0"/>
  <text x="30" y="35" class="title">迭代 1 · 枢轴完成后</text>
  <text x="520" y="35" class="subtitle">入基: x₂ | 出基: s₂</text>
  <rect x="20" y="70" width="130" height="34" class="header-cell"/>
  <text x="85" y="87" class="header-text">基变量</text>
  <rect x="150" y="70" width="90" height="34" class="header-cell"/>
  <text x="195" y="87" class="header-text">x₁</text>
  <rect x="240" y="70" width="90" height="34" class="header-cell"/>
  <text x="285" y="87" class="header-text">x₂</text>
  <rect x="330" y="70" width="90" height="34" class="header-cell"/>
  <text x="375" y="87" class="header-text">x₃</text>
  <rect x="420" y="70" width="90" height="34" class="header-cell"/>
  <text x="465" y="87" class="header-text">s₁</text>
  <rect x="510" y="70" width="90" height="34" class="header-cell"/>
  <text x="555" y="87" class="header-text">s₂</text>
  <rect x="600" y="70" width="100" height="34" class="header-cell"/>
  <text x="650" y="87" class="header-text">RHS</text>
  <rect x="20" y="110" width="130" height="32" class="data-cell"/>
  <text x="85" y="126" class="cell-text">x₁</text>
  <rect x="150" y="110" width="90" height="32" class="data-cell"/>
  <text x="195" y="126" class="cell-text">1</text>
  <rect x="240" y="110" width="90" height="32" class="data-cell"/>
  <text x="285" y="126" class="cell-text">0.5</text>
  <rect x="330" y="110" width="90" height="32" class="data-cell"/>
  <text x="375" y="126" class="cell-text">0</text>
  <rect x="420" y="110" width="90" height="32" class="data-cell"/>
  <text x="465" y="126" class="cell-text">0.5</text>
  <rect x="510" y="110" width="90" height="32" class="data-cell"/>
  <text x="555" y="126" class="cell-text">0</text>
  <rect x="600" y="110" width="100" height="32" class="data-cell"/>
  <text x="650" y="126" class="cell-text">120</text>
  <rect x="20" y="146" width="130" height="32" class="data-cell"/>
  <text x="85" y="162" class="cell-text">s₂</text>
  <rect x="150" y="146" width="90" height="32" class="data-cell"/>
  <text x="195" y="162" class="cell-text">0</text>
  <rect x="240" y="146" width="90" height="32" class="pivot-cell"/>
  <text x="285" y="162" class="cell-text">2.5</text>
  <rect x="330" y="146" width="90" height="32" class="data-cell"/>
  <text x="375" y="162" class="cell-text">1</text>
  <rect x="420" y="146" width="90" height="32" class="data-cell"/>
  <text x="465" y="162" class="cell-text">-0.5</text>
  <rect x="510" y="146" width="90" height="32" class="data-cell"/>
  <text x="555" y="162" class="cell-text">1</text>
  <rect x="600" y="146" width="100" height="32" class="data-cell"/>
  <text x="650" y="162" class="cell-text">240</text>
  <rect x="20" y="182" width="130" height="32" class="data-cell"/>
  <text x="85" y="198" class="cell-text">Z</text>
  <rect x="150" y="182" width="90" height="32" class="data-cell"/>
  <text x="195" y="198" class="cell-text">0</text>
  <rect x="240" y="182" width="90" height="32" class="data-cell"/>
  <text x="285" y="198" class="cell-text">-1.5</text>
  <rect x="330" y="182" width="90" height="32" class="data-cell"/>
  <text x="375" y="198" class="cell-text">-3</text>
  <rect x="420" y="182" width="90" height="32" class="data-cell"/>
  <text x="465" y="198" class="cell-text">2.5</text>
  <rect x="510" y="182" width="90" height="32" class="data-cell"/>
  <text x="555" y="198" class="cell-text">0</text>
  <rect x="600" y="182" width="100" height="32" class="data-cell"/>
  <text x="650" y="198" class="cell-text">600</text>
  <text x="30" y="222" class="note">下一步 pivot 在 x₂ 列：s₂ 行 240 ÷ 2.5 = 96 &lt; 120 ÷ 0.5 → 选择 s₂ 离基</text>
</svg>
<figcaption style="font-size:0.9rem;color:#475569;">完成第一个枢轴后，第二轮由 x₂ 入基，图中高亮提示下一次换基。</figcaption>
</figure>
</div>
"""

# 导入模型类
from beverage_optimization_model import BeverageOptimizationModel, model

# 导入机器学习功能模块
try:
    from streamlit_ml_features import (
        sidebar_navigation,
        render_ml_page,
        init_session_state,
        check_model_status
    )
    ML_FEATURES_AVAILABLE = True
except ImportError:
    ML_FEATURES_AVAILABLE = False

def setup_page():
    """设置页面配置"""
    st.set_page_config(
        page_title="饮料生产企业线性规划优化系统",
        page_icon="🥤",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 添加自定义CSS样式
    st.markdown("""
    <style>
    /* 页面标题样式 */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* 分区标题样式 */
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #4682B4;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #4682B4;
        padding-bottom: 0.5rem;
    }

    /* 参数卡片背景和字体颜色 */
    .parameter-card {
        background-color: #f8f9fa;
        color: #333333;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin-bottom: 1rem;
    }

    /* 结果卡片背景和字体颜色 */
    .result-card {
        background-color: #e8f5e8;
        color: #333333;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin-bottom: 1rem;
    }

    /* 紧约束和非紧约束背景颜色 */
    .constraint-binding {
        background-color: #fff3cd;
        border-color: #ffc107;
    }
    .constraint-non-binding {
        background-color: #d4edda;
        border-color: #28a745;
    }

    /* 按钮样式 */
    .stButton > button {
        background-color: #2E8B57;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        border: none;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #228B22;
        transform: translateY(-2px);
    }
    </style>
    """, unsafe_allow_html=True)

def display_header():
    """显示页面标题"""
    st.markdown('<div class="main-header">🥤 饮料生产企业线性规划优化系统</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">
        运筹学专家系统 - 解决原料和运输双重约束下的利润最大化问题
        </div>
        """, unsafe_allow_html=True)

def sidebar_parameters():
    """侧边栏参数设置"""
    st.sidebar.markdown("## 📊 模型参数设置")
    
    # 创建参数分组
    with st.sidebar.expander("💰 利润参数", expanded=True):
        profits = []
        for i, beverage in enumerate(model.beverage_types):
            profit = st.number_input(
                f"{beverage} 利润 (元/升)",
                value=float(model.profits[i]),
                min_value=0.1,
                max_value=50.0,
                step=0.1,
                key=f"profit_{i}"
            )
            profits.append(profit)
    
    with st.sidebar.expander("📦 原料供应限制", expanded=True):
        material_limits = []
        for i, material in enumerate(model.material_types):
            limit = st.number_input(
                f"{material} 供应量 (千克)",
                value=float(model.material_limits[i]),
                min_value=100.0,
                max_value=50000.0,
                step=100.0,
                key=f"material_{i}"
            )
            material_limits.append(limit)
    
    with st.sidebar.expander("🚛 运输能力限制", expanded=True):
        transport_limits = []
        for i, region in enumerate(model.transport_regions):
            limit = st.number_input(
                f"{region} 运输能力 (升)",
                value=float(model.transport_limits[i]),
                min_value=100.0,
                max_value=10000.0,
                step=50.0,
                key=f"transport_{i}"
            )
            transport_limits.append(limit)
    
    with st.sidebar.expander("⚙️ 生产约束参数", expanded=True):
        min_ratio = st.slider(
            "最小生产比例 (相对于上期销售)",
            min_value=0.3,
            max_value=1.0,
            value=0.8,
            step=0.05,
            key="min_ratio"
        )
        
        max_multiplier = st.slider(
            "最大生产倍数 (相对于上期销售)",
            min_value=1.0,
            max_value=3.0,
            value=1.5,
            step=0.1,
            key="max_multiplier"
        )
    
    # 更新模型参数
    if st.sidebar.button("🔄 更新参数", key="update_params"):
        params = {
            'profits': profits,
            'material_limits': material_limits,
            'transport_limits': transport_limits,
            'min_production_ratio': min_ratio,
            'max_production_multiplier': max_multiplier
        }
        model.update_parameters(params)
        st.session_state['parameters_updated'] = True
        st.rerun()

def display_model_overview():
    """显示模型概览"""
    st.markdown('<div class="section-header">📋 模型概览</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="parameter-card">
        <h4>🎯 决策变量</h4>
        <ul>
        <li>碳酸饮料生产量 (升)</li>
        <li>果汁饮料生产量 (升)</li>
        <li>茶饮料生产量 (升)</li>
        <li>功能饮料生产量 (升)</li>
        <li>矿泉水生产量 (升)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # 构建目标函数卡片的完整HTML，避免多次调用 st.markdown 导致元素脱离容器
        profit_items = "".join([
            f"<li>{bev}: {model.profits[i]:.1f}元/升</li>" for i, bev in enumerate(model.beverage_types)
        ])
        target_html = f"""
        <div class="parameter-card">
            <h4>📊 目标函数</h4>
            <p><strong>最大化总利润</strong></p>
            <p>总利润 = Σ(各饮料单位利润 × 生产量)</p>
            <p>当前单位利润设置：</p>
            <ul>
                {profit_items}
            </ul>
        </div>
        """
        st.markdown(target_html, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="parameter-card">
        <h4>🔗 约束条件</h4>
        <ul>
        <li><strong>原料供应限制</strong>：5种原料供应量约束</li>
        <li><strong>运输能力限制</strong>：5个区域运输能力约束</li>
        <li><strong>生产量约束</strong>：最小和最大生产量限制</li>
        <li><strong>非负约束</strong>：所有生产量 ≥ 0</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

def solve_and_display():
    """求解模型并显示结果"""
    st.markdown('<div class="section-header">🧮 模型求解</div>', unsafe_allow_html=True)
    
    # 求解按钮
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔍 开始求解", key="solve_model", use_container_width=True):
            with st.spinner("正在使用单纯形法求解线性规划模型..."):
                solution = model.solve_model()
                st.session_state['solution'] = solution
                
                if solution['success']:
                    # 进行灵敏度分析
                    sensitivity = model.sensitivity_analysis(solution)
                    st.session_state['sensitivity'] = sensitivity
            
            st.success("✅ 模型求解完成！")
    
    # 显示求解结果
    if 'solution' in st.session_state:
        solution = st.session_state['solution']
        display_solution_results(solution)
    
    # 显示灵敏度分析
    if 'sensitivity' in st.session_state:
        sensitivity = st.session_state['sensitivity']
        display_sensitivity_analysis(sensitivity)

def display_solution_results(solution):
    """显示求解结果"""
    st.markdown('<div class="section-header">📈 求解结果</div>', unsafe_allow_html=True)
    
    if not solution['success']:
        st.error(f"❌ 求解失败: {solution.get('message', '未知错误')}")
        return
    
    # 1. 最优解概览
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="result-card">
        <h4>💰 最大利润</h4>
        <p style="font-size: 2rem; color: #28a745; font-weight: bold;">
        {solution['optimal_value']:,.2f} 元
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_production = np.sum(solution['decision_variables'])
        st.markdown(f"""
        <div class="result-card">
        <h4>📦 总产量</h4>
        <p style="font-size: 2rem; color: #007bff; font-weight: bold;">
        {total_production:,.0f} 升
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="result-card">
        <h4>⚡ 求解效率</h4>
        <p style="font-size: 1.2rem;">
        迭代次数: {solution['iterations']}<br>
        求解状态: ✅ 成功
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    # 2. 最优生产方案
    st.markdown("### 🎯 最优生产方案")
    
    production_data = []
    for i, beverage in enumerate(model.beverage_types):
        production_data.append({
            '饮料类型': beverage,
            '最优生产量(升)': f"{solution['decision_variables'][i]:.0f}",
            '占总产量比例': f"{solution['decision_variables'][i]/total_production*100:.1f}%",
            '单位利润(元/升)': f"{model.profits[i]:.1f}",
            '贡献利润(元)': f"{solution['decision_variables'][i] * model.profits[i]:.0f}"
        })
    
    production_df = pd.DataFrame(production_data)
    st.dataframe(production_df, use_container_width=True)
    
    # 3. 生产方案可视化
    col1, col2 = st.columns(2)
    
    with col1:
        # 生产量柱状图
        fig_production = go.Figure(data=[
            go.Bar(
                x=model.beverage_types,
                y=solution['decision_variables'],
                marker_color=['#2E8B57', '#4682B4', '#DAA520', '#CD853F', '#708090'],
                text=[f"{val:.0f}" for val in solution['decision_variables']],
                textposition='auto',
            )
        ])
        
        fig_production.update_layout(
            title="各饮料最优生产量",
            xaxis_title="饮料类型",
            yaxis_title="生产量 (升)",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig_production, use_container_width=True)
    
    with col2:
        # 利润贡献饼图
        profit_contributions = solution['decision_variables'] * model.profits
        
        fig_pie = go.Figure(data=[
            go.Pie(
                labels=model.beverage_types,
                values=profit_contributions,
                hole=0.4,
                marker_colors=['#2E8B57', '#4682B4', '#DAA520', '#CD853F', '#708090']
            )
        ])
        
        fig_pie.update_layout(
            title="各饮料利润贡献分布",
            height=400
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # 4. 约束条件分析
    display_constraint_analysis(solution['constraint_analysis'])

    # 5. 单纯形迭代详细过程
    simplex_payload = solution.get('simplex_iterations')
    if simplex_payload and simplex_payload.get('iterations'):
        with st.expander("🔁 展开查看单纯形法迭代表", expanded=False):
            display_simplex_iteration_history(simplex_payload)

    # 5. 单纯形迭代详细过程（折叠展示）

def display_constraint_analysis(constraint_analysis):
    """显示约束条件分析"""
    st.markdown("### 🔗 约束条件分析")
    
    # 原料约束
    st.markdown("#### 📦 原料约束分析")
    
    material_data = []
    for material in model.material_types:
        if material in constraint_analysis['material_constraints']:
            info = constraint_analysis['material_constraints'][material]
            material_data.append({
                '原料类型': material,
                '使用量(千克)': f"{info['usage']:.0f}",
                '供应限制(千克)': f"{info['limit']:.0f}",
                '利用率': f"{info['utilization_rate']*100:.1f}%",
                '松弛量(千克)': f"{info['slack']:.1f}",
                '影子价格': f"{info['shadow_price']:.3f}",
                '状态': '紧约束' if info['is_binding'] else '非紧约束'
            })
    
    material_df = pd.DataFrame(material_data)
    st.dataframe(material_df, use_container_width=True)
    
    # 运输约束
    st.markdown("#### 🚛 运输约束分析")
    
    transport_data = []
    for region in model.transport_regions:
        if region in constraint_analysis['transport_constraints']:
            info = constraint_analysis['transport_constraints'][region]
            transport_data.append({
                '运输区域': region,
                '运输量(升)': f"{info['usage']:.0f}",
                '运输限制(升)': f"{info['limit']:.0f}",
                '利用率': f"{info['utilization_rate']*100:.1f}%",
                '松弛量(升)': f"{info['slack']:.1f}",
                '影子价格': f"{info['shadow_price']:.3f}",
                '状态': '紧约束' if info['is_binding'] else '非紧约束'
            })
    
    transport_df = pd.DataFrame(transport_data)
    st.dataframe(transport_df, use_container_width=True)
    
    # 约束状态可视化
    col1, col2 = st.columns(2)
    
    with col1:
        # 原料利用率图
        material_utilizations = []
        material_names = []
        for material in model.material_types:
            if material in constraint_analysis['material_constraints']:
                info = constraint_analysis['material_constraints'][material]
                material_names.append(material)
                material_utilizations.append(info['utilization_rate'] * 100)
        
        fig_material = go.Figure(data=[
            go.Bar(
                x=material_names,
                y=material_utilizations,
                marker_color=['#dc3545' if u > 95 else '#ffc107' if u > 80 else '#28a745' for u in material_utilizations],
                text=[f"{u:.1f}%" for u in material_utilizations],
                textposition='auto',
            )
        ])
        
        fig_material.update_layout(
            title="原料利用率分析",
            xaxis_title="原料类型",
            yaxis_title="利用率 (%)",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig_material, use_container_width=True)
    
    with col2:
        # 运输利用率图
        transport_utilizations = []
        transport_names = []
        for region in model.transport_regions:
            if region in constraint_analysis['transport_constraints']:
                info = constraint_analysis['transport_constraints'][region]
                transport_names.append(region)
                transport_utilizations.append(info['utilization_rate'] * 100)
        
        fig_transport = go.Figure(data=[
            go.Bar(
                x=transport_names,
                y=transport_utilizations,
                marker_color=['#dc3545' if u > 95 else '#ffc107' if u > 80 else '#28a745' for u in transport_utilizations],
                text=[f"{u:.1f}%" for u in transport_utilizations],
                textposition='auto',
            )
        ])
        
        fig_transport.update_layout(
            title="运输能力利用率分析",
            xaxis_title="运输区域",
            yaxis_title="利用率 (%)",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig_transport, use_container_width=True)

def display_sensitivity_analysis(sensitivity):
    """显示灵敏度分析"""
    st.markdown('<div class="section-header">📊 灵敏度分析</div>', unsafe_allow_html=True)
    
    if 'error' in sensitivity:
        st.error(f"❌ 灵敏度分析失败: {sensitivity['error']}")
        return
    
    # 1. 目标函数系数分析
    st.markdown("### 💰 利润系数灵敏度分析")
    
    profit_data = []
    for beverage in model.beverage_types:
        if beverage in sensitivity['objective_coefficients']:
            info = sensitivity['objective_coefficients'][beverage]
            profit_data.append({
                '饮料类型': beverage,
                '当前利润(元/升)': info['current_profit'],
                '最优生产量(升)': f"{info['optimal_production']:.0f}",
                '减少成本': f"{info['reduced_cost']:.3f}",
                '建议': '保持当前利润' if info['reduced_cost'] < 1e-6 else f'建议提高利润至{info["current_profit"] + info["reduced_cost"]:.2f}元/升'
            })
    
    profit_df = pd.DataFrame(profit_data)
    st.dataframe(profit_df, use_container_width=True)
    
    # 2. 约束条件RHS灵敏度分析
    st.markdown("### 🔗 约束条件灵敏度分析")
    
    if sensitivity['rhs_changes']:
        rhs_data = []
        for constraint, info in sensitivity['rhs_changes'].items():
            rhs_data.append({
                '约束类型': constraint,
                '当前限制': info['current_limit'],
                '影子价格': f"{info['shadow_price']:.3f}",
                '改进建议': info['recommendation']
            })
        
        rhs_df = pd.DataFrame(rhs_data)
        st.dataframe(rhs_df, use_container_width=True)
    else:
        st.info("ℹ️ 当前没有紧约束条件，灵敏度分析显示模型具有较好的稳健性")
    
    # 3. 管理建议
    st.markdown("### 💡 管理建议")
    
    recommendations = sensitivity.get('recommendations', [])
    
    # 添加基于分析的建议
    if 'constraint_analysis' in st.session_state.get('solution', {}):
        constraint_analysis = st.session_state['solution']['constraint_analysis']
        
        # 分析紧约束
        binding_constraints = constraint_analysis.get('binding_constraints', [])
        if binding_constraints:
            recommendations.append(f"发现 {len(binding_constraints)} 个紧约束条件，建议优先扩展这些资源")
            for constraint in binding_constraints:
                recommendations.append(f"- {constraint}")
        
        # 分析非紧约束
        non_binding_constraints = constraint_analysis.get('non_binding_constraints', [])
        if non_binding_constraints:
            recommendations.append(f"有 {len(non_binding_constraints)} 个约束条件存在松弛，资源配置相对充足")
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"**{i}.** {rec}")
    else:
        st.success("✅ 当前生产方案已达到最优，建议保持现有策略")

    # 4. 过程记录
    display_sensitivity_step_logs(sensitivity)


def display_model_explanation():
    """显示模型解释"""
    st.markdown('<div class="section-header">📚 模型解释与算法说明</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 数学模型构建
        
        **决策变量：**
        - 设 x₁, x₂, x₃, x₄, x₅ 分别代表5种饮料的生产量
        
        **目标函数：**
        - max Z = c₁x₁ + c₂x₂ + c₃x₃ + c₄x₄ + c₅x₅
        - 其中 cᵢ 为第i种饮料的单位利润
        
        **约束条件：**
        1. **原料约束：** Aₘₐₜₑᵣᵢₐₗ × X ≤ bₘₐₜₑᵣᵢₐₗ
        2. **运输约束：** Tʳᵃⁿˢᵖᵒʳᵗ × X ≤ bʳᵃⁿˢᵖᵒʳᵗ
        3. **生产约束：** Xₘᵢₙ ≤ X ≤ Xₘₐₓ
        4. **非负约束：** X ≥ 0
        """)
    
    with col2:
        st.markdown("""
        ### ⚡ 求解算法说明
        
        **单纯形法原理：**
        1. 将线性规划问题转换为标准形式
        2. 构建初始单纯形表
        3. 通过迭代寻找最优解
        4. 检验最优性条件
        
        **影子价格意义：**
        - 表示约束条件右侧每增加1单位时目标函数的改善程度
        - 反映资源的稀缺程度和价值
        
        **灵敏度分析：**
        - 分析参数变化对最优解的影响
        - 确定参数的稳定区间
        - 提供管理决策依据
        """)
    
    st.markdown("""
    ### 🔄 求解步骤详解
    
    1. **问题识别与建模**
       - 确定决策变量和目标函数
       - 识别所有约束条件
       - 构建数学模型
    
    2. **数据准备与验证**
       - 收集历史数据和预测信息
       - 验证参数的合理性
       - 设置约束条件边界
    
    3. **模型求解**
       - 使用单纯形法求解
       - 获得最优解和影子价格
       - 验证解的可行性
    
    4. **结果分析与解释**
       - 分析最优生产方案
       - 计算各约束的利用率
       - 进行灵敏度分析
    
    5. **决策支持**
       - 提供管理建议
       - 识别关键约束因素
       - 制定改进策略
    """)

    with st.expander("🔍 查看单纯形法单纯形表迭代", expanded=False):
        st.markdown(SIMPLEX_TABLEAU_HTML, unsafe_allow_html=True)




def display_simplex_iteration_history(iteration_payload):
    """以交互方式展示单纯形表迭代步骤（父级负责折叠容器）。"""
    if not iteration_payload or 'iterations' not in iteration_payload:
        return

    iterations = iteration_payload.get('iterations') or []
    if not iterations:
        return

    indices = list(range(len(iterations)))

    def format_label(idx: int) -> str:
        item = iterations[idx]
        return f"{item.get('phase')} · 第 {item.get('iteration')} 步 ({item.get('status')})"

    selected_idx = st.selectbox("选择要查看的迭代步骤", indices, format_func=format_label, index=len(indices) - 1)
    entry = iterations[selected_idx]

    column_labels = entry.get('column_labels', [])
    row_labels = entry.get('row_labels', [])
    tableau_before = entry.get('tableau_before', [])
    tableau_after = entry.get('tableau_after', tableau_before)

    st.markdown(f"**阶段**：{entry.get('phase')}　|　**状态**：{entry.get('status')}")
    st.markdown(f"**入基变量**：{entry.get('entering') or '-'}　|　**出基变量**：{entry.get('leaving') or '-'}")
    st.markdown(f"**当前目标值**：{entry.get('objective_value', 0):.2f}")
    if entry.get('reason'):
        st.info(entry['reason'])

    if tableau_before:
        before_df = pd.DataFrame(tableau_before, columns=column_labels + ['RHS'], index=row_labels)
        after_df = pd.DataFrame(tableau_after, columns=column_labels + ['RHS'], index=row_labels)
        tab1, tab2 = st.tabs(["迭代前", "迭代后"])
        with tab1:
            st.dataframe(before_df, use_container_width=True)
        with tab2:
            st.dataframe(after_df, use_container_width=True)

    cj_values = entry.get('cj_minus_zj')
    if cj_values:
        cj_df = pd.DataFrame([cj_values], columns=column_labels)
        st.caption("Cj - Zj 行")
        st.dataframe(cj_df, use_container_width=True)

    ratios = entry.get('ratios') or []
    if ratios:
        ratio_df = pd.DataFrame(ratios)
        st.caption("最小比值检验记录")
        st.dataframe(ratio_df, use_container_width=True)


def display_sensitivity_step_logs(sensitivity):
    """展示灵敏度分析扫描的每一步细节。"""
    steps = sensitivity.get('step_logs')
    if not steps:
        return

    st.markdown("### 🧪 灵敏度分析迭代记录")
    with st.expander("展开查看灵敏度扫描过程", expanded=False):
        category_options = ['全部', '目标系数', '约束RHS']
        selected_category = st.selectbox("筛选类别", category_options, index=0)

        def match_category(item):
            if selected_category == '全部':
                return True
            if selected_category == '目标系数':
                return item.get('category') == 'objective'
            return item.get('category') == 'rhs'

        filtered_steps = [step for step in steps if match_category(step)]

        if not filtered_steps:
            st.info("暂无匹配的灵敏度迭代记录。")
            return

        display_rows = []
        for step in filtered_steps:
            display_rows.append({
                '步骤': step.get('step'),
                '类别': '目标系数' if step.get('category') == 'objective' else '约束RHS',
                '对象': step.get('target'),
                '方向': '增加' if step.get('direction') == 'increase' else '减少',
                '测试值': step.get('tested_value'),
                '状态': step.get('status'),
                '可行': '是' if step.get('feasible') else '否',
                '目标值': step.get('objective_value'),
                '方案快照': ", ".join(f"{val:.1f}" for val in step.get('solution_snapshot', [])) if step.get('solution_snapshot') else '',
                '备注': step.get('note') or ''
            })

        log_df = pd.DataFrame(display_rows)
        st.dataframe(log_df, use_container_width=True)


def main():
    """主函数"""
    setup_page()

    # 初始化机器学习功能的 session state
    if ML_FEATURES_AVAILABLE:
        init_session_state()

    # 侧边栏参数设置
    sidebar_parameters()

    # 侧边栏机器学习功能导航
    if ML_FEATURES_AVAILABLE:
        sidebar_navigation()

    # 检查是否在机器学习页面
    if ML_FEATURES_AVAILABLE:
        is_ml_page = render_ml_page(model)
        if is_ml_page:
            # 如果在ML页面，只显示页脚后返回
            st.markdown("---")
            st.markdown("""
            <div style="text-align: center; color: #666; padding: 1rem;">
            <p>© 2025 饮料生产企业线性规划优化系统 | 运筹学专家系统</p>
            <p>基于单纯形法和灵敏度分析的企业决策支持工具</p>
            </div>
            """, unsafe_allow_html=True)
            # 注入 Dify 聊天机器人
            inject_coze_chatbot()
            return

    # 主页面内容
    display_header()

    # 主要内容区域
    display_model_overview()
    solve_and_display()

    # 模型解释
    with st.expander("📖 查看模型详细解释", expanded=False):
        display_model_explanation()

    # 页脚信息
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
    <p>© 2025 饮料生产企业线性规划优化系统 | 运筹学专家系统</p>
    <p>基于单纯形法和灵敏度分析的企业决策支持工具</p>
    </div>
    """, unsafe_allow_html=True)

    # 注入 Dify 聊天机器人（放在页面最后）
    inject_coze_chatbot()

if __name__ == "__main__":
    main()
