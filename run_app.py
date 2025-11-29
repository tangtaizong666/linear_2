#!/usr/bin/env python3
"""
饮料生产企业线性规划优化系统启动脚本
"""

import subprocess
import sys
import os

def main():
    """启动Streamlit应用"""
    print("🥤 启动饮料生产企业线性规划优化系统...")
    print("🔄 正在初始化应用...")
    
    # 确保当前目录在Python路径中
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    try:
        # 启动Streamlit应用
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_app.py",
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--browser.gatherUsageStats=False"
        ])
    except KeyboardInterrupt:
        print("\n👋 应用已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()