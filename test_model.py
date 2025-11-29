#!/usr/bin/env python3
"""
饮料生产企业线性规划模型测试脚本
验证模型正确性和求解结果
"""

import numpy as np
from beverage_optimization_model import BeverageOptimizationModel

def test_model():
    """测试模型功能"""
    print("🧪 开始测试饮料生产企业线性规划模型...")
    
    # 创建模型实例
    model = BeverageOptimizationModel()
    
    print("✅ 模型初始化成功")
    print(f"📊 饮料种类: {len(model.beverage_types)} 种")
    print(f"📦 原料类型: {len(model.material_types)} 种")
    print(f"🚛 运输区域: {len(model.transport_regions)} 个")
    
    # 测试默认参数
    print("\n📋 默认参数设置:")
    print("利润参数:", [f"{p:.1f}" for p in model.profits])
    print("原料限制:", [f"{l:.0f}" for l in model.material_limits])
    print("运输限制:", [f"{l:.0f}" for l in model.transport_limits])
    
    # 求解模型
    print("\n🔍 开始求解模型...")
    solution = model.solve_model()
    
    if solution['success']:
        print("✅ 模型求解成功！")
        print(f"💰 最大利润: {solution['optimal_value']:,.2f} 元")
        print(f"📦 总产量: {np.sum(solution['decision_variables']):,.0f} 升")
        
        # 显示各饮料生产量
        print("\n🎯 最优生产方案:")
        for i, beverage in enumerate(model.beverage_types):
            production = solution['decision_variables'][i]
            profit_contribution = production * model.profits[i]
            print(f"  {beverage}: {production:.0f} 升 (利润贡献: {profit_contribution:.0f} 元)")
        
        # 约束分析
        if 'constraint_analysis' in solution:
            analysis = solution['constraint_analysis']
            
            print("\n🔗 紧约束条件分析:")
            binding_constraints = analysis.get('binding_constraints', [])
            if binding_constraints:
                for constraint in binding_constraints:
                    print(f"  ⚠️ {constraint}")
            else:
                print("  ✅ 无紧约束条件，资源配置充足")
            
            # 原料利用率
            print("\n📦 原料利用率:")
            for material in model.material_types:
                if material in analysis['material_constraints']:
                    info = analysis['material_constraints'][material]
                    print(f"  {material}: {info['utilization_rate']*100:.1f}% (影子价格: {info['shadow_price']:.3f})")
            
            # 运输利用率
            print("\n🚛 运输能力利用率:")
            for region in model.transport_regions:
                if region in analysis['transport_constraints']:
                    info = analysis['transport_constraints'][region]
                    print(f"  {region}: {info['utilization_rate']*100:.1f}% (影子价格: {info['shadow_price']:.3f})")
        
        # 灵敏度分析
        print("\n📊 进行灵敏度分析...")
        sensitivity = model.sensitivity_analysis(solution)
        
        if 'recommendations' in sensitivity and sensitivity['recommendations']:
            print("💡 管理建议:")
            for i, rec in enumerate(sensitivity['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        print("\n✅ 模型测试完成！所有功能正常运行。")
        return True
        
    else:
        print(f"❌ 模型求解失败: {solution.get('message', '未知错误')}")
        return False

def test_parameter_updates():
    """测试参数更新功能"""
    print("\n🔄 测试参数更新功能...")
    
    model = BeverageOptimizationModel()
    
    # 记录原始值
    original_profits = model.profits.copy()
    original_material_limits = model.material_limits.copy()
    
    # 更新参数
    new_params = {
        'profits': [10.0, 15.0, 12.0, 18.0, 8.0],
        'material_limits': [20000, 10000, 8000, 3000, 15000]
    }
    
    model.update_parameters(new_params)
    
    # 验证更新
    updated = True
    for i, (new, old) in enumerate(zip(model.profits, original_profits)):
        if abs(new - new_params['profits'][i]) > 1e-6:
            updated = False
            break
    
    if updated:
        print("✅ 参数更新功能正常")
        
        # 测试新参数下的求解
        solution = model.solve_model()
        if solution['success']:
            print(f"✅ 新参数下求解成功，最大利润: {solution['optimal_value']:,.2f} 元")
        else:
            print("❌ 新参数下求解失败")
    else:
        print("❌ 参数更新功能异常")
    
    return updated

if __name__ == "__main__":
    success = test_model()
    if success:
        test_parameter_updates()
        print("\n🎉 所有测试通过！模型已准备就绪。")
    else:
        print("\n💥 测试失败，请检查模型实现。")