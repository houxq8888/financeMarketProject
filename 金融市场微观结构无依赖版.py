#!/usr/bin/env python
# coding: utf-8

"""
金融市场微观结构分析（无依赖版）
不使用任何外部库，仅展示DID模型的核心逻辑
"""

import random

print("=== 金融市场微观结构分析（无依赖版） ===")

# --------------------------
# 1. 生成模拟数据
# --------------------------
print("\n1. 正在生成模拟数据...")

# 设定参数
n_firms = 100  # 公司数量
n_months = 24  # 时间跨度（2年）
policy_month = 12  # 政策实施时间（第12个月）

data = []
for firm in range(n_firms):
    # 随机分配处理组和控制组
    treat = random.choice([0, 1])
    
    for month in range(n_months):
        # 政策前后标识
        post = 1 if month >= policy_month else 0
        
        # 计算交互项
        treat_post = treat * post
        
        # 生成流动性指标（ILLIQ）
        # 基础流动性 + 处理效应 + 政策效应 + 交互效应 + 随机扰动
        base_illiq = 0.1  # 基础非流动性
        treat_effect = 0.02 if treat == 1 else 0  # 处理组基础效应
        post_effect = 0.03 if post == 1 else 0  # 政策整体效应
        did_effect = 0.05 if treat_post == 1 else 0  # DID交互效应
        noise = random.gauss(0, 0.01)  # 随机扰动
        illiq = base_illiq + treat_effect + post_effect + did_effect + noise
        
        # 生成换手率（TURN）
        base_turn = 100  # 基础换手率
        treat_effect_turn = 5 if treat == 1 else 0  # 处理组基础效应
        post_effect_turn = 10 if post == 1 else 0  # 政策整体效应
        did_effect_turn = -8 if treat_post == 1 else 0  # DID交互效应
        noise_turn = random.gauss(0, 2)  # 随机扰动
        turn = base_turn + treat_effect_turn + post_effect_turn + did_effect_turn + noise_turn
        
        # 添加到数据
        data.append([firm, month, treat, post, treat_post, illiq, turn])

print(f"   生成数据完成！样本量：{len(data)}")

# --------------------------
# 2. 手动实现DID模型
# --------------------------
print("\n2. 正在进行DID模型估计...")

# 分离不同组别的数据
control_pre = []  # 控制组政策前
control_post = []  # 控制组政策后
treat_pre = []    # 处理组政策前
treat_post = []   # 处理组政策后

for obs in data:
    firm, month, treat, post, treat_post, illiq, turn = obs
    if treat == 0:
        if post == 0:
            control_pre.append(illiq)
        else:
            control_post.append(illiq)
    else:
        if post == 0:
            treat_pre.append(illiq)
        else:
            treat_post.append(illiq)

# 计算平均值
def mean(arr):
    return sum(arr) / len(arr) if len(arr) > 0 else 0

# 计算DID效应
control_diff = mean(control_post) - mean(control_pre)
treat_diff = mean(treat_post) - mean(treat_pre)
did_effect = treat_diff - control_diff

print("   DID模型估计完成！")

# --------------------------
# 3. 输出结果
# --------------------------
print("\n=== DID模型分析结果 ===")
print(f"控制组政策前后变化：{control_diff:.4f}")
print(f"处理组政策前后变化：{treat_diff:.4f}")
print(f"DID交互效应：{did_effect:.4f}")
print("\n=== 经济意义解释 ===")
print("- 控制组政策前后变化：反映了时间趋势对流动性的影响")
print("- 处理组政策前后变化：反映了处理组整体的政策效应")
print("- DID交互效应：反映了政策对处理组的净影响（去除了时间趋势和组间差异）")

print("\n=== 分析完成 ===")
