#!/usr/bin/env python
# coding: utf-8

"""
金融市场微观结构分析（极简版）
使用最基础的Python和pandas功能，确保代码可运行
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm

print("=== 金融市场微观结构分析（极简版） ===")

# --------------------------
# 1. 生成简单模拟数据
# --------------------------
print("\n1. 正在生成模拟数据...")

# 创建简单的数据
n_obs = 1000  # 1000个观测值

# 生成核心变量
Treat = np.random.randint(0, 2, size=n_obs)  # 处理组分配
Post = np.random.randint(0, 2, size=n_obs)   # 政策前后
Treat_Post = Treat * Post                    # 交互项

# 生成流动性指标
ILLIQ = np.random.normal(0.1, 0.05, size=n_obs)  # 非流动性指标
TURN = np.random.normal(100, 20, size=n_obs)     # 换手率

# 生成控制变量
SIZE = np.random.normal(10000000000, 5000000000, size=n_obs)
ROE = np.random.normal(0.1, 0.05, size=n_obs)

# 创建数据框
df = pd.DataFrame({
    'ILLIQ': ILLIQ,
    'TURN': TURN,
    'Treat': Treat,
    'Post': Post,
    'Treat_Post': Treat_Post,
    'SIZE': SIZE,
    'ROE': ROE
})

print(f"   生成数据完成！样本量：{df.shape[0]}")

# --------------------------
# 2. 简单回归分析
# --------------------------
print("\n2. 正在进行简单回归分析...")

# DID模型：ILLIQ对Treat_Post的回归
X = df[['Treat_Post', 'Treat', 'Post', 'SIZE', 'ROE']]
X = sm.add_constant(X)  # 添加常数项
y = df['ILLIQ']

model = sm.OLS(y, X)
result = model.fit()

print("   回归分析完成！")

# --------------------------
# 3. 输出结果
# --------------------------
print("\n3. 回归结果摘要：")
print(result.summary())

print("\n=== 分析完成 ===")
