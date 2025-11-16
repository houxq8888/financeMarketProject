#!/usr/bin/env python
# coding: utf-8

"""
金融市场微观结构分析（模拟数据版）
使用模拟数据进行DID模型分析，确保代码逻辑正确且可运行
"""

import pandas as pd
import numpy as np
import datetime as dt
import statsmodels.formula.api as smf
import statsmodels.api as sm

print("=== 金融市场微观结构分析（模拟数据版） ===")

# --------------------------
# 1. 生成模拟数据
# --------------------------
print("\n1. 正在生成模拟数据...")

# 设置随机种子
np.random.seed(42)

# 生成5000个样本
n_samples = 5000

# 生成股票基本信息
stock_codes = [f"{i:06d}" for i in range(100)]  # 100只股票
months = pd.date_range('2020-01-01', '2024-12-31', freq='M')  # 5年月度数据

# 创建数据框
data = pd.DataFrame({
    'stk_id': np.random.choice(stock_codes, n_samples),
    'stk_name': np.random.choice([f"股票{i:06d}" for i in range(100)], n_samples),
    'month': np.random.choice(months, n_samples),
    'market_type': np.random.choice(['主板', '中小板', '创业板'], n_samples),
    'industry_code': np.random.choice(['C', 'G', 'I', 'K', 'M'], n_samples),
    'listing_date': np.random.choice(pd.date_range('2000-01-01', '2020-01-01'), n_samples),
    'close': np.random.normal(10, 5, n_samples),  # 收盘价
    'VOL': np.random.normal(1000000, 500000, n_samples),  # 成交量
    'amount': np.random.normal(10000000, 5000000, n_samples),  # 成交额
    'SIZE': np.random.normal(10000000000, 5000000000, n_samples),  # 市值
    'ROE': np.random.normal(0.1, 0.05, n_samples),  # 净资产收益率
    'GROWTH': np.random.normal(0.2, 0.1, n_samples),  # 营业收入增长率
    'LEV': np.random.normal(0.5, 0.2, n_samples),  # 资产负债率
    'GDP': np.random.normal(10000000000000, 5000000000000, n_samples),  # GDP
    'DIS': np.random.normal(50, 10, n_samples)  # 分析师关注度
})

# 将month转换为period类型
data['month'] = data['month'].dt.to_period('M')

print(f"   生成数据完成！样本量：{data.shape[0]}")

# --------------------------
# 2. 数据预处理
# --------------------------
print("\n2. 正在进行数据预处理...")

# 生成核心变量
# 处理组：假设随机分配
data['Treat'] = np.random.choice([0, 1], size=len(data), p=[0.5, 0.5])

# 政策后：假设政策实施时间为2023年4月
data['Post'] = (data['month'] >= pd.Period('2023-04', freq='M')).astype(int)

# 交互项
data['Treat_Post'] = data['Treat'] * data['Post']

# 计算流动性指标
# ILLIQ（非流动性指标）
data['ILLIQ'] = data['amount'] / data['VOL'] / data['close']

# TURN（换手率）
data['TURN'] = data['VOL'] / 1000000  # 简化处理

# 添加market_cap作为控制变量
data['market_cap'] = data['SIZE'].copy()

print("   数据预处理完成！")

# --------------------------
# 3. 数据清洗
# --------------------------
print("\n3. 正在清洗数据...")

# 剔除ST企业（模拟数据中简单过滤）
data = data[~data['stk_name'].str.contains('ST', na=False) & ~data['stk_name'].str.contains('*ST', na=False)]

# 剔除上市不满一年的企业
one_year = dt.timedelta(days=365)
data['listing_days'] = (pd.to_datetime(data['month'].dt.strftime('%Y-%m-%d')) - data['listing_date']).dt.days
data = data[data['listing_days'] >= 365]

# 剔除净资产收益率小于-1的企业
data = data[data['ROE'] >= -1]

# 对连续变量进行1%水平的缩尾处理
continuous_vars = ['ILLIQ', 'TURN', 'SIZE', 'ROE', 'GROWTH', 'LEV', 'GDP', 'DIS']
for var in continuous_vars:
    data[var] = data[var].clip(lower=data[var].quantile(0.01), upper=data[var].quantile(0.99))

print(f"   数据清洗完成！剩余样本量：{data.shape[0]}")

# --------------------------
# 4. 基准回归分析
# --------------------------
print("\n4. 正在进行基准回归分析...")

# DID模型：ILLIQ对Treat_Post的回归（加入个体和时间固定效应）
formula_illiq = 'ILLIQ ~ Treat_Post + Treat + Post + SIZE + ROE + GROWTH + LEV + GDP + DIS + C(stk_id) + C(month)'
model_illiq = smf.ols(formula_illiq, data=data)
result_illiq = model_illiq.fit()

# DID模型：TURN对Treat_Post的回归（加入个体和时间固定效应）
formula_turn = 'TURN ~ Treat_Post + Treat + Post + SIZE + ROE + GROWTH + LEV + GDP + DIS + C(stk_id) + C(month)'
model_turn = smf.ols(formula_turn, data=data)
result_turn = model_turn.fit()

print("   基准回归完成！")

# --------------------------
# 5. 输出结果
# --------------------------
print("\n5. 正在输出回归结果...")

# 输出到Excel
output_file = "基准回归结果（模拟数据）.xlsx"
with pd.ExcelWriter(output_file) as writer:
    # ILLIQ回归结果
    illiq_results_df = pd.DataFrame({
        '系数': result_illiq.params,
        '标准误': result_illiq.bse,
        't值': result_illiq.tvalues,
        'p值': result_illiq.pvalues,
        'R-squared': [result_illiq.rsquared] * len(result_illiq.params)
    })
    illiq_results_df.to_excel(writer, sheet_name='ILLIQ回归结果', index=True)

    # TURN回归结果
    turn_results_df = pd.DataFrame({
        '系数': result_turn.params,
        '标准误': result_turn.bse,
        't值': result_turn.tvalues,
        'p值': result_turn.pvalues,
        'R-squared': [result_turn.rsquared] * len(result_turn.params)
    })
    turn_results_df.to_excel(writer, sheet_name='TURN回归结果', index=True)

print(f"   结果已输出到：{output_file}")

# 打印核心结果摘要
print("\n=== 基准回归结果摘要 ===")
print("\n1. ILLIQ回归结果：")
print(f"   Treat_Post系数: {result_illiq.params.get('Treat_Post', 0):.6f}")
print(f"   p值: {result_illiq.pvalues.get('Treat_Post', 1):.6f}")
print(f"   R-squared: {result_illiq.rsquared:.6f}")

print("\n2. TURN回归结果：")
print(f"   Treat_Post系数: {result_turn.params.get('Treat_Post', 0):.6f}")
print(f"   p值: {result_turn.pvalues.get('Treat_Post', 1):.6f}")
print(f"   R-squared: {result_turn.rsquared:.6f}")

print("\n=== 分析完成 ===")
