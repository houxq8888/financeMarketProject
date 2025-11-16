#!/usr/bin/env python
# coding: utf-8

# 金融市场微观结构分析（模拟数据版）

import pandas as pd
import numpy as np
import datetime as dt
import statsmodels.formula.api as smf

print("=== 金融市场微观结构分析（模拟数据版） ===")

# --------------------------
# 1. 生成模拟数据
# --------------------------
print("\n1. 正在生成模拟数据...")

# 生成股票列表
stock_list = [f"{i:06d}" for i in range(1, 101)]  # 100只股票

# 生成时间范围
months = pd.period_range(start='2021-01', end='2024-12', freq='M')  # 48个月

all_data = []

for stock_code in stock_list:
    for month in months:
        # 生成月度数据
        data_point = {
            'stk_id': stock_code,
            'stk_name': f"股票{stock_code}",
            'month': month,
            'close': np.random.normal(10, 2),  # 收盘价
            'VOL': np.random.uniform(1000000, 100000000),  # 成交量
            'amount': np.random.uniform(10000000, 1000000000),  # 成交额
            'market_type': "主板",
            'industry_code': np.random.choice(['C', 'A', 'B', 'D', 'E', 'F', 'G'], p=[0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
            'listing_date': dt.datetime(2010, 1, 1)
        }
        all_data.append(data_point)

data = pd.DataFrame(all_data)
print(f"   模拟数据生成完成！样本量：{data.shape[0]}")

# --------------------------
# 2. 数据预处理
# --------------------------
print("\n2. 正在进行数据预处理...")

# 计算ILLIQ（非流动性指标）
data['ILLIQ'] = data['amount'] / data['VOL'] / data['close']

# 计算TURN（换手率）
data['TURN'] = data['VOL'] / 1000000

# 生成核心变量
# 处理组：假设政策影响一半的股票
data['Treat'] = np.random.choice([0, 1], size=len(data), p=[0.5, 0.5])

# 政策后：假设政策实施时间为2023年4月
data['Post'] = (data['month'] >= pd.Period('2023-04', freq='M')).astype(int)

# 交互项
data['Treat_Post'] = data['Treat'] * data['Post']

# 生成其他控制变量
data['SIZE'] = np.random.normal(10000000000, 5000000000, size=len(data))
data['ROE'] = np.random.normal(0.1, 0.05, size=len(data))
data['GROWTH'] = np.random.normal(0.2, 0.1, size=len(data))
data['LEV'] = np.random.normal(0.5, 0.2, size=len(data))

print("   数据预处理完成！")

# --------------------------
# 3. 数据清洗
# --------------------------
print("\n3. 正在清洗数据...")

# 剔除上市不足1年企业
sample_start = dt.datetime(2021, 1, 1)
data['listing_date'] = pd.to_datetime(data['listing_date'], errors='coerce')
data['list_days'] = (sample_start - data['listing_date']).dt.days
data = data[data['list_days'] >= 365]

# 剔除金融行业
data = data[data['industry_code'] != 'J']

# 1%分位缩尾处理
def winsorize(df, cols, lower=0.01, upper=0.99):
    for col in cols:
        q_low = df[col].quantile(lower)
        q_high = df[col].quantile(upper)
        df[col] = df[col].clip(q_low, q_high)
    return df

winsor_cols = ['ILLIQ', 'TURN', 'SIZE', 'ROE', 'GROWTH', 'LEV']
data = winsorize(data, winsor_cols)

print(f"   清洗完成！剩余样本量：{data.shape[0]}")

# --------------------------
# 4. 基准回归
# --------------------------
print("\n4. 正在进行基准回归（DID模型）...")

if data.shape[0] < 10:
    print("   样本量不足，无法进行回归")
    exit()

try:
    # 确保stk_id是字符串格式
    data['stk_id_str'] = data['stk_id'].astype(str)
    data['month_str'] = data['month'].astype(str)
    
    # ILLIQ回归
    model_illiq = smf.ols(
        formula='ILLIQ ~ Treat_Post + SIZE + ROE + GROWTH + LEV + C(stk_id_str) + C(month_str)',
        data=data
    )
    result_illiq = model_illiq.fit(cov_type='cluster', cov_kwds={'groups': data['stk_id_str']})
    
    # TURN回归
    model_turn = smf.ols(
        formula='TURN ~ Treat_Post + SIZE + ROE + GROWTH + LEV + C(stk_id_str) + C(month_str)',
        data=data
    )
    result_turn = model_turn.fit(cov_type='cluster', cov_kwds={'groups': data['stk_id_str']})
    
    print("   回归完成！")
    print("\n=== 基准回归结果（ILLIQ） ===")
    print(result_illiq.summary().tables[1])
    print("\n=== 基准回归结果（TURN） ===")
    print(result_turn.summary().tables[1])
    
    # 保存结果
    with pd.ExcelWriter('模拟数据回归结果.xlsx') as writer:
        pd.DataFrame(result_illiq.params).to_excel(writer, sheet_name='ILLIQ系数')
        pd.DataFrame(result_turn.params).to_excel(writer, sheet_name='TURN系数')
    print("\n   回归结果已保存到 '模拟数据回归结果.xlsx'")
    
except Exception as e:
    print(f"   回归出错：{str(e)}")
    import traceback
    traceback.print_exc()
