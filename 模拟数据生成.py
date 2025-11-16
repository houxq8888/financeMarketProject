#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import datetime as dt

# 设置随机种子确保结果可复现
np.random.seed(123)

# 生成样本数据
n_samples = 10000

# 企业ID
stk_ids = [f'stock_{i}' for i in range(n_samples)]

# 股票名称
stk_names = [f'股票_{i}' for i in range(n_samples)]

# 市场类型（主板/创业板）
market_types = np.random.choice(['主板', '创业板'], size=n_samples, p=[0.6, 0.4])

# 上市日期（2010-01-01至2020-12-31）
listing_dates = pd.date_range(start='2010-01-01', end='2020-12-31', periods=n_samples)

# 行业代码（排除J）
industry_codes = np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q'], size=n_samples)

# 月度日期（2021-01至2025-12）
monthly_dates = pd.date_range(start='2021-01', end='2025-12', freq='MS')
monthly_dates = np.random.choice(monthly_dates, size=n_samples)

# Treat（处理组虚拟变量：1=处理组，0=控制组）
treat = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])

# Post（政策实施虚拟变量：2023年1月及以后为1，之前为0）
post = monthly_dates >= dt.datetime(2023, 1, 1)
post = post.astype(int)

# Treat_Post（交互项：Treat*Post）
treat_post = treat * post

# ILLIQ（非流动性指标）
illiq = np.random.normal(0.01, 0.005, size=n_samples)

# TURN（换手率）
turn = np.random.normal(0.1, 0.05, size=n_samples)

# SPR（买卖价差）
spr = np.random.normal(0.005, 0.002, size=n_samples)

# DEPTH（订单深度）
depth = np.random.normal(1000000, 500000, size=n_samples)

# SIZE（市值）
size = np.random.normal(10000000000, 5000000000, size=n_samples)

# ROE（净资产收益率）
roe = np.random.normal(0.1, 0.05, size=n_samples)

# GROWTH（增长率）
growth = np.random.normal(0.2, 0.1, size=n_samples)

# LEV（杠杆率）
lev = np.random.normal(0.5, 0.2, size=n_samples)

# VOL（成交量）
vol = np.random.normal(10000000, 5000000, size=n_samples)

# GDP（国内生产总值）
gdp = np.random.normal(10000000000000, 5000000000000, size=n_samples)

# DIS（信息披露质量）
dis = np.random.normal(50, 10, size=n_samples)

# market_cap（市值，同SIZE）
market_cap = size.copy()

# 创建DataFrame
data = pd.DataFrame({
    'stk_id': stk_ids,
    'stk_name': stk_names,
    'market_type': market_types,
    'listing_date': listing_dates,
    'industry_code': industry_codes,
    'monthly_date': monthly_dates,
    'Treat': treat,
    'Post': post,
    'Treat_Post': treat_post,
    'ILLIQ': illiq,
    'TURN': turn,
    'SPR': spr,
    'DEPTH': depth,
    'SIZE': size,
    'ROE': roe,
    'GROWTH': growth,
    'LEV': lev,
    'VOL': vol,
    'GDP': gdp,
    'DIS': dis,
    'market_cap': market_cap
})

# 保存为Stata文件
data.to_stata('A股创业板主板数据_2021-2025.dta', write_index=False)
print(f'模拟数据已生成，样本量：{data.shape[0]}，变量数：{data.shape[1]}')