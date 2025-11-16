#!/usr/bin/env python
# coding: utf-8

# 简化版金融市场微观结构分析代码
# 无需依赖Stata文件，直接生成模拟数据并分析

import pandas as pd
import numpy as np
import datetime as dt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import norm

def main():
    print("=== 金融市场微观结构分析 ===")
    
    # --------------------------
    # 1. 生成模拟数据
    # --------------------------
    print("\n1. 正在生成模拟数据...")
    
    n_samples = 5000  # 调整样本量以提高运行速度
    np.random.seed(123)  # 固定随机种子确保结果可复现
    
    # 生成基础数据
    stk_ids = [f'stock_{i}' for i in range(n_samples)]
    stk_names = [f'股票_{i}' for i in range(n_samples)]
    market_types = np.random.choice(['主板', '创业板'], size=n_samples, p=[0.6, 0.4])
    listing_dates = pd.date_range(start='2010-01-01', end='2020-12-31', periods=n_samples)
    industry_codes = np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q'], size=n_samples)
    monthly_dates = pd.date_range(start='2021-01', end='2025-12', freq='MS')
    monthly_dates = np.random.choice(monthly_dates, size=n_samples)
    
    # 核心变量（Treat/Post）
    treat = (market_types == '创业板').astype(int)  # 处理组：创业板=1，主板=0
    post = (monthly_dates >= dt.datetime(2023, 4, 1)).astype(int)  # 政策后：2023年4月后=1
    treat_post = treat * post  # 交互项
    
    # 其他经济变量
    illiq = np.random.normal(0.01, 0.005, size=n_samples)
    turn = np.random.normal(0.1, 0.05, size=n_samples)
    spr = np.random.normal(0.005, 0.002, size=n_samples)
    depth = np.random.normal(1000000, 500000, size=n_samples)
    size = np.random.normal(10000000000, 5000000000, size=n_samples)
    roe = np.random.normal(0.1, 0.05, size=n_samples)
    growth = np.random.normal(0.2, 0.1, size=n_samples)
    lev = np.random.normal(0.5, 0.2, size=n_samples)
    vol = np.random.normal(10000000, 5000000, size=n_samples)
    gdp = np.random.normal(10000000000000, 5000000000000, size=n_samples)
    dis = np.random.normal(50, 10, size=n_samples)
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
    
    print(f"   生成成功！样本量：{data.shape[0]}")
    
    # --------------------------
    # 2. 数据清洗
    # --------------------------
    print("\n2. 正在清洗数据...")
    
    # 剔除ST企业
    data = data[~data['stk_name'].str.contains('ST', na=False) & ~data['stk_name'].str.contains('*ST', na=False)]
    
    # 剔除上市不足1年企业
    sample_start = dt.datetime(2021, 1, 1)
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
    
    winsor_cols = ['ILLIQ', 'TURN', 'SPR', 'DEPTH', 'SIZE', 'ROE', 'GROWTH', 'LEV', 'VOL']
    data = winsorize(data, winsor_cols)
    
    print(f"   清洗完成！剩余样本量：{data.shape[0]}")
    
    # --------------------------
    # 3. 基准回归
    # --------------------------
    print("\n3. 正在进行基准回归（DID模型）...")
    
    # 确保stk_id和monthly_date是字符串格式（用于固定效应）
    data['stk_id_str'] = data['stk_id'].astype(str)
    data['monthly_date_str'] = data['monthly_date'].dt.strftime('%Y-%m')
    
    try:
        # ILLIQ回归
        model_illiq = smf.ols(
            formula='ILLIQ ~ Treat_Post + SIZE + ROE + GROWTH + LEV + VOL + GDP + C(stk_id_str) + C(monthly_date_str)',
            data=data
        )
        result_illiq = model_illiq.fit(cov_type='cluster', cov_kwds={'groups': data['stk_id_str']})
        
        # TURN回归
        model_turn = smf.ols(
            formula='TURN ~ Treat_Post + SIZE + ROE + GROWTH + LEV + VOL + GDP + C(stk_id_str) + C(monthly_date_str)',
            data=data
        )
        result_turn = model_turn.fit(cov_type='cluster', cov_kwds={'groups': data['stk_id_str']})
        
        print("   回归完成！")
        print("\n=== 基准回归结果（ILLIQ） ===")
        print(result_illiq.summary().tables[1])
        print("\n=== 基准回归结果（TURN） ===")
        print(result_turn.summary().tables[1])
        
    except Exception as e:
        print(f"   回归出错：{str(e)}")
    
    print("\n=== 分析完成 ===")

if __name__ == "__main__":
    main()
