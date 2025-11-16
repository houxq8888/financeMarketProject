#!/usr/bin/env python
# coding: utf-8

# # 数据清洗代码

# In[ ]:


import pandas as pd
import numpy as np
import datetime as dt

# 1. 导入原始数据
data = pd.read_stata("A股创业板主板数据_2021-2025.dta")

# 2. 剔除ST/*ST企业
data = data[~data['stk_name'].str.contains('ST', na=False) & 
           ~data['stk_name'].str.contains('*ST', na=False)]

# 3. 剔除上市不足1年企业（上市日期距2021-01-01不足365天）
data['listing_date'] = pd.to_datetime(data['listing_date'], format='%Y-%m-%d')
sample_start = dt.datetime(2021, 1, 1)
data['list_days'] = (sample_start - data['listing_date']).dt.days
data = data[data['list_days'] >= 365]

# 4. 剔除金融行业（证监会行业代码J）
data = data[data['industry_code'] != 'J']

# 5. 1%分位缩尾处理连续变量
def winsorize(df, cols, lower=0.01, upper=0.99):
    for col in cols:
        q_low = df[col].quantile(lower)
        q_high = df[col].quantile(upper)
        df[col] = df[col].clip(q_low, q_high)
    return df

winsor_cols = ['ILLIQ', 'TURN', 'SPR', 'DEPTH', 'SIZE', 'ROE', 'GROWTH', 'LEV', 'VOL']
data = winsorize(data, winsor_cols)

# 6. 保存清洗后数据
data.to_stata("清洗后数据_2021-2025.dta", write_index=False)
print(f"清洗后样本量：{data.shape[0]}，变量数：{data.shape[1]}")


# 基准回归（双重差分模型）代码

# In[ ]:


import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# 1. 加载数据并预处理
data = pd.read_stata("清洗后数据_2021-2025.dta")
data['monthly_date'] = pd.to_datetime(data['monthly_date'])  # 确保日期格式正确

# 2. 定义核心变量
data['Treat'] = (data['market_type'] == '创业板').astype(int)  # 处理组：创业板=1，主板=0
data['Post'] = (data['monthly_date'] >= '2023-04-01').astype(int)  # 政策后：2023年4月后=1
data['Treat_Post'] = data['Treat'] * data['Post']  # 交互项：政策净效应

# 3. 双向固定效应DID回归（被解释变量：ILLIQ）
# 加入个体固定效应（stk_id）和时间固定效应（monthly_date）
model_illiq = smf.ols(
    formula='ILLIQ ~ Treat_Post + SIZE + ROE + GROWTH + LEV + VOL + GDP + C(stk_id) + C(monthly_date)',
    data=data
)
result_illiq = model_illiq.fit(cov_type='cluster', cov_kwds={'groups': data['stk_id']})  # 聚类稳健标准误

# 4. 双向固定效应DID回归（被解释变量：TURN）
model_turn = smf.ols(
    formula='TURN ~ Treat_Post + SIZE + ROE + GROWTH + LEV + VOL + GDP + C(stk_id) + C(monthly_date)',
    data=data
)
result_turn = model_turn.fit(cov_type='cluster', cov_kwds={'groups': data['stk_id']})

# 5. 输出结果
print("=== 基准回归结果（ILLIQ） ===")
print(result_illiq.summary().tables[1])  # 核心系数表
print("\n=== 基准回归结果（TURN） ===")
print(result_turn.summary().tables[1])

# 保存结果到Excel
with pd.ExcelWriter('基准回归结果.xlsx') as writer:
    pd.DataFrame(result_illiq.params).to_excel(writer, sheet_name='ILLIQ系数')
    pd.DataFrame(result_illiq.bse).to_excel(writer, sheet_name='ILLIQ标准误')
    pd.DataFrame(result_turn.params).to_excel(writer, sheet_name='TURN系数')
    pd.DataFrame(result_turn.bse).to_excel(writer, sheet_name='TURN标准误')


# 中介效应检验代码（信息披露质量路径）

# In[ ]:


import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import norm

# 1. 加载数据
data = pd.read_stata("清洗后数据_2021-2025.dta")
data['Treat_Post'] = data['Treat'] * data['Post']  # 沿用前文定义的交互项

# 2. 第一步：总效应（同基准回归ILLIQ模型）
model_total = smf.ols(
    formula='ILLIQ ~ Treat_Post + SIZE + ROE + GROWTH + LEV + VOL + GDP + C(stk_id) + C(monthly_date)',
    data=data
)
result_total = model_total.fit(cov_type='cluster', cov_kwds={'groups': data['stk_id']})
total_effect = result_total.params['Treat_Post']

# 3. 第二步：核心解释变量对中介变量（DIS）的影响
model_med1 = smf.ols(
    formula='DIS ~ Treat_Post + SIZE + ROE + GROWTH + LEV + VOL + GDP + C(stk_id) + C(monthly_date)',
    data=data
)
result_med1 = model_med1.fit(cov_type='cluster', cov_kwds={'groups': data['stk_id']})
a = result_med1.params['Treat_Post']  # 核心解释变量→中介变量的系数

# 4. 第三步：加入中介变量后的回归
model_med2 = smf.ols(
    formula='ILLIQ ~ Treat_Post + DIS + SIZE + ROE + GROWTH + LEV + VOL + GDP + C(stk_id) + C(monthly_date)',
    data=data
)
result_med2 = model_med2.fit(cov_type='cluster', cov_kwds={'groups': data['stk_id']})
b = result_med2.params['DIS']  # 中介变量→被解释变量的系数
direct_effect = result_med2.params['Treat_Post']  # 直接效应

# 5. Bootstrap检验中介效应（重复500次）
np.random.seed(123)  # 固定随机种子，确保结果可复现
n_boot = 500
boot_mediation = []

for _ in range(n_boot):
    # 有放回抽样（按企业聚类抽样）
    unique_ids = data['stk_id'].unique()
    boot_ids = np.random.choice(unique_ids, size=len(unique_ids), replace=True)
    boot_data = pd.concat([data[data['stk_id'] == id_] for id_ in boot_ids])
    
    # 重新估计第二步和第三步系数
    boot_med1 = smf.ols('DIS ~ Treat_Post + SIZE + ROE + GROWTH + LEV + VOL + GDP + C(stk_id) + C(monthly_date)', data=boot_data).fit()
    boot_med2 = smf.ols('ILLIQ ~ Treat_Post + DIS + SIZE + ROE + GROWTH + LEV + VOL + GDP + C(stk_id) + C(monthly_date)', data=boot_data).fit()
    boot_mediation.append(boot_med1.params['Treat_Post'] * boot_med2.params['DIS'])

# 计算中介效应及置信区间
mediation_effect = a * b  # 中介效应= a*b
boot_ci = np.percentile(boot_mediation, [2.5, 97.5])  # 95%置信区间
mediation_ratio = abs(mediation_effect) / abs(total_effect)  # 中介效应占比

# 6. 输出结果
print(f"总效应（Treat_Post系数）：{total_effect:.4f}")
print(f"中介变量系数a（Treat_Post→DIS）：{a:.4f}")
print(f"中介变量系数b（DIS→ILLIQ）：{b:.4f}")
print(f"中介效应（a*b）：{mediation_effect:.4f}")
print(f"中介效应占比：{mediation_ratio:.2%}")
print(f"Bootstrap 95%置信区间：[{boot_ci[0]:.4f}, {boot_ci[1]:.4f}]")
print(f"是否显著（区间不含0）：{boot_ci[0] < 0 < boot_ci[1] is False}")


# 异质性分析代码（按市值分组）

# In[ ]:


import pandas as pd
import statsmodels.formula.api as smf

# 1. 加载数据并分组
data = pd.read_stata("清洗后数据_2021-2025.dta")
data['Treat_Post'] = data['Treat'] * data['Post']

# 2. 按市值中位数分组（全样本市值中位数）
market_cap_median = data['market_cap'].median()
data['small_cap'] = (data['market_cap'] <= market_cap_median).astype(int)  # 小盘股组
data['large_cap'] = (data['market_cap'] > market_cap_median).astype(int)   # 大盘股组

# 3. 小盘股组回归
model_small = smf.ols(
    formula='ILLIQ ~ Treat_Post + SIZE + ROE + GROWTH + LEV + VOL + GDP + C(stk_id) + C(monthly_date)',
    data=data[data['small_cap'] == 1]
)
result_small = model_small.fit(cov_type='cluster', cov_kwds={'groups': data['stk_id']})

# 4. 大盘股组回归
model_large = smf.ols(
    formula='ILLIQ ~ Treat_Post + SIZE + ROE + GROWTH + LEV + VOL + GDP + C(stk_id) + C(monthly_date)',
    data=data[data['large_cap'] == 1]
)
result_large = model_large.fit(cov_type='cluster', cov_kwds={'groups': data['stk_id']})

# 5. 输出结果
print("=== 小盘股组回归结果 ===")
print(result_small.summary().tables[1])
print("\n=== 大盘股组回归结果 ===")
print(result_large.summary().tables[1])

# 对比核心系数
print(f"\n小盘股组Treat_Post系数：{result_small.params['Treat_Post']:.4f}")
print(f"大盘股组Treat_Post系数：{result_large.params['Treat_Post']:.4f}")
print(f"系数差异：{result_small.params['Treat_Post'] - result_large.params['Treat_Post']:.4f}")


# In[ ]:





# In[ ]:





# In[ ]:




