import pandas as pd 
import numpy as np
# import tushare as ts
import sys
import xlwt
import seaborn
import matplotlib.pyplot as plt
import requests
import time
from numpy import *

# 读取财务指标数据
fina_indicators_2019_Q1 = pd.read_excel('fina_indicators_2019_Q1.xlsx')

# 处理财务数据数据库
for i in fina_indicators_2019_Q1.columns[1:]:
    if len(str(i)) == 9:
        fina_indicators_2019_Q1.rename(columns={i:str(str(i)[:-3])}, inplace=True)
fina_indicators_2019_Q1 = fina_indicators_2019_Q1.set_index(fina_indicators_2019_Q1['Unnamed: 0']).drop(['Unnamed: 0'],axis=1)
fina_indicators_2019_Q1 = fina_indicators_2019_Q1.drop('end_date')
fina_indicators_2019_Q1 = fina_indicators_2019_Q1.drop('ann_date')

# 读取股票收益率数据
data = pd.read_excel('return_rate_data.xlsx',sheet_name='Sheet2')
data = data.fillna(0)
data.set_index(["ts_code"], inplace=True)
data = data.T
for i in data.columns:
    if len(str(i)) == 1:
        data.rename(columns={i:'00000'+str(i)}, inplace=True)
    if len(str(i)) == 2:
        data.rename(columns={i:'0000'+str(i)}, inplace=True)
    if len(str(i)) == 3:
        data.rename(columns={i:'000'+str(i)}, inplace=True)
    if len(str(i)) == 4:
        data.rename(columns={i:'00'+str(i)}, inplace=True)
    if len(str(i)) == 5:
        data.rename(columns={i:'0'+str(i)}, inplace=True)
    if len(str(i)) == 6:
        data.rename(columns={i:str(i)}, inplace=True)

# 算股票累积收益率
return_list = []
for i in data.columns:
    investment = 1
    for a in data[i]:
        investment = investment * (1 + a)
    return_list.append(investment)

# 比较好坏股票
return_mean = mean(return_list)
return_label_list = []
for i in return_list:
    if i > return_mean:       
        return_label_list.append(1)
    elif i <= return_mean:
        return_label_list.append(0)

# 把好坏股票结果并到财务指标里
return_label = pd.DataFrame(columns=data.columns).T
return_label['return_label']=return_label_list

training_dataset = pd.concat([return_label.T,fina_indicators_2019_Q1],join='inner')
# 均值填充
values = dict([(col_name, col_mean) for col_name, col_mean in zip(training_dataset.T.columns.tolist(), training_dataset.T.mean().tolist())]) 
training_dataset.T.fillna(value=values, inplace=True)

training_dataset.T.to_excel('training_dataset.xlsx')