import pandas as pd 
import numpy as np
import sys
import xlwt
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import time
from scipy import stats

# 读取训练集数据
pic = pd.read_excel('training_dataset.xlsx')

# 特征相关性热力图
data = pic[pic.columns[2:]]
sns.set_context({"figure.figsize":(6,6)})
sns.heatmap(data=data.corr(),square=True,cmap="RdBu_r")
plt.show() 

# 特征数据统计
skew_list = []
for i in pic.columns[2:]:
    skew_list.append(stats.skew(pic[i].values))
    
kurtosis_list = []
for i in pic.columns[2:]:
    kurtosis_list.append(stats.kurtosis(pic[i].values))
    
autocorr_list = []
for i in pic.columns[2:]:
    autocorr_list.append(pd.Series(pic[i].values).autocorr())
    
skew_list = pd.DataFrame(skew_list).T
kurtosis_list = pd.DataFrame(kurtosis_list).T
autocorr_list = pd.DataFrame(autocorr_list).T

a = pd.concat([skew_list,kurtosis_list,autocorr_list],keys=['skew','kurtosis','autocorr']).reset_index(drop=True)
a['key']=['skew','kurtosis','autocorr']
a = a.set_index(keys=['key'],drop=True)
a.columns = pic[pic.columns[2:]].describe().columns

features_describe = pic[pic.columns[2:]].describe().append(a)

print(features_describe)
features_describe.to_excel('dataset_description.xlsx')

# 部分数据统计图例
plt.figure(figsize=(8, 6))
sns.kdeplot(features_describe['eps'], shade=True,color='black',alpha=0.4)
#sns.kdeplot(SSEI_data_return['000001'], shade=True,color='red',alpha = 0.4,label='Index')
plt.xticks(size=12)
plt.yticks(size=12)
plt.xlabel('Earning Per Share',size=16)
plt.ylabel( 'Density',size=16)
plt.grid()
plt.show()



# 读取训练集数据
training_dataset = pd.read_excel('training_dataset.xlsx')
training_dataset.set_index(["Unnamed: 0"], inplace=True)
training_dataset = training_dataset.T

for i in training_dataset.columns:
    if len(str(i)) == 1:
        training_dataset.rename(columns={i:'00000'+str(i)}, inplace=True)
    if len(str(i)) == 2:
        training_dataset.rename(columns={i:'0000'+str(i)}, inplace=True)
    if len(str(i)) == 3:
        training_dataset.rename(columns={i:'000'+str(i)}, inplace=True)
    if len(str(i)) == 4:
        training_dataset.rename(columns={i:'00'+str(i)}, inplace=True)
    if len(str(i)) == 5:
        training_dataset.rename(columns={i:'0'+str(i)}, inplace=True)
    if len(str(i)) == 6:
        training_dataset.rename(columns={i:str(i)}, inplace=True)

# ROE 对于收益率影响情况
result = pd.qcut(training_dataset.T['roe'],4)
print(training_dataset.T.groupby(result)[['return_label']].mean())