import pandas as pd 
import numpy as np
import tushare as ts
import sys
import xlwt
import seaborn
import matplotlib.pyplot as plt
import requests
import time

ts.set_token('9203526c35de212c7bb198947d2961fb20cf93beafcf547f3e50b24f')
pro = ts.pro_api()

stock_list = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name')

fina_indicators_2019_Q1 = pd.DataFrame(columns=stock_list['ts_code'].T,index=pro.query('fina_indicator', ts_code='600000.SH', start_date='20200101', end_date='20200930').columns[1:])
fina_indicators_2019_Q2 = pd.DataFrame(columns=stock_list['ts_code'].T,index=pro.query('fina_indicator', ts_code='600000.SH', start_date='20200101', end_date='20200930').columns[1:])
fina_indicators_2019_Q3 = pd.DataFrame(columns=stock_list['ts_code'].T,index=pro.query('fina_indicator', ts_code='600000.SH', start_date='20200101', end_date='20200930').columns[1:])
fina_indicators_2019_Q4 = pd.DataFrame(columns=stock_list['ts_code'].T,index=pro.query('fina_indicator', ts_code='600000.SH', start_date='20200101', end_date='20200930').columns[1:])
fina_indicators_2020_Q1 = pd.DataFrame(columns=stock_list['ts_code'].T,index=pro.query('fina_indicator', ts_code='600000.SH', start_date='20200101', end_date='20200930').columns[1:])

for i in stock_list['ts_code']:
    if i == '000002.SZ':
        a = pro.query('fina_indicator', ts_code=i, start_date='20190101', end_date='20201101')
        dates = a['end_date'].to_list()
        for date in dates:
            if int(date) == 20190331: #一季度
                if (a.loc[a['end_date'] == date].T[1:]).shape[0] == 107:#行数为107
                    fina_indicators_2019_Q1[i] = a.loc[a['end_date'] == date].T[1:].values
                else:
                    pass
            elif int(date) == 20190630:#二季度
                if (a.loc[a['end_date'] == date].T[1:]).shape[0] == 107:
                    fina_indicators_2019_Q2[i] = a.loc[a['end_date'] == date].T[1:].values
                else:
                    pass
            elif int(date) == 20190930:#三季度
                if (a.loc[a['end_date'] == date].T[1:]).shape[0] == 107:
                    fina_indicators_2019_Q3[i] = a.loc[a['end_date'] == date].T[1:].values
                else:
                    pass
            elif int(date) == 20191231:#四季度
                if (a.loc[a['end_date'] == date].T[1:]).shape[0] == 107:
                    fina_indicators_2019_Q4[i] = a.loc[a['end_date'] == date].T[1:].values
                else:
                    pass
            elif int(date) == 20200331:
                if (a.loc[a['end_date'] == date].T[1:]).shape[0] == 107:
                    fina_indicators_2020_Q1[i] = a.loc[a['end_date'] == date].T[1:].values
                else:
                    pass
            time.sleep(0.05)
            print('Finish------------------',i)

fina_indicators_2019_Q1.to_excel('fina_indicators_2019_Q1.xlsx')
fina_indicators_2019_Q2.to_excel('fina_indicators_2019_Q2.xlsx')
fina_indicators_2019_Q3.to_excel('fina_indicators_2019_Q3.xlsx')
fina_indicators_2019_Q4.to_excel('fina_indicators_2019_Q4.xlsx')
fina_indicators_2020_Q1.to_excel('fina_indicators_2020_Q1.xlsx')