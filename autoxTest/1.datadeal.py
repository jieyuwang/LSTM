import pandas as pd
import numpy as np
import os
from tqdm import tqdm
# train = pd.read_csv('汽车销量预测/train.csv')
# test  = pd.read_csv('汽车销量预测/test.csv')
# train['time'] = train['time'].apply(lambda x: '20' + x)
# test['time'] = test['time'].apply(lambda x: '20' + x)
# month_map = {'Jan': '01-01',
#              'Feb': '02-01',
#              'Mar': '03-01',
#              'Apr': '04-01',
#              'May': '05-01',
#              'Jun': '06-01',
#              'Jul': '07-01',
#              'Aug': '08-01',
#              'Sep': '09-01',
#              'Oct': '10-01',
#              'Nov': '11-01',
#              'Dec': '12-01',
#             }
# for item in month_map:
#     key = item
#     val = month_map[item]
#     train['time'] = train['time'].apply(lambda x: x.replace(key, val))
#     test['time'] = test['time'].apply(lambda x: x.replace(key, val))
# train.to_csv('./train.csv', index = False)
# test.to_csv('./test.csv', index = False)
#
#
#
#
# import pandas
#
# # data = pandas.read_csv("D:\project\pycharm\LSTM\StoreSales\data\data_v1_train.csv")
# # print(data)
# timeList = pandas.date_range('1/6/2022', '20220106 23:59:59', freq='H')
# print(timeList)

list = []
for i in range(1, 73):
    if i >= 2:
        break
    for j in range(0, 24):
        list.append(i)
print(len(list))
# print(pandas.DataFrame(list))
df = pd.DataFrame()
df['ID'] = pd.DataFrame(list)
df['Time'] = pd.DataFrame(timeList)
df.to_csv('folwTest.csv',index=None)
# pandas.DataFrame(pandas.concat(list,timeList))