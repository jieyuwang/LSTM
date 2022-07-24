import pandas as pd
import numpy as np
import os
from tqdm import tqdm
train = pd.read_csv('汽车销量预测/train.csv')
test  = pd.read_csv('汽车销量预测/test.csv')
train['time'] = train['time'].apply(lambda x: '20' + x)
test['time'] = test['time'].apply(lambda x: '20' + x)
month_map = {'Jan': '01-01',
             'Feb': '02-01',
             'Mar': '03-01',
             'Apr': '04-01',
             'May': '05-01',
             'Jun': '06-01',
             'Jul': '07-01',
             'Aug': '08-01',
             'Sep': '09-01',
             'Oct': '10-01',
             'Nov': '11-01',
             'Dec': '12-01',
            }
for item in month_map:
    key = item
    val = month_map[item]
    train['time'] = train['time'].apply(lambda x: x.replace(key, val))
    test['time'] = test['time'].apply(lambda x: x.replace(key, val))
train.to_csv('./train.csv', index = False)
test.to_csv('./test.csv', index = False)