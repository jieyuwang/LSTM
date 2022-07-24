from autox import AutoX

# 选择数据集
data_name = '汽车销量预测'
path = f'流量预测'

feature_type = {
    'test.csv': {
        'cityid': 'cat',
        'carid': 'cat',
        'time': 'datetime'},
    'train.csv': {
        'cityid': 'cat',
        'carid': 'cat',
        'time': 'datetime',
        'sales': 'num'}
}

autox = AutoX(target = 'sales', train_name = 'train.csv', test_name = 'test.csv',
               id = ['cityid'], path = path, time_series=True, ts_unit='day',time_col = 'time',
               feature_type = feature_type,metric='mae')
sub = autox.get_submit_ts()
print(sub)

