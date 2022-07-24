import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras import optimizers, Sequential, Model
import tensorflow.keras.layers as L
sub_rowdata_group = pd.read_csv(r'../data/data_v1_train.csv')
print(sub_rowdata_group.shape)


train_hours = sub_rowdata_group[['Time','ID','flow']]
#train_hours = train_hours.sort_values('Time').groupby(['ID'], as_index=False)
#train_hours = train_hours.agg({'flow'})
#train_hours.columns = ['Time', 'ID', 'flow']
train_hours = train_hours.query('Time >= "2021-11-11 00:00:00" and Time <= "2022-01-05 00:00:00"')
print(train_hours.head())
train_hours['flow_cnt_hours'] = train_hours.sort_values('Time').groupby(['ID'])['flow'].shift(-1)
print(train_hours.head())

monthly_series = train_hours.pivot_table(index=['ID'], columns='Time',values='flow', fill_value=0).reset_index()
monthly_series.head()


data_series = pd.DataFrame(monthly_series)
test_network_ids = data_series['ID'].unique()
#print(data_series.head())
#print(data_series.shape)
from sklearn.model_selection import train_test_split
# 最后N条数据作为测试数据
testNum = 1
# 将数据分割为训练集和测试集，此时分割的数据集是二维数组（取最后12条数据作为测试数据）
train, test = data_series.iloc[:,:-testNum], monthly_series.iloc[:,-testNum:]
print(train.shape)
print(test.shape)
train, valid, Y_train, Y_valid = train_test_split(train, test.values, test_size=0.10, random_state=0)

#print(train)
#print(valid)
X_train = train.values.reshape((train.shape[0], train.shape[1], 1))
X_valid = valid.values.reshape((valid.shape[0], valid.shape[1], 1))

print("Train set reshaped", X_train.shape)
print("Validation set reshaped", X_valid.shape)





serie_size =  X_train.shape[1] # 12
n_features =  X_train.shape[2] # 1

epochs = 3
batch = 128
lr = 0.0001

lstm_model = Sequential()
lstm_model.add(L.LSTM(10, input_shape=(serie_size, n_features), return_sequences=True))
lstm_model.add(L.LSTM(6, activation='relu', return_sequences=True))
lstm_model.add(L.LSTM(1, activation='relu'))
lstm_model.add(L.Dense(10, kernel_initializer='glorot_normal', activation='relu'))
lstm_model.add(L.Dense(10, kernel_initializer='glorot_normal', activation='relu'))
lstm_model.add(L.Dense(1))
lstm_model.summary()

adam = optimizers.Adam(lr)
lstm_model.compile(loss='mean_absolute_error', optimizer='adam',metrics = ['mae'])


lstm_history = lstm_model.fit(X_train, Y_train,
                              validation_data=(X_valid, Y_valid),
                              batch_size=batch,
                              epochs=epochs,
                              verbose=2)

from sklearn.metrics import mean_absolute_error
print(X_valid.shape)
lstm_train_pred = lstm_model.predict(X_train)
lstm_val_pred = lstm_model.predict(X_valid)
print(lstm_val_pred.shape)
print('Train mae:', mean_absolute_error(Y_train, lstm_train_pred))
print('Validation mae:', mean_absolute_error(Y_valid, lstm_val_pred))



encoder_decoder = Sequential()
encoder_decoder.add(L.LSTM(serie_size, activation='relu', input_shape=(serie_size, n_features), return_sequences=True))
encoder_decoder.add(L.LSTM(6, activation='relu', return_sequences=True))
encoder_decoder.add(L.LSTM(1, activation='relu'))
encoder_decoder.add(L.RepeatVector(serie_size))
encoder_decoder.add(L.LSTM(serie_size, activation='relu', return_sequences=True))
encoder_decoder.add(L.LSTM(6, activation='relu', return_sequences=True))
encoder_decoder.add(L.TimeDistributed(L.Dense(1)))
encoder_decoder.summary()

adam = optimizers.Adam(lr)
encoder_decoder.compile(loss='mse', optimizer=adam)


encoder_decoder_history = encoder_decoder.fit(X_train, X_train,
                                              batch_size=batch,
                                              epochs=epochs,
                                              verbose=2)



rpt_vector_layer = Model(inputs=encoder_decoder.inputs, outputs=encoder_decoder.layers[3].output)
time_dist_layer = Model(inputs=encoder_decoder.inputs, outputs=encoder_decoder.layers[5].output)
encoder_decoder.layers

rpt_vector_layer_output = rpt_vector_layer.predict(X_train[:1])
print('Repeat vector output shape', rpt_vector_layer_output.shape)
print('Repeat vector output sample')
print(rpt_vector_layer_output[0])

time_dist_layer_output = time_dist_layer.predict(X_train[:1])
print('Time distributed output shape', time_dist_layer_output.shape)
print('Time distributed output sample')
print(time_dist_layer_output[0])

encoder = Model(inputs=encoder_decoder.inputs, outputs=encoder_decoder.layers[2].output)
train_encoded = encoder.predict(X_train)
validation_encoded = encoder.predict(X_valid)
print('Encoded time-series shape', train_encoded.shape)
print('Encoded time-series sample', train_encoded[0])

train['encoded'] = train_encoded
train['label'] = Y_train

valid['encoded'] = validation_encoded
valid['label'] = Y_valid

train.head(10)


last_month = serie_size - 1
Y_train_encoded = train['label']
train.drop('label', axis=1, inplace=True)
X_train_encoded = train[[last_month, 'encoded']]

Y_valid_encoded = valid['label']
valid.drop('label', axis=1, inplace=True)
X_valid_encoded = valid[[last_month, 'encoded']]

print("Train set", X_train_encoded.shape)
print("Validation set", X_valid_encoded.shape)


mlp_model = Sequential()
mlp_model.add(L.Dense(10, kernel_initializer='glorot_normal', activation='relu', input_dim=X_train_encoded.shape[1]))
mlp_model.add(L.Dense(10, kernel_initializer='glorot_normal', activation='relu'))
mlp_model.add(L.Dense(1))
mlp_model.summary()

adam = optimizers.Adam(lr)
mlp_model.compile(loss='mse', optimizer=adam)

mlp_history = mlp_model.fit(X_train_encoded.values, Y_train_encoded.values, epochs=epochs, batch_size=batch, validation_data=(X_valid_encoded, Y_valid_encoded), verbose=2)


fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(22,7))

ax1.plot(lstm_history.history['loss'], label='Train loss')
ax1.plot(lstm_history.history['val_loss'], label='Validation loss')
ax1.legend(loc='best')
ax1.set_title('Regular LSTM')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('MSE')

ax2.plot(mlp_history.history['loss'], label='Train loss')
ax2.plot(mlp_history.history['val_loss'], label='Validation loss')
ax2.legend(loc='best')
ax2.set_title('MLP with LSTM encoder')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('MSE')

plt.show()