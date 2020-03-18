# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 16:35:40 2020

@author: 將軍
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM,Dense
from sklearn.metrics import r2_score
from keras.models import Sequential
import keras.backend as K
from keras.callbacks import EarlyStopping



df = pd.read_csv('2330.TW.csv')
df = df.dropna() 

# label欄
target_columns = pd.DataFrame(df['Close'])

# feature欄
feature_columns = ['Open', 'High', 'Low', 'Volume']

#對featre欄位做正規化，方便訓練
scaler = MinMaxScaler()
feature_transform_data = scaler.fit_transform(df[feature_columns])
feature_transform = pd.DataFrame(columns=feature_columns, data=feature_transform_data, index=df.index)

#要預測n+1天的股價，將2019年1月的資料命名為X_test、y_test，其他的命名為X_train、y_train
target_columns = target_columns.shift(-1)
y_test = target_columns[-22:-1]
y_train = target_columns[:-22]

X_test = feature_transform[-22:-1]
X_train = feature_transform[:-22]

X_train =np.array(X_train)
X_test =np.array(X_test)

#維度轉換，來符合LSTM的input
X_tr_t = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_tst_t = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

#開始深度學習
K.clear_session()
model_lstm = Sequential()
model_lstm.add(LSTM(16, input_shape=(1, X_train.shape[1]), activation='relu', return_sequences=False))
model_lstm.add(Dense(1))
model_lstm.compile(loss='mean_squared_error', optimizer='adam')
early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
history_model_lstm = model_lstm.fit(X_tr_t, y_train, epochs=20, batch_size=10, verbose=1, shuffle=False, callbacks=[early_stop])

#R_Square愈接近1.0，代表此模式愈有解釋能力。
y_test_pred_lstm = model_lstm.predict(X_tst_t)
y_train_pred_lstm = model_lstm.predict(X_tr_t)

print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred_lstm)))
print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_test_pred_lstm)))



#將後validation:72899筆的資料預測股價和真實股價做比對
plt.plot(y_test.index, y_test,'b', label='Actual')
plt.plot(y_test.index, y_test_pred_lstm,'r', label='Predict')
plt.ylabel('Price')
#plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
#plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xlabel('Date')
plt.title('LSTM Predict vs Actual')
plt.legend()

