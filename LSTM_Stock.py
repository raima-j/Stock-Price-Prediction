#Long-Short Term Memory based model for Stock Prediction

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from datetime import date,timedelta
import math
import matplotlib.dates as mdates
import random
import time

#For eager execution.
tf.config.run_functions_eagerly(True)

#The Dataset for Long Short Term Memory (LSTM)
def stock_data(symbol,start_date,end_date):
    stocks=yf.download(symbol,start=start_date,end=end_date)
    return stocks

def lstm_dataset(data,look_back):
    x,y=[],[]
    for i in range(len(data)-look_back):
        x.append(data[i:i+look_back])
        y.append(data[i+look_back])
    return np.array(x),np.array(y)
    
#For the Beyond Meat company.
sym=['BYND'] 
start_date = '2019-07-01' #5 years of historical data.
prediction_date='2023-06-01'  
end_date = '2023-11-10' #Taking one year.
look_back = 60   #Number of past days to consider for LSTM input.

#Data retreival using YFinance.
data=stock_data(sym,start_date,end_date)['Close'].values.reshape(-1,1)#converts from 1D to  2D Matrix
dataD=stock_data(sym,start_date,end_date)
print ('\nSample data for %s:'% (sym))
print (dataD.head(5) )

#For the start time.
start_time=time.time()

dates_data=dataD.index 
dates_list= ([i.strftime("%Y-%m-%d") for i in dates_data.date])
dates_list=pd.DataFrame(dates_list)
dates_list=(dates_list.values.reshape(-1,1))

#Beyond Meat Data Scaling and Model fitting.
scaler=MinMaxScaler() 
data=scaler.fit_transform(data) 
x_lstm,y_lstm=lstm_dataset(data,look_back)
print ('\nScaled dataset:\n',data[:5]) 

train_size=math.ceil(len(x_lstm)*0.8)
x_train_lstm, x_test_lstm=x_lstm[:train_size],x_lstm[train_size:]
y_train_lstm, y_test_lstm=y_lstm[:train_size],y_lstm[train_size:]
X_train_lstm = x_train_lstm.reshape(x_train_lstm.shape[0], look_back, 1)
X_test_lstm = x_test_lstm.reshape(x_test_lstm.shape[0], look_back, 1)

model_lstm=Sequential() #Linear stack of layers for the model.
model_lstm.add(LSTM(50,input_shape=(look_back,1))) 
model_lstm.add(Dense(1)) 
model_lstm.compile(loss='mean_squared_error',optimizer='adam') 
model_lstm.fit(x_train_lstm,y_train_lstm,epochs=100,batch_size=32,verbose=1)
input_shape = model_lstm.layers[0].input_shape
print("\nExpected input shape of LSTM model:", input_shape)

#Prediction and visualization.
y_pred_lstm=model_lstm.predict(x_test_lstm)
y_pred_lstm=scaler.inverse_transform(y_pred_lstm)
y_test_lstm=scaler.inverse_transform(y_test_lstm)
mse_lstm = mean_squared_error(y_test_lstm, y_pred_lstm)
print("\nMean Squared Error (LSTM):", mse_lstm)

dates_pred = pd.date_range(start=prediction_date, periods=len(y_pred_lstm), freq='D')
dates_actual = dates_data[train_size + look_back:]
end_time=time.time()
training_time=end_time-start_time

#The plot.
plt.figure(figsize=(12, 6))
plt.plot(dates_actual, y_pred_lstm, label='LSTM Predictions', color='green')
plt.scatter(dates_actual, y_test_lstm, color='blue', label='Actual Price', marker='o')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Stock Price Prediction for %s (LSTM)' % (sym))

plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1)) #to get weekly dates instead of crowding

# Format x-axis tick labels as 'yyyy-mm-dd'
date_labels = [date.strftime('%Y-%m-%d') for date in dates_pred]
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(8))
plt.tight_layout()
plt.show()

print ("\nThe training time is: ",training_time)