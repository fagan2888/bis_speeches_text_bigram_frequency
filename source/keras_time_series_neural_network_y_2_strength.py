import tensorflow as tf
print(tf.__version__)
from tensorflow import keras
from tensorflow.python.keras import backend as k

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

from tensorflow.keras import backend

from utils import * 
from config import parameters

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from math import sqrt
import math
from statsmodels.tsa.stattools import grangercausalitytests

from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

def rescale_rolling(values, rolling_window):
    df = pd.DataFrame(values)
    normalized = (df - df.rolling(rolling_window).mean()) / df.rolling(rolling_window).std()
    return normalized

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), :].reshape(-1)
        dataX.append(a)
        dataY.append(dataset[i + look_back, -1])
    return np.array(dataX), np.array(dataY)

processed_list_pkl_filepath = os.path.join(parameters.output_base_dir, 'topn_50_rolling_3_period_x_y_dict_NOT_ALIGNED.pkl') 
[period_dict, proposed_data_x_dict, fred_data_y_dict] = load_pkl(processed_list_pkl_filepath)

# s1 = pd.Series(proposed_data_x_dict['frequency']['unemployment_not_adjusted'])
# s1 = rescale_rolling(s1, 3)
s2 = pd.Series(proposed_data_x_dict['strength']['unemployment_not_adjusted'])
s2 = rescale_rolling(s2, 3)
# s3 = pd.Series(proposed_data_x_dict['emerging_topic_score']['unemployment_not_adjusted'])
# s3 = rescale_rolling(s3, 3)

s4 = pd.Series(fred_data_y_dict['unemployment_not_adjusted'])
s4 = rescale_rolling(s2, 3)

dataset = pd.concat([s2, s4], axis=1)
values = dataset.values   # pd.Series -> numpy.ndarray

# specify columns to plot
feature_num = 2
groups = [i for i in range(0,feature_num)]   
i = 1
# plot each column
pyplot.figure()
for group in groups:
    pyplot.subplot(len(groups), 1, i)
    pyplot.plot(values[:, group])
    pyplot.title(dataset.columns[group], y=0.5, loc='right')
    i += 1
pyplot.show()

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# integer encode direction
encoder = LabelEncoder()
values[:,feature_num - 1] = encoder.fit_transform(values[:,feature_num - 1])
# ensure all data is float
values = values.astype('float32')
# normalize features
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled = scaler.fit_transform(values)
# frame as supervised learning
# reframed = series_to_supervised(scaled, 1, 1)
reframed = series_to_supervised(values, 1, 1)

# drop columns we don't want to predict
reframed.drop(reframed.columns[[2]], axis=1, inplace=True)

# split into train and test sets
values = reframed.values
n_train_num = 50
train = values[:n_train_num, :]
test = values[n_train_num:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(8, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=1, validation_data=(test_X, test_y), verbose=2, shuffle=False)
#     # plot history
#     pyplot.plot(history.history['loss'], label='train')
#     pyplot.plot(history.history['val_loss'], label='test')
#     pyplot.legend()
#     pyplot.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
