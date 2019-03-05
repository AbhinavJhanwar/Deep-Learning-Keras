# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 15:56:47 2018

@author: abhinav.jhanwar
"""

''' increased number of time_steps, increased number of nodes in each layer, increase number of LSTM layers'''

##############################################
# Part 1 - Data Preprocessing
##############################################

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.metrics import mean_squared_error

# Importing the training set
dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
# extract only open column as this is the column we will predict
# convert to numpy array as keras takes only numpy as input
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
# using normalisation
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
# normalisation is preferred in RNN when activation function is sigmoid in output layer
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
# 60 timestep: model will look for every 60 time interval and generate output
# X_train: each row of X_train will have 60 columns with values of training_set_scaled data 50 rows
# y_train: each row will have 1 column with next value in training_set_scaled that was added in X_train and
# this way this value will be the next value or 61st value of training_set_scaled and will be the value to be predicted
X_train = []
y_train = []
for i in range(120, 1258):
    X_train.append(training_set_scaled[i-120:i, 0])
    y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
# batch_size: no. of observations or to be trained at once - 1198: X_train.shape[0]
# time_step: 120: X_train.shape[1]
# no. of predictors: 1 
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#############################################
# Part 2 - Building the RNN
#############################################

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
# return_sequences: whether to add next LSTM layer or not
regressor.add(LSTM(units = 80, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 80, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 80, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 80, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 80, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 80))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
history = regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

################################################################
# Part 3 - Making the predictions and visualising the results
################################################################

# Getting the real stock price of 2017
dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
# first collect the whole data: this is required as to predict stock price on any date we require data of previous 60 days, 
# hence some data will be in train and some in test
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
# get 120 records before the first test data
inputs = dataset_total[len(dataset_total)-len(dataset_test)-120:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

# create test set in same format as X_train to predict the data
# every row of X_test will have 60 columns with previous data after which we have to peform prediction
# i.e. for first test set value we need last 60 train set rows and
# for second test set value we need last 59 train set + first test value and so on
X_test = []
for i in range(120, 120+len(dataset_test)):
    X_test.append(inputs[i-120:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# predict stock price
predicted_stock_price = regressor.predict(X_test)
# inverse transfrom to get original stock price
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# error in prediction
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))/np.mean(real_stock_price)
