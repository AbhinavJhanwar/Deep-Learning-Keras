# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 14:59:16 2018

@author: abhinav.jhanwar
"""

# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Installing keras saving module
# conda install h5py

# ctrl+I to check function details

################################
# Part 1 - Data Preprocessing
################################

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

##########################################
# Part 2 - Now let's make the ANN!
##########################################

# Importing the Keras libraries and packages
from keras.models import Sequential, load_model
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()
# weights: defines the weightage of each node/feature according to their impact on target value
# learning rate: defines the value by which weights of nodes will be changed while back propagation
# units = output_dimension: usually average of input features and output variable

# Adding the input layer and the first hidden layer
# units = 6 as input = 11, output is binary = 1, average = 11+1/2=6
# kernel_initializer: defines how weightage will be given to nodes, with 'uniform' we make sure it is least possible
# activation: defines the output function = [relu', 'tanh']
# kernel_initializer: defines the initial weights of nodes = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
# sigmoid activation function is used to enable probablity and when units is 1 then it works as logistic regression
# if output is more than 1 class then activation will go as 'softmax' which is nothing but sigmoid function just to handle multi classes
# softmax also converts the final output into 1 i.e. it take cares that probability of all categories sum up to 1
# activation = [regression: 'linear', binary classification: 'sigmoid', multiclass: 'softmax']
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
# Some of the most popular optimization algorithms used are
# the Stochastic Gradient Descent (SGD), ADAM and RMSprop.
# Depending on whichever algorithm you choose,
# you'll need to tune certain parameters, such as learning rate or momentum.
# The choice for a loss function depends on the task that you have at hand:
# for example, for a regression problem, you'll usually use the Mean Squared Error (MSE).
# As you see in this example,
# you used binary_crossentropy for the binary classification problem of determining whether a person will leave bank or not.
# Lastly, with mulit-class classification, you'll make use of categorical_crossentropy.
# optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
# loss = [regression: 'mean_squared_error', binary: 'binary_crossentropy', multi: 'categorical_crossentropy']
# crossentropy(works only for classification) is prefered over rms as it provides logarithmic values hence more accurate value to determine any error in prediction
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# saving the keras model with all the weights
if not os.path.exists("SavedModels/classifier"):
            os.makedirs("SavedModels/classifier")
classifier.save("SavedModels/classifier/model.h5")

# load model
classifier = load_model("SavedModels/classifier/model.h5")

# Fitting the ANN to the Training set & get history of training & validation in each epoch
# batch_size: defines the number of records after which weights will be modified or backpropagation takes place
# epochs: defines number of times back propagation will take place
history = classifier.fit(X_train, y_train, batch_size = 100, epochs = 100, validation_split=0.2).history

########################################################
# Part 3 - Making predictions and evaluating the model
########################################################

# Predicting the Test set results
y_pred = classifier.predict(X_test)
# y_pred has all the probabilities but we need True or False
# hence converting accordingly
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

loss, accuracy = classifier.evaluate(X_test, y_test,verbose=1)

#######################################################
# Part 4 -  Testing with new Data
#######################################################

newData = pd.DataFrame(data={
        'CreditScore': 600,
        'Geography': 'France',
        'Gender': 'Male',
        'Age': 40,
        'Tenure': 3,
        'Balance': 60000,
        'NumOfProducts': 2,
        'HasCrCard': 1,
        'IsActiveMember': 1,
        'EstimatedSalary': 50000}, index=[0])

newData['Geography'] = labelencoder_X_1.transform(newData['Geography'])
newData['Gender'] = labelencoder_X_2.transform(newData['Gender'])
newData = onehotencoder.transform(newData).toarray()
newData = newData[:, 1:]
newData = sc.transform(newData)
prediction = classifier.predict(newData)
prediction = (prediction>0.5)