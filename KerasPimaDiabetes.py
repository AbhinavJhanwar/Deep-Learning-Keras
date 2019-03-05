'''
Created on Mar 8, 2018

@author: abhinav.jhanwar
'''
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.optimizers import Nadam

# Function to create model, required for KerasClassifier
def create_model():
    # create model
    init_mode = 'lecun_uniform'
    optimizer = Nadam(lr=0.01)
    activation = 'softplus'
    neurons = 10
    
    model = Sequential()
    model.add(Dense(neurons, input_dim=8, kernel_initializer=init_mode, activation=activation))
    model.add(Dense(8, kernel_initializer=init_mode, activation=activation))
    model.add(Dense(1, kernel_initializer=init_mode, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# load pima indians dataset
dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# type1
# create model
model = Sequential()
model.add(Dense(10, input_dim=8, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=150, batch_size=10, verbose=1)
# evaluate the model
loss, accuracy = model.evaluate(X, Y)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))

# calculate predictions
probabilities = model.predict(X)

# round predictions
y_pred = [float(np.round(x)) for x in probabilities]
print(y_pred[:5], Y[:5])

# type2
# define the grid search parameters
''' # get best batch_size, epochs
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
grid_result = grid.fit(X, Y)'''

'''# get best optimizer
model = KerasClassifier(build_fn=create_model, verbose=0, epochs=100, batch_size=20)
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
param_grid = dict(optimizer=optimizer)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
grid_result = grid.fit(X, Y)'''

'''# get best learn_rate
model = KerasClassifier(build_fn=create_model, verbose=0, epochs=100, batch_size=20)
learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
param_grid = dict(learn_rate=learn_rate)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
grid_result = grid.fit(X, Y)'''

'''# get best weight initialization
model = KerasClassifier(build_fn=create_model, verbose=0, epochs=100, batch_size=20)
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
param_grid = dict(init_mode=init_mode)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
grid_result = grid.fit(X, Y)'''

'''# get best activation
model = KerasClassifier(build_fn=create_model, verbose=0, epochs=100, batch_size=20)
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
param_grid = dict(activation=activation)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
grid_result = grid.fit(X, Y)'''

'''# get best no. of neurons
model = KerasClassifier(build_fn=create_model, verbose=0, epochs=100, batch_size=20)
neurons = [1, 5, 10, 15, 20, 25, 30]
param_grid = dict(neurons=neurons)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
grid_result = grid.fit(X, Y)'''

'''# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))'''

'''# create model
init_mode = 'lecun_uniform'
optimizer = Nadam(lr=0.01)
activation = 'softplus'
neurons = 10
    
model = Sequential()
model.add(Dense(neurons, input_dim=8, kernel_initializer=init_mode, activation=activation))
model.add(Dense(8, kernel_initializer=init_mode, activation=activation))
model.add(Dense(1, kernel_initializer=init_mode, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Fit the model
model.fit(X, Y, verbose=1)
# evaluate the model
loss, accuracy = model.evaluate(X, Y)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))

# calculate predictions
probabilities = model.predict(X)

# round predictions
y_pred = [float(np.round(x)) for x in probabilities]
print(y_pred[:5], Y[:5])'''