'''
Created on Mar 8, 2018

@author: abhinav.jhanwar
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Import `Sequential` from `keras.models`
from keras.models import Sequential
# Import `Dense` from `keras.layers`
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
# Import optimizers: Stochastic Gradient Descent
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import  Pipeline

# load data
white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')
red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')

#print(red.sample())
# get dataframe details
#print(white.info())
#print(white.describe())

# check presence of null values
#print(pd.isnull(red))

# combining datasets
# Add `type` column to `red` with value 1
red['type'] = 1
# Add `type` column to `white` with value 0
white['type'] = 0
# Append `white` to `red`
wines = red.append(white, ignore_index=True)

# plotting data
'''fig, ax = plt.subplots(1, 2)
ax[0].hist(red.alcohol, 10, facecolor='red', alpha=0.5, label="Red Wine")
ax[1].hist(white.alcohol, 10, facecolor='white', ec='black', lw=0.5, alpha=0.5, label="White Wine")

fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=1)
ax[0].set_ylim([0, 1000])
ax[0].set_xlabel("Alcohol in % Vol")
ax[0].set_ylabel("Frequency")
ax[1].set_xlabel("Alcohol in % Vol")
ax[1].set_ylabel("Frequency")
fig.suptitle("Distribution of Alcohol in % Vol")

plt.show()'''

'''fig, ax = plt.subplots(1, 2, figsize=(8, 4))

ax[0].scatter(red['quality'], red["sulphates"], color="red")
ax[1].scatter(white['quality'], white['sulphates'], color="white", edgecolors="black", lw=0.5)

ax[0].set_title("Red Wine")
ax[1].set_title("White Wine")
ax[0].set_xlabel("Quality")
ax[1].set_xlabel("Quality")
ax[0].set_ylabel("Sulphates")
ax[1].set_ylabel("Sulphates")
ax[0].set_xlim([0,10])
ax[1].set_xlim([0,10])
ax[0].set_ylim([0,2.5])
ax[1].set_ylim([0,2.5])
fig.subplots_adjust(wspace=0.5)
fig.suptitle("Wine Quality by Amount of Sulphates")

plt.show()'''

'''np.random.seed(570)

redlabels = np.unique(red['quality'])
whitelabels = np.unique(white['quality'])

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
redcolors = np.random.rand(6,4)
whitecolors = np.append(redcolors, np.random.rand(1,4), axis=0)

for i in range(len(redcolors)):
    redy = red['alcohol'][red.quality == redlabels[i]]
    redx = red['volatile acidity'][red.quality == redlabels[i]]
    ax[0].scatter(redx, redy, c=redcolors[i])
for i in range(len(whitecolors)):
    whitey = white['alcohol'][white.quality == whitelabels[i]]
    whitex = white['volatile acidity'][white.quality == whitelabels[i]]
    ax[1].scatter(whitex, whitey, c=whitecolors[i])
    
ax[0].set_title("Red Wine")
ax[1].set_title("White Wine")
ax[0].set_xlim([0,1.7])
ax[1].set_xlim([0,1.7])
ax[0].set_ylim([5,15.5])
ax[1].set_ylim([5,15.5])
ax[0].set_xlabel("Volatile Acidity")
ax[0].set_ylabel("Alcohol")
ax[1].set_xlabel("Volatile Acidity")
ax[1].set_ylabel("Alcohol") 
ax[0].legend(redlabels, loc='best', bbox_to_anchor=(1.3, 1))
ax[1].legend(whitelabels, loc='best', bbox_to_anchor=(1.3, 1))
#fig.suptitle("Alcohol - Volatile Acidity")
fig.subplots_adjust(top=0.85, wspace=0.7)

plt.show()'''

'''# checking correlation
corr = wines.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.show()'''


###########
########### PREDICTING TYPE OF WINE with classification of wine type

# Specify the data 
X=wines.ix[:,0:11]
# Specify the target labels and flatten the array 
y=np.ravel(wines.type)
# Split the data up in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# check if all the classes are in equal proportion
#print(wines.type.value_counts())

# process the data:  standardizing
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the constructor
model = Sequential()
# Add an input layer 
# arguemtns of dense: output shape, activation, input shape
# activation(define the output function) = [relu', 'tanh']
model.add(Dense(12, activation='relu', input_shape=(11,)))
# Add one hidden layer 
# after first layer no need to give input shape
model.add(Dense(8, activation='relu'))
# Add an output layer 
# activation = [regression: 'linear', binary classification: 'sigmoid', multiclass: 'softmax']
# threshold: >0 or <0
# sigmoid: 1/(1+e^-x), usually applied in output layer
# rectifier: max(x,0), usually applied in hidden layer: relu
# hyperbolic tangent tanh: (1-e^-2x)/(1+e^-2x)
model.add(Dense(1, activation='sigmoid'))

# Model output shape
#print(model.output_shape)
# Model summary
#print(model.summary())
# Model config
#print(model.get_config())
# List all weight tensors 
#print(model.get_weights())

# Compile and fit model
# Some of the most popular optimization algorithms used are
# the Stochastic Gradient Descent (SGD), ADAM and RMSprop.
# Depending on whichever algorithm you choose,
# you'll need to tune certain parameters, such as learning rate or momentum.
# The choice for a loss function depends on the task that you have at hand:
# for example, for a regression problem, you'll usually use the Mean Squared Error (MSE).
# As you see in this example,
# you used binary_crossentropy for the binary classification problem of determining whether a wine is  red or white.
# Lastly, with mulit-class classification, you'll make use of categorical_crossentropy.

# optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
# loss = [regression: 'mse', binary: 'binary_crossentropy', multi: 'categorical_crossentropy']
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# epochs = iterations, batch_size= how many samples will be passed to neural network in one time       
fit = model.fit(X_train, y_train, epochs=20, batch_size=20, verbose=1, validation_split=0.2)

# predict probabilities
y_pred_prob = model.predict_proba(X_test)
# predict actual values
y_pred = model.predict(X_test)
y_pred = [float(np.round(x)) for x in y_pred]
print(y_pred[:5], y_test[:5])

# Final evaluation of the model
loss, accuracy = model.evaluate(X_test, y_test,verbose=1)
print("Accuracy: ", accuracy*100, "%")

# plot training vs validation for overfiiting
plt.plot(fit.history['acc'])
plt.plot(fit.history['val_acc'])
plt.title('model_accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()




# Confusion matrix
#print(confusion_matrix(y_test, y_pred))
# Precision 
#print(precision_score(y_test, y_pred))
# Recall
#print(recall_score(y_test, y_pred))
# F1 score
#print(f1_score(y_test,y_pred))
# Cohen's kappa
#print(cohen_kappa_score(y_test, y_pred))



###############
############### PREDICTING QUALITY OF WINE using regression
###############
''''y = wines.quality
X = wines.drop('quality', axis=1)

X = StandardScaler().fit_transform(X)

# Modifying optimizers
# lr = learning rate
#rmsprop = RMSprop(lr = 0.0001)
#sgd=SGD(lr=0.1)

# evaluating model using kfold
seed = 7
np.random.seed(seed)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
mse, mae = 0, 0
for train, test in kfold.split(X, y):
    model = Sequential()
    model.add(Dense(64, input_dim=12, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    #model.compile(optimizer=rmsprop, loss='mse', metrics=['mae'])
    #model.compile(optimizer=sgd, loss='mse', metrics=['mae'])
    model.fit(X[train], y[train], epochs=10, verbose=1)
    mse_value, mae_value = model.evaluate(X[test], y[test], verbose=0)
    mse += mse_value
    mae += mae_value

print('mse:', mse/5)
print('mae:', mae/5)'''


#################
################# PREDICTING QUALITY OF WINE with multiclass classification model
#################
'''y = wines.quality
dummy_y = np_utils.to_categorical(y)
X = wines.drop('quality', axis=1)

def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=12, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# evaluate model with standardized dataset
pipeline = Pipeline([('standardize', StandardScaler()),
                     ('mlp', KerasClassifier(build_fn=baseline_model, epochs=10, batch_size=1, verbose=1))
                     ])

kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, dummy_y, cv=kfold)
print("\nAccuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))'''

