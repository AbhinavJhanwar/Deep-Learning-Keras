# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 12:03:26 2018

@author: abhinav.jhanwar
"""

# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

##########################################
# Part 1 - Building the CNN
##########################################

# Importing the Keras libraries and packages
from keras import Sequential
# 2D to deal with images, 3D to deal with videos(x,y,time)
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
# apply various filters to get various layers of image
# 32: number of convolution layers or 2d matrices
# 3,3: size of filter
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

# Step 2 - Pooling
# Max Pooling - get 2x2 matrix to filter image by max value
# usually take size 2x2 so that much information is not lost
# in this way assume we have n number of 2d matrix then each matrix represent one specific feature of image
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
# convert into a single dimensional array where each element contains a feature of image
classifier.add(Flatten())

# Step 4 - Full connection
# fully connected layer or hidden layer
classifier.add(Dense(units = 128, activation = 'relu'))
# create output layer
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

###############################################
# Part 2 - Fitting the CNN to the images
###############################################
# perform image augmentation to save model from overfitting the training data
# overfitting may occur due to lack of data here so augmentation helps in augmenting the available data
# augmentation will create various batches of images & will apply various transformations on images like flipping
# hence providing the overall more training data without adding anymore images

from keras.preprocessing.image import ImageDataGenerator

# shear_range & zoom_range are to define how much randomization we want to apply on data
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

# target size: same as input shape in classifier as this will be the input to classifier
# batch_size: number of inputs after which weights will be updated
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size = (64, 64),
                                                batch_size = 32,
                                                class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# steps_per_epoch: number of images in training set 
history = classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)

import os
if not os.path.exists("SavedModels/classifier"):
            os.makedirs("SavedModels/classifier")
classifier.save("SavedModels/classifier/model.h5")

#############################################
# Part 3 - Making new prediction
#############################################
from keras.models import load_model
classifier = load_model("SavedModels/classifier/model.h5")
import numpy as np
from keras.preprocessing import image
# load image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size=(64, 64))
# convert into array
# dimenstion = 64, 64, 3
test_image = image.img_to_array(test_image)
# add batch size as 4th dimension
test_image = np.expand_dims(test_image, axis = 0)
# predict
result = classifier.predict(test_image)
# get class attributes
training_set.class_indices

if result[0][0]==1:
    prediction = 'dog'
else:
    prediction = 'cat'

