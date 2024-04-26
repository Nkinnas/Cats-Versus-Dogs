import os
import random
import warnings

import keras


warnings.filterwarnings("ignore")
from utils import train_test_split


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator


from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img



model = Sequential()

# ** ADD YOUR CODE HERE **

#def the varables 
# 
# 
# (need to change these)
#
#
#
filterSize = (3, 3)
filters = 32
inputSize = (256, 256, 3)  
poolSize = (2, 2)

#adds the the convolution layer 
model.add(Conv2D(filters, filterSize, dilation_rate=poolSize, activation='relu', input_shape= inputSize))
#adds the the Max poling
model.add(MaxPooling2D(pool_size = (2, 2)))
#adds the the convolution layer
model.add(Conv2D(filters, filterSize, dilation_rate=poolSize, activation='relu', input_shape= inputSize))
#adds the the Max poling
model.add(MaxPooling2D(pool_size = (2,2)))

#turns the 4d thing and to a 2d tuple
model.add(Flatten())

# For instance, for a 2D input with shape (batch_size, input_dim), the output would have shape (batch_size, units)
#1 = units: Positive integer, dimensionality of the output space.
model.add(Dense( 128 ,activation='relu' ))

#The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting. 
#Inputs not set to 0 are scaled up by 1 / (1 - rate) 
#Float between 0 and 1. Fraction of the input units to drop
model.add(Dropout(0.5))

# For instance, for a 2D input with shape (batch_size, input_dim), the output would have shape (batch_size, units)
#1 = units: Positive integer, dimensionality of the output space.
model.add(Dense(1, activation='sigmoid'))

#__________________________________

#model.summary()

model.compile(loss='binary_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])

#Question 3




# Assuming your model is defined and compiled as before

#print(dir(keras.metrics))


model.compile(
    optimizer=keras.optimizers.Adam(epsilon=1e-07),
    loss=keras.losses.binary_crossentropy,
    metrics=[
        keras.metrics.binary_accuracy,
    ],
)

batchSize = 16
stepsPerEpoch = 345 // batchSize 
epochs = 2

testImgData = ImageDataGenerator(rescale=1./255)
trainImgData = ImageDataGenerator(rescale=1./255)

testDir = 'PetImages/Test/'
trainDir = 'PetImages/Train/'

trainGenerator = testImgData.flow_from_directory(
    testDir,
    batch_size= batchSize,
    class_mode='binary')

validation_generator = trainImgData.flow_from_directory(
    trainDir,
    batch_size= batchSize,
    class_mode='binary')
 
model.fit_generator(
    trainGenerator,
    steps_per_epoch=2000 // batchSize,
    epochs= epochs,
    validation_data=validation_generator,
    validation_steps= 800 // batchSize)

model.save_weights('first_try.h5')

