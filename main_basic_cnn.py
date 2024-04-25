import os
import random
import warnings



warnings.filterwarnings("ignore")
from utils import train_test_split


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

model = Sequential()

# ** ADD YOUR CODE HERE **

#def the varables
filter_size = (3, 3)
filters = 32
input_size = (128, 128, 3)  
pool_size = (2, 2)

#adds the the convolution layer 
model.add(Conv2D(filters, filter_size, dilation_rate=pool_size, activation='relu', input_shape=( 128, 128, 3)))
#adds the the Max poling
model.add(MaxPooling2D(pool_size = (2, 2)))
#adds the the convolution layer
model.add(Conv2D(filters, filter_size, dilation_rate=pool_size, activation='relu', input_shape=( 128, 128, 3)))
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

model.summary()

model.compile(loss='binary_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])

