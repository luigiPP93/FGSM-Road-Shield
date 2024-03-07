import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import pickle
import pandas as pd
import random
import cv2
np.random.seed(0)


#Define the 4 layers model for sign prediction
def modified_model(num_classes=43):
    model = Sequential()
    model.add(Conv2D(60,(5,5),input_shape=(32,32,1),activation='relu'))
    model.add(Conv2D(60,(5,5),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))


    model.add(Conv2D(30,(3,3),activation='relu'))
    model.add(Conv2D(30,(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
#     model.add(Dropout(0.4))


    model.add(Flatten())
    model.add(Dense(500,activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes ,activation='softmax'))
    #Compile model
    model.compile(Adam(learning_rate = 0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

if __name__ == '__main__':
    model = modified_model()
    print(model)