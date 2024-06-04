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


# Definizione del modello a 4 livelli per la previsione dei segnali
def modified_model(num_classes=43):
    # Inizializzazione del modello sequenziale
    model = Sequential()
    # Aggiunta del primo strato convoluzionale con 60 filtri di dimensione 5x5, attivazione ReLU e input di dimensione 32x32x3
    model.add(Conv2D(60,(5,5),input_shape=(32,32,3),activation='relu'))
    # Aggiunta del secondo strato convoluzionale con 60 filtri di dimensione 5x5 e attivazione ReLU
    model.add(Conv2D(60,(5,5),activation='relu'))
    # Aggiunta del primo strato di pooling per ridurre la dimensione spaziale dell'output
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Aggiunta del terzo strato convoluzionale con 30 filtri di dimensione 3x3 e attivazione ReLU
    model.add(Conv2D(30,(3,3),activation='relu'))
    # Aggiunta del quarto strato convoluzionale con 30 filtri di dimensione 3x3 e attivazione ReLU
    model.add(Conv2D(30,(3,3),activation='relu'))
    # Aggiunta del secondo strato di pooling
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Appiattimento dell'output per prepararlo per lo strato completamente connesso
    model.add(Flatten())
    # Aggiunta dello strato completamente connesso con 500 neuroni e attivazione ReLU
    model.add(Dense(500,activation='relu'))
    # Aggiunta di uno strato di dropout per prevenire l'overfitting
    model.add(Dropout(0.4))
    # Aggiunta dello strato di output con un numero di neuroni pari al numero di classi e attivazione softmax
    model.add(Dense(num_classes ,activation='softmax'))
    # Compilazione del modello con l'ottimizzatore Adam, la funzione di perdita cross-entropy categorica e la metrica di accuratezza
    model.compile(Adam(learning_rate = 0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

if __name__ == '__main__':
    model = modified_model()
    print(model)