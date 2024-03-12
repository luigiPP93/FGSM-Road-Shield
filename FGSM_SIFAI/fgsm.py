import pickle
import pandas as pd
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import model as fgsm
import tensorflow as tf
import json
from sklearn.metrics import confusion_matrix
import numpy as np
import requests
from PIL import Image

def readData():
        #opening pickle files and creating variables for testing, training and validation data
    with open('FGSM_SIFAI/german-traffic-signs/train.p','rb') as f:    #rb means read binary format.
        train_data = pickle.load(f)                                    #f is pointer
    with open('FGSM_SIFAI/german-traffic-signs/test.p','rb') as f:
        test_data = pickle.load(f)
    with open('FGSM_SIFAI/german-traffic-signs/valid.p','rb') as f:
        valid_data = pickle.load(f)

    print(type(train_data))

    X_train, y_train = train_data['features'], train_data['labels']
    X_val, y_val = valid_data['features'], valid_data['labels']
    X_test, y_test = test_data['features'], test_data['labels']

    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)

    assert(X_train.shape[0] == y_train.shape[0]), 'The number of images is not equal to the number of labels'
    assert(X_val.shape[0] == y_val.shape[0]), 'The number of images is not equal to the number of labels'
    assert(X_test.shape[0] == y_test.shape[0]), 'The number of images is not equal to the number of labels'

    assert(X_train.shape[1:] == (32,32,3)), "The dimensions of the images are not 32x32x3"
    assert(X_val.shape[1:] == (32,32,3)), "The dimensions of the images are not 32x32x3"
    assert(X_test.shape[1:] == (32,32,3)), "The dimensions of the images are not 32x32x3"

    data = pd.read_csv('FGSM_SIFAI/german-traffic-signs/signnames.csv')
    print("dim",data.shape)
    print(data)
    
    return data, X_train, y_train,X_val,X_test,y_test,y_val
import random

def visualization_of_image(data, X_train, y_train):
    num_of_samples = []
    list_signs = []
    cols = 5
    num_classes = 43
    fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(10, 50))  # Aumenta le dimensioni della figura
    fig.tight_layout()

    for j in range(num_classes):
        x_selected = X_train[y_train == j]
        for i in range(cols):
            idx = random.randint(1, len(x_selected) - 1)  # Aggiusta l'indice casuale valido
            axs[j][i].imshow(x_selected[idx], cmap=plt.get_cmap("gray"))  # Usa directamente x_selected[idx]
            axs[j][i].axis("off")
            if i == 2:
                axs[j][i].set_title(str(j) + "-" + data.iloc[j]["SignName"])
                list_signs.append(data.iloc[j]["SignName"])
                num_of_samples.append(len(x_selected))
    plt.show()
    return num_of_samples, num_classes, list_signs

def datasetDistribution(num_of_samples,num_classes):
    print(num_of_samples)
    sum = 0
    for i in range(len(num_of_samples)):
        sum = sum + num_of_samples[i]
    print(sum)

    plt.figure(figsize=(12,4))
    plt.bar(range(0, num_classes), num_of_samples)
    plt.title("Distribiution of the training dataset")
    plt.xlabel("Class number")
    plt.ylabel("Number of images")
    plt.show()
    
#converting image into gray scale so that neural network can learn the pattern easily
def gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img
  #equalize histogram extract reigon of interest very correctly

def preprocess(img):
    #img = gray(img)
    #img = equalize(img)
    img = img/255 #normalizing of images
    return img

def view_image_processed(X_train):
    # Supponendo che X_train sia stato già definito come un array NumPy
    # con le immagini preprocessate

    # Stampa le prime 5 immagini utilizzando imshow
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 4))

    for i in range(5):
        axes[i].imshow(X_train[i])  # Mostra l'immagine
        axes[i].set_title('Immagine {}'.format(i+1))  # Imposta il titolo
        axes[i].axis('off')  # Nasconde gli assi

    plt.tight_layout()  # Aggiusta la disposizione delle immagini
    plt.show()  # Mostra il plot
    
def Reshape_mapped_and_preprocessed_images(X_train,X_test,X_val):
    #Reshape mapped and preprocessed images
    X_train = X_train.reshape(34799, 32, 32, 3)
    X_test = X_test.reshape(12630, 32, 32, 3)
    X_val = X_val.reshape(4410, 32, 32, 3)

    img_rows, img_cols, channels = 32, 32, 3

    #Display dataset shape
    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)
    
    return X_train,X_test,X_val

def manipulate_data(X_train, y_train):
    #Manipulate data within the batches for better model recognition
    datagen = ImageDataGenerator(width_shift_range=0.1,
                                height_shift_range=0.1,
                                zoom_range=0.2,
                                shear_range=0.1,
                                rotation_range=10.)
    datagen.fit(X_train)
    # for X_batch, y_batch in

    batches = datagen.flow(X_train, y_train, batch_size = 15)
    X_batch, y_batch = next(batches)

    fig, axs = plt.subplots(1, 15, figsize=(20, 5))
    fig.tight_layout()

    #Display batch of random 15 images
    for i in range(15):
        axs[i].imshow(X_batch[i].reshape(32, 32,3))
        axs[i].axis("off")
        plt.show

    print(X_batch.shape)
    return datagen

def save_model(model,history):
    model.save('FGSM_SIFAI/FGSM_modello.h5')
    with open('FGSM_SIFAI/history/history.json', 'w') as f:
        json.dump(history.history, f)

def load_the_model():
    model = load_model('FGSM_SIFAI/FGSM_modello.h5')

    # Carica la storia dell'addestramento
    with open('FGSM_SIFAI/history/history.json', 'r') as f:
        history = json.load(f)
        
    return model,history

def confusion_metrix(X_test,y_test):
    y_pred = model.predict(X_test)
    # Converti le previsioni in classi predette
    y_pred_classes = np.argmax(y_pred, axis=1)
    # Converti le etichette reali in classi reali
    y_true = np.argmax(y_test, axis=1)

    # Calcola la matrice di confusione
    confusion_mtx = confusion_matrix(y_true, y_pred_classes)

    # Ora puoi stampare la matrice di confusione
    print(confusion_mtx)
    
    return y_pred,y_pred_classes,y_true,confusion_mtx
    
def metriche(y_pred_classes,y_true):
    # Calcola l'accuratezza
    accuracy = accuracy_score(y_true, y_pred_classes)
    print("Accuracy:", accuracy)

    # Calcola la precision
    precision = precision_score(y_true, y_pred_classes, average='weighted')
    print("Precision:", precision)

    # Calcola il recall
    recall = recall_score(y_true, y_pred_classes, average='weighted')
    print("Recall:", recall)

    # Calcola l'F1-score
    f1 = f1_score(y_true, y_pred_classes, average='weighted')
    print("F1-score:", f1)
    
def plot_metrics(history):
    # Estrarre i dati dall'oggetto history
    loss = history['loss']
    accuracy = history['accuracy']
    val_loss = history['val_loss']
    val_accuracy = history['val_accuracy']
    
    plt.figure(figsize=(10, 5))
    plt.plot(accuracy, label='Training Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.legend(['accuracy', 'val_accuracy'])
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.show()
    
import cv2

def load_image_from_file(file_or_filepath):
    # Leggi l'immagine dal percorso del file in formato BGR
    img = cv2.imread(file_or_filepath)

    # Controlla se l'immagine è stata caricata correttamente
    if img is not None:
        # Converti l'immagine da BGR a RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Ridimensiona l'immagine alle dimensioni desiderate (ad esempio, 32x32)
        img = cv2.resize(img, (32, 32))
        
        # Esegui eventuali preelaborazioni sull'immagine (ad esempio, normalizzazione)
        # Assicurati che la funzione preprocess() sia definita altrove nel tuo codice
        img = preprocess(img)
        
        # Aggiungi una dimensione per la batch (se necessario)
        img = img.reshape(1, 32, 32, 3)
   
    else:
        print("Errore nel caricamento dell'immagine")

    return img


def predict(model,img):
    list_signs = [
    'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (60km/h)', 
    'Speed limit (70km/h)', 'Speed limit (80km/h)', 'End of speed limit (80km/h)', 'Speed limit (100km/h)', 
    'Speed limit (120km/h)', 'No passing', 'No passing for vechiles over 3.5 metric tons', 
    'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop', 'No vechiles', 
    'Vechiles over 3.5 metric tons prohibited', 'No entry', 'General caution', 'Dangerous curve to the left', 
    'Dangerous curve to the right', 'Double curve', 'Bumpy road', 'Slippery road', 
    'Road narrows on the right', 'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing', 
    'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing', 'End of all speed and passing limits', 
    'Turn right ahead', 'Turn left ahead', 'Ahead only', 'Go straight or right', 'Go straight or left', 
    'Keep right', 'Keep left', 'Roundabout mandatory', 'End of no passing', 
    'End of no passing by vechiles over 3.5 metric tons'
    ]
    prediction=np.argmax(model.predict(img), axis=-1)
    predizione = (prediction[0], (list_signs[prediction[0]]))
    return predizione

def adversarial_pattern(image, label,model):
    image = tf.cast(image, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        #print("Predictionnnn",prediction)
        loss = tf.keras.losses.MSE(label, prediction)
        #print("loss", loss)
        #print("lable",label)
        #print("prediction",prediction)
    gradient = tape.gradient(loss, image)
    print("gradient",gradient)
    signed_grad = tf.sign(gradient)
    return signed_grad

if __name__ == '__main__':
    import sys
    
    #print(sys.executable)
    data, X_train, y_train,X_val,X_test,y_test,y_val=readData()
    num_of_samples,num_classes,list_signs=visualization_of_image(data,X_train,y_train)
    print("lista segnaliiiiiii",list_signs)
    datasetDistribution(num_of_samples,num_classes)
    X_train = np.array(list(map(preprocess, X_train)))
    X_val = np.array(list(map(preprocess, X_val)))
    X_test = np.array(list(map(preprocess, X_test)))
    view_image_processed(X_train)
    X_train,X_test,X_val=Reshape_mapped_and_preprocessed_images(X_train,X_test,X_val)
    datagen=manipulate_data(X_train, y_train)
    #Categorise the images
    y_train = to_categorical(y_train, 43)
    y_test = to_categorical(y_test, 43)
    y_val = to_categorical(y_val, 43)
    
    #Train model
    model = fgsm.modified_model()
    #history = model.fit(datagen.flow(X_train,y_train, batch_size=50), steps_per_epoch = X_train.shape[0]/50, epochs = 10, validation_data= (X_val, y_val), shuffle = 1)
    #print(model.summary())
    #save_model(model,history)
    
    model,history=load_the_model()
    predictions = model.predict(X_test)

    # Valuta le prestazioni del modello sui nuovi dati, ad esempio calcolando l'accuratezza
    accuracy = model.evaluate(X_test, y_test)
    print('Accuracy:', accuracy)
    y_pred,y_pred_classes,y_true,confusion_mtx=confusion_metrix(X_test,y_test)
    metriche(y_pred_classes,y_true)
    plot_metrics(history)
    img=load_image_from_file('FGSM_SIFAI\Screenshot 2024-03-02 103033.jpg')
    print(predict(model,img))
    
    img_rows, img_cols, channels = 32, 32, 3
    image = X_test[3000]
    plt.clf()
    plt.axis("off")
    plt.imshow(image.reshape((img_rows, img_cols, channels)))
    plt.savefig('FGSM_SIFAI\static\img', bbox_inches='tight', pad_inches=0)
    
    image_label = y_test[3000]
    print("image_lable",image_label)
    #print("image_lable",image_label.shape())
    perturbations = adversarial_pattern(image.reshape((1, img_rows, img_cols, channels)), image_label,model).numpy()
    adversarial = image + perturbations * 0.1

    print("Model prediction == ",list_signs[model.predict(image.reshape((1, img_rows, img_cols, channels))).argmax()])
    print("Prediction with Intrusion== ", list_signs[model.predict(adversarial).argmax()])

    if channels == 1:
        plt.imshow(adversarial.reshape((img_rows, img_cols)))
    else:
        plt.imshow(adversarial.reshape((img_rows, img_cols, channels)))

    plt.show()
    
    
    
    
    
    