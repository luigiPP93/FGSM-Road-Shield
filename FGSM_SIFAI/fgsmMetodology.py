import pickle
import pandas as pd
import matplotlib
import seaborn as sns
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import model as fgsm
import model_intrusion
import tensorflow as tf
import json
from sklearn.metrics import confusion_matrix
import numpy as np
import requests
from PIL import Image
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from codecarbon import OfflineEmissionsTracker 
import os

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

def datasetDistribution(file_path, num_classes, class_names):
    # Carica il file pickle
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # Estrarre le etichette delle classi
    labels = data['labels']
    
    # Contare il numero di immagini per ciascuna classe
    train_number = [0] * num_classes
    for label in labels:
        train_number[label] += 1

    # Generare i nomi delle classi
    class_num = [class_names[i] for i in range(num_classes)]

    print(train_number)
    total_images = sum(train_number)
    print(total_images)

    # Plotting the number of images in each class
    plt.figure(figsize=(15,8))
    plt.bar(class_num, train_number)
    
    # Regolare le etichette
    plt.xticks(rotation=90, ha='right', fontsize=12)  # Inclinare le etichette e aumentare la dimensione del font
    plt.yticks(fontsize=12)  # Aumentare la dimensione del font per l'asse y
    
    plt.title("Distribution of the training dataset", fontsize=16)
    plt.xlabel("Class", fontsize=14)
    plt.ylabel("Number of images", fontsize=14)
    
    plt.tight_layout()  # Migliora il layout per evitare che le etichette siano tagliate
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
        
def save_model2(model,history):
    model.save('FGSM_SIFAI/FGSM_modello_intrusion.h5')
    with open('FGSM_SIFAI/history/history_intrusion.json', 'w') as f:
        json.dump(history.history, f)

def load_the_model():
    model = load_model('FGSM_SIFAI/FGSM_modello.h5')

    # Carica la storia dell'addestramento
    with open('FGSM_SIFAI/history/history.json', 'r') as f:
        history = json.load(f)
        
    return model,history

def load_the_model2():
    model = load_model('FGSM_SIFAI/FGSM_modello_intrusion.h5')

    # Carica la storia dell'addestramento
    with open('FGSM_SIFAI/history/history_intrusion.json', 'r') as f:
        history = json.load(f)
        
    return model,history

def confusion_metrix(X_test, y_test, model):
    # Effettua le previsioni sul set di test
    y_pred = model.predict(X_test)
    
    # Converti le previsioni e le etichette reali da one-hot a classi intere
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Calcola la matrice di confusione
    confusion_mtx = confusion_matrix(y_true, y_pred_classes)

    # Ora puoi stampare la matrice di confusione
    print(confusion_mtx)
    
    plt.figure(figsize=(15,10))  # Aumenta le dimensioni del grafico
    sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 10})  # Riduci la dimensione del testo
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()
    
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
    #print("gradient",gradient)
    signed_grad = tf.sign(gradient)
    return signed_grad

def generate_adversarials(model, y_train, X_train):
    while True:
        x = []
        y = []
        for image, label in zip(X_train, y_train):
            perturbations = adversarial_pattern(image.reshape((1, img_rows, img_cols, channels)), label, model).numpy()
            epsilon = 0.4
            adversarial = image + perturbations * epsilon
            x.append(adversarial)
            y.append(label)
        x = np.asarray(x).reshape((len(X_train), img_rows, img_cols, channels))
        y = np.asarray(y)

        yield x, y

def TestModel(X_train, y_train,model,defence_model,list_signs):
    # Create test sample with 10 only attacked image and predict right label
    test_set = datagen.flow(X_train, y_train, batch_size = 10)
    X_test_set, y_test_set = next(test_set)
    d2 = 0
    nd2 = 0
    for X_test_set, y_test_set in zip(X_test_set, y_test_set):
            perturbations = adversarial_pattern(X_test_set.reshape((1, img_rows, img_cols, channels)), y_test_set,model).numpy()
            adversarial = X_test_set + perturbations * 0.4
            Model_Prediction = list_signs[model.predict(adversarial.reshape((1, img_rows, img_cols, channels))).argmax()]
            Truth_label = list_signs[y_test_set.argmax()]
            print('Model Prediction:', Model_Prediction,",", 'Truth label:', Truth_label)
            if channels == 1:
                plt.imshow(X_test_set.reshape(img_rows, img_cols))
            else:
                plt.imshow(X_test_set)
            plt.show()

            if Model_Prediction != Truth_label:
                Defence_Model = list_signs[defence_model.predict(adversarial.reshape((1, img_rows, img_cols, channels))).argmax()]
                print("Image was attacked")
                if Defence_Model == Truth_label:
                    print("Defence Model prediction:", Defence_Model)
                    d2=d2+1
                else:
                    print("Can not detect correctly")
                    nd2=nd2+1
    print("Number of correct defence predictions",d2)
    print("Number of not detected predictions",nd2)


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with respect to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

from tensorflow.keras.preprocessing import image
import matplotlib.cm as cm  # Add this import

def display_gradcam(img_path, heatmap, alpha=0.3):
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    # Display Heatmap
    plt.figure(figsize=(5, 5))
    plt.imshow(superimposed_img)
    plt.axis("off")
    plt.show()
    
if __name__ == '__main__':
    import sys
    classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)',
            2:'Speed limit (50km/h)',
            3:'Speed limit (60km/h)',
            4:'Speed limit (70km/h)',
            5:'Speed limit (80km/h)',
            6:'End of speed limit (80km/h)',
            7:'Speed limit (100km/h)',
            8:'Speed limit (120km/h)',
            9:'No passing',
            10:'No passing veh over 3.5 tons',
            11:'Right-of-way at intersection',
            12:'Priority road',
            13:'Yield',
            14:'Stop',
            15:'No vehicles',
            16:'Veh > 3.5 tons prohibited',
            17:'No entry',
            18:'General caution',
            19:'Dangerous curve left',
            20:'Dangerous curve right',
            21:'Double curve',
            22:'Bumpy road',
            23:'Slippery road',
            24:'Road narrows on the right',
            25:'Road work',
            26:'Traffic signals',
            27:'Pedestrians',
            28:'Children crossing',
            29:'Bicycles crossing',
            30:'Beware of ice/snow',
            31:'Wild animals crossing',
            32:'End speed + passing limits',
            33:'Turn right ahead',
            34:'Turn left ahead',
            35:'Ahead only',
            36:'Go straight or right',
            37:'Go straight or left',
            38:'Keep right',
            39:'Keep left',
            40:'Roundabout mandatory',
            41:'End of no passing',
            42:'End no passing veh > 3.5 tons' }
    
    #print(sys.executable)
    data, X_train, y_train,X_val,X_test,y_test,y_val=readData()
    num_of_samples,num_classes,list_signs=visualization_of_image(data,X_train,y_train)
    print("lista segnaliiiiiii",list_signs)
    datasetDistribution("C:/Users/luigi/OneDrive/Documenti/GitHub/Image-Perturbation-for-Visual-Security-Enhancement/FGSM_SIFAI/german-traffic-signs/train.p",num_classes,classes)
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
    tracker = OfflineEmissionsTracker(country_iso_code="ITA") # the tracker object will compute the
                                                         # power consumption of our machine's components
                                                         # when executing a portion of the code.


    #tracker.start() # we tell CodeCarbon to start tracking
    model = fgsm.modified_model()
    #history = model.fit(datagen.flow(X_train,y_train, batch_size=50), steps_per_epoch = X_train.shape[0]/50, epochs = 10, validation_data= (X_val, y_val), shuffle = 1)
    #print(model.summary())
    #save_model(model,history)
    
    model,history=load_the_model()
    predictions = model.predict(X_test)
    path = 'FGSM_SIFAI\Screenshot 2024-03-02 103033.jpg'
    image = cv2.imread(path)
    resized = cv2.resize(image, (32, 32))
    x = resized / 255.0

    # Apply Gaussian blur
    x = cv2.GaussianBlur(x, (5, 5), 0)
    x = np.expand_dims(x, axis=0)
    # Ottieni l'ultimo layer di convoluzione nel modello
    last_conv_layer_name = [layer.name for layer in model.layers if 'conv' in layer.name][-1]
    heatmap = make_gradcam_heatmap(x, model, last_conv_layer_name)  # last_conv_layer_name

    # Visualize it
    display_gradcam(path, heatmap)
    #tracker.stop() # lastly, we stop the tracker

    #emissions_csv = pd.read_csv("emissions.csv")
    #emissions_csv.columns

    #last_emissions = emissions_csv.tail(1) # we get the tail (the last computed values)

    #emissions = last_emissions["emissions"] * 1000
    #energy = last_emissions["energy_consumed"]
    #cpu = last_emissions["cpu_energy"]
    #gpu = last_emissions["gpu_energy"]
    #ram = last_emissions["ram_energy"]

    #print(f"{emissions} Grams of CO2-equivalents")
    #print(f"{energy} Sum of cpu_energy, gpu_energy and ram_energy (kWh)")
    #print(f"{cpu} Energy used per CPU (kWh)")
    #print(f"{gpu} Energy used per GPU (kWh)")
    #print(f"{ram} Energy used per RAM (kWh)")

    # Valuta le prestazioni del modello sui nuovi dati, ad esempio calcolando l'accuratezza
    accuracy = model.evaluate(X_test, y_test)
    print('Accuracy:', accuracy)
    y_pred,y_pred_classes,y_true,confusion_mtx=confusion_metrix(X_test,y_test,model)
    metriche(y_pred_classes,y_true)
   
    plot_metrics(history)
    img=load_image_from_file('FGSM_SIFAI\Screenshot 2024-03-02 103033.jpg')
    print(predict(model,img))
    
    img_rows, img_cols, channels = 32, 32, 3
    image = X_test[6000]
    plt.clf()
    plt.axis("off")
    plt.imshow(image.reshape((img_rows, img_cols, channels)))
    plt.savefig('FGSM_SIFAI\static\img3', bbox_inches='tight', pad_inches=0)
    
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
    
    #x_adversarial_train, y_adversarial_train = next(generate_adversarials(model,y_train,X_train))
    #x_adversarial_val, y_adversarial_val = next(generate_adversarials(model,y_val,X_val))
    #x_adversarial_test, y_adversarial_test = next(generate_adversarials(model,y_test,X_test))
    
    # Salvataggio dei dati su disco
    #np.save('FGSM_SIFAI/dati_modificati/x_adversarial_train.npy', x_adversarial_train)
    #np.save('FGSM_SIFAI/dati_modificati/y_adversarial_train.npy', y_adversarial_train)
    #np.save('FGSM_SIFAI/dati_modificati/x_adversarial_val.npy', x_adversarial_val)
    #np.save('FGSM_SIFAI/dati_modificati/y_adversarial_val.npy', y_adversarial_val)
    #np.save('FGSM_SIFAI/dati_modificati/x_adversarial_test.npy', x_adversarial_test)
    #np.save('FGSM_SIFAI/dati_modificati/y_adversarial_test.npy', y_adversarial_test)

    x_adversarial_train = np.load('FGSM_SIFAI/dati_modificati/x_adversarial_train.npy')
    y_adversarial_train = np.load('FGSM_SIFAI/dati_modificati/y_adversarial_train.npy')
    
    x_adversarial_val = np.load('FGSM_SIFAI/dati_modificati/x_adversarial_val.npy')
    y_adversarial_val = np.load('FGSM_SIFAI/dati_modificati/y_adversarial_val.npy')
    
    x_adversarial_test = np.load('FGSM_SIFAI/dati_modificati/x_adversarial_test.npy')
    y_adversarial_test = np.load('FGSM_SIFAI/dati_modificati/y_adversarial_test.npy')
    
    import numpy as np

    # Unione dei dati adversariali con i dati originali
    x_combined_train = np.concatenate((X_train, x_adversarial_train), axis=0)
    y_combined_train = np.concatenate((y_train, y_adversarial_train), axis=0)
    
    x_combined_val = np.concatenate((X_val, x_adversarial_val), axis=0)
    y_combined_val = np.concatenate((y_val, y_adversarial_val), axis=0)
    
    x_combined_test = np.concatenate((X_test, x_adversarial_test), axis=0)
    y_combined_test = np.concatenate((y_test, y_adversarial_test), axis=0)
    # Creazione del modello
    
    defence_model=model_intrusion.create_robust_model()
    # Addestramento del modello con dati combinati
    #combined_history = defence_model.fit(datagen.flow(x_combined_train, y_combined_train, batch_size=50), steps_per_epoch=x_combined_train.shape[0]/50, epochs=10, validation_data=(x_combined_val, y_combined_val), shuffle=1)
    #save_model2(defence_model,combined_history)
    defence_model,combined_history=load_the_model2()
    plot_metrics(combined_history)
    accuracy = defence_model.evaluate(x_combined_test,y_combined_test )
    
    print('Accuracy:', accuracy)
    y_pred,y_pred_classes,y_true,confusion_mtx=confusion_metrix(x_combined_test,y_combined_test,defence_model)
    metriche(y_pred_classes,y_true)
    # Taking example of 20km/h before and after creating new defence model
    image = X_train[1000]
    image_label = y_train[1000]
    perturbations = adversarial_pattern(image.reshape((1, img_rows, img_cols, channels)), image_label,model).numpy()
    adversarial = image + perturbations * 0.4

    print("Model Prediction on original image = ",list_signs[model.predict(image.reshape((1, img_rows, img_cols, channels))).argmax()])
    print("Defence Model Prediction on intrusion image = ", list_signs[defence_model.predict(adversarial).argmax()])

    if channels == 1:
        plt.imshow(adversarial.reshape((img_rows, img_cols)))
    else:
        plt.imshow(adversarial.reshape((img_rows, img_cols, channels)))

    plt.show()
    
    TestModel(X_train, y_train,model,defence_model,list_signs)

    # Carica il modello VGG16 pre-addestrato
    #base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, channels))

    # Congela tutti gli strati del modello base
    #for layer in base_model.layers:
    #    layer.trainable = False

    # Aggiungi nuovi livelli di output
    #x = base_model.output
    #x = Flatten()(x)
    #x = Dense(256, activation='relu')(x)
    #predictions = Dense(43, activation='softmax')(x)

    # Modello finale
    #modelPreso = Model(inputs=base_model.input, outputs=predictions)

    # Compilazione del modello
    #modelPreso.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Addestramento del modello
    #modelPreso.fit(x_combined_train, y_combined_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
    #accuracy = modelPreso.evaluate(X_test, y_test)
    
    #Performace modello base con quello non base
    accuracy = model.evaluate(x_adversarial_test, y_test)
    print('Accuracy:', accuracy)
    y_pred,y_pred_classes,y_true,confusion_mtx=confusion_metrix(x_adversarial_test,y_test,model)
    metriche(y_pred_classes,y_true)
    
    