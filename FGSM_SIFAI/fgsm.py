import pickle
import pandas as pd
import matplotlib
#matplotlib.use('Agg')
import seaborn as sns
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
    """
    Reads the training, testing, and validation data from pickle files.

    Returns:
    - data: DataFrame containing the class labels and names.
    - X_train: NumPy array of training images.
    - y_train: NumPy array of training labels.
    - X_val: NumPy array of validation images.
    - X_test: NumPy array of testing images.
    - y_test: NumPy array of testing labels.
    - y_val: NumPy array of validation labels.
    """
    # Opening pickle files and creating variables for testing, training, and validation data
    with open('FGSM_SIFAI/german-traffic-signs/train.p','rb') as f:
        train_data = pickle.load(f)
    with open('FGSM_SIFAI/german-traffic-signs/test.p','rb') as f:
        test_data = pickle.load(f)
    with open('FGSM_SIFAI/german-traffic-signs/valid.p','rb') as f:
        valid_data = pickle.load(f)

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

    return data, X_train, y_train, X_val, X_test, y_test, y_val

def visualization_of_image(data, X_train, y_train):
    """
    Visualizes a random sample of images from each class.

    Args:
    - data: DataFrame containing the class labels and names.
    - X_train: NumPy array of training images.
    - y_train: NumPy array of training labels.

    Returns:
    - num_of_samples: List containing the number of samples for each class.
    - num_classes: Total number of classes.
    - list_signs: List containing the names of each class.
    """
    num_of_samples = []
    list_signs = []
    cols = 5
    num_classes = 43
    fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(10, 50))
    fig.tight_layout()

    for j in range(num_classes):
        x_selected = X_train[y_train == j]
        for i in range(cols):
            idx = random.randint(1, len(x_selected) - 1)
            axs[j][i].imshow(x_selected[idx], cmap=plt.get_cmap("gray"))
            axs[j][i].axis("off")
            if i == 2:
                axs[j][i].set_title(str(j) + "-" + data.iloc[j]["SignName"])
                list_signs.append(data.iloc[j]["SignName"])
                num_of_samples.append(len(x_selected))
    plt.show()
    return num_of_samples, num_classes, list_signs

def datasetDistribution(file_path, num_classes, class_names):
    """
    Displays the distribution of images in each class.

    Args:
    - file_path: Path to the pickle file containing the dataset.
    - num_classes: Total number of classes.
    - class_names: List of class names.

    Returns: None
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    labels = data['labels']

    train_number = [0] * num_classes
    for label in labels:
        train_number[label] += 1

    class_num = [class_names[i] for i in range(num_classes)]

    total_images = sum(train_number)

    plt.figure(figsize=(15,8))
    plt.bar(class_num, train_number)

    plt.xticks(rotation=90, ha='right', fontsize=12)
    plt.yticks(fontsize=12)

    plt.title("Distribution of the training dataset", fontsize=16)
    plt.xlabel("Class", fontsize=14)
    plt.ylabel("Number of images", fontsize=14)

    plt.tight_layout()
    plt.show()

def gray(img):
    """
    Converts an image to grayscale.

    Args:
    - img: Input image.

    Returns:
    - img: Grayscale image.
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    """
    Applies histogram equalization to an image.

    Args:
    - img: Input image.

    Returns:
    - img: Image after histogram equalization.
    """
    img = cv2.equalizeHist(img)
    return img

def preprocess(img):
    """
    Preprocesses an image by converting it to grayscale and normalizing the pixel values.

    Args:
    - img: Input image.

    Returns:
    - img: Preprocessed image.
    """
    img = img/255
    return img

def view_image_processed(X_train):
    """
    Displays the first 5 preprocessed images from the training set.

    Args:
    - X_train: NumPy array of training images.

    Returns: None
    """
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 4))

    for i in range(5):
        axes[i].imshow(X_train[i])
        axes[i].set_title('Image {}'.format(i+1))
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

def Reshape_mapped_and_preprocessed_images(X_train, X_test, X_val):
    """
    Reshapes the mapped and preprocessed images.

    Args:
    - X_train: NumPy array of training images.
    - X_test: NumPy array of testing images.
    - X_val: NumPy array of validation images.

    Returns:
    - X_train: Reshaped training images.
    - X_test: Reshaped testing images.
    - X_val: Reshaped validation images.
    """
    X_train = X_train.reshape(34799, 32, 32, 3)
    X_test = X_test.reshape(12630, 32, 32, 3)
    X_val = X_val.reshape(4410, 32, 32, 3)

    img_rows, img_cols, channels = 32, 32, 3

    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)

    return X_train, X_test, X_val

def manipulate_data(X_train, y_train):
    """
    Manipulates the data within the batches for better model recognition.

    Args:
    - X_train: NumPy array of training images.
    - y_train: NumPy array of training labels.

    Returns:
    - datagen: ImageDataGenerator object for data manipulation.
    """
    datagen = ImageDataGenerator(width_shift_range=0.1,
                                height_shift_range=0.1,
                                zoom_range=0.2,
                                shear_range=0.1,
                                rotation_range=10.)
    datagen.fit(X_train)

    batches = datagen.flow(X_train, y_train, batch_size=15)
    X_batch, y_batch = next(batches)

    fig, axs = plt.subplots(1, 15, figsize=(20, 5))
    fig.tight_layout()

    for i in range(15):
        axs[i].imshow(X_batch[i].reshape(32, 32, 3))
        axs[i].axis("off")
        plt.show

    print(X_batch.shape)
    return datagen

def save_model(model, history):
    """
    Saves the trained model and its training history.

    Args:
    - model: Trained model.
    - history: Training history.

    Returns: None
    """
    model.save('FGSM_SIFAI/FGSM_modello.h5')
    with open('FGSM_SIFAI/history/history.json', 'w') as f:
        json.dump(history.history, f)

def save_model2(model, history):
    """
    Saves the trained intrusion model and its training history.

    Args:
    - model: Trained intrusion model.
    - history: Training history.

    Returns: None
    """
    model.save('FGSM_SIFAI/FGSM_modello_intrusion.h5')
    with open('FGSM_SIFAI/history/history_intrusion.json', 'w') as f:
        json.dump(history.history, f)

def load_the_model():
    """
    Loads the trained model and its training history.

    Returns:
    - model: Loaded model.
    - history: Loaded training history.
    """
    model = load_model('FGSM_SIFAI/FGSM_modello.h5')

    with open('FGSM_SIFAI/history/history.json', 'r') as f:
        history = json.load(f)

    return model, history

def load_the_model2():
    """
    Loads the trained intrusion model and its training history.

    Returns:
    - model: Loaded intrusion model.
    - history: Loaded training history.
    """
    model = load_model('FGSM_SIFAI/FGSM_modello_intrusion.h5')

    with open('FGSM_SIFAI/history/history_intrusion.json', 'r') as f:
        history = json.load(f)

    return model, history

def confusion_metrix(X_test, y_test, model):
    """
    Computes and displays the confusion matrix.

    Args:
    - X_test: NumPy array of testing images.
    - y_test: NumPy array of testing labels.
    - model: Trained model.

    Returns:
    - y_pred: Predicted labels.
    - y_pred_classes: Predicted classes.
    - y_true: True labels.
    - confusion_mtx: Confusion matrix.
    """
    y_pred = model.predict(X_test)

    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    confusion_mtx = confusion_matrix(y_true, y_pred_classes)

    print(confusion_mtx)

    plt.figure(figsize=(15,10))
    sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 10})
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()

    return y_pred, y_pred_classes, y_true, confusion_mtx

def metriche(y_pred_classes, y_true):
    """
    Computes and displays the accuracy, precision, recall, and F1-score.

    Args:
    - y_pred_classes: Predicted classes.
    - y_true: True labels.

    Returns: None
    """
    accuracy = accuracy_score(y_true, y_pred_classes)
    print("Accuracy:", accuracy)

    precision = precision_score(y_true, y_pred_classes, average='weighted')
    print("Precision:", precision)

    recall = recall_score(y_true, y_pred_classes, average='weighted')
    print("Recall:", recall)

    f1 = f1_score(y_true, y_pred_classes, average='weighted')
    print("F1-score:", f1)

def plot_metrics(history):
    """
    Plots the training and validation accuracy over epochs.

    Args:
    - history: Training history.

    Returns: None
    """
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

def main():
    """
    Main function to execute the code.
    """
    data, X_train, y_train, X_val, X_test, y_test, y_val = readData()
    num_of_samples, num_classes, list_signs = visualization_of_image(data, X_train, y_train)
    datasetDistribution('FGSM_SIFAI/german-traffic-signs/train.p', num_classes, list_signs)
    X_train = preprocess(X_train)
    view_image_processed(X_train)
    X_train, X_test, X_val = Reshape_mapped_and_preprocessed_images(X_train, X_test, X_val)
    datagen = manipulate_data(X_train, y_train)
    save_model(model, history)
    save_model2(model, history)
    model, history = load_the_model()
    model, history = load_the_model2()
    y_pred, y_pred_classes, y_true, confusion_mtx = confusion_metrix(X_test, y_test, model)
    metriche(y_pred_classes, y_true)
    plot_metrics(history)

if __name__ == "__main__":
    main()
    plt.plot(history['val_accuracy'])
    plt.legend(['accuracy', 'val_accuracy'])
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.show()
    
import cv2

def load_image_from_file(file_or_filepath):
    """
    Load an image from a file and perform necessary preprocessing.

    Args:
        file_or_filepath (str): The path to the image file.

    Returns:
        numpy.ndarray: The preprocessed image as a NumPy array.

    Raises:
        None

    """
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


def predict(model, img):
    """
    Predicts the class label for an input image using a given model.

    Args:
        model (keras.Model): The trained model used for prediction.
        img (numpy.ndarray): The input image to be classified.

    Returns:
        tuple: A tuple containing the predicted class label index and the corresponding class name.
    """
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
    prediction = np.argmax(model.predict(img), axis=-1)
    predizione = (prediction[0], list_signs[prediction[0]])
    return predizione

def adversarial_pattern(image, label, model):
    """
    Generates an adversarial pattern for a given image and label using the Fast Gradient Sign Method (FGSM).

    Parameters:
        image (tf.Tensor): The input image.
        label (tf.Tensor): The true label of the image.
        model (tf.keras.Model): The model used for prediction.

    Returns:
        tf.Tensor: The adversarial pattern generated using FGSM.
    """
    image = tf.cast(image, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = tf.keras.losses.MSE(label, prediction)
    gradient = tape.gradient(loss, image)
    signed_grad = tf.sign(gradient)
    return signed_grad

def generate_adversarials(model, y_train, X_train):
    """
    Generates adversarial examples for a given model using the Fast Gradient Sign Method (FGSM).

    Args:
        model (tf.keras.Model): The target model to generate adversarial examples for.
        y_train (numpy.ndarray): The labels of the training dataset.
        X_train (numpy.ndarray): The images of the training dataset.

    Yields:
        tuple: A tuple containing the adversarial examples (x) and their corresponding labels (y).
    """
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

def TestModel(X_train, y_train, model, defence_model, list_signs):
    """
    Test the model's performance on attacked images and evaluate the defense model's predictions.

    Args:
        X_train (numpy.ndarray): Training images.
        y_train (numpy.ndarray): Training labels.
        model: The model to be tested.
        defence_model: The defense model used for evaluation.
        list_signs (list): List of sign labels.

    Returns:
        None
    """
    # Create test sample with 10 only attacked image and predict right label
    test_set = datagen.flow(X_train, y_train, batch_size=10)
    X_test_set, y_test_set = next(test_set)
    d2 = 0
    nd2 = 0
    for X_test_set, y_test_set in zip(X_test_set, y_test_set):
        perturbations = adversarial_pattern(X_test_set.reshape((1, img_rows, img_cols, channels)), y_test_set).numpy()
        adversarial = X_test_set + perturbations * 0.4
        Model_Prediction = list_signs[model.predict(adversarial.reshape((1, img_rows, img_cols, channels))).argmax()]
        Truth_label = list_signs[y_test_set.argmax()]
        print('Model Prediction:', Model_Prediction, ",", 'Truth label:', Truth_label)
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
                d2 = d2 + 1
            else:
                print("Can not detect correctly")
                nd2 = nd2 + 1
    print("Number of correct defence predictions", d2)
    print("Number of not detected predictions", nd2)


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generates a Grad-CAM heatmap for the given input image array.

    Args:
        img_array (numpy.ndarray): The input image array.
        model (tf.keras.Model): The model used for prediction and gradient computation.
        last_conv_layer_name (str): The name of the last convolutional layer in the model.
        pred_index (int, optional): The index of the predicted class. If not provided, the top predicted class is used.

    Returns:
        numpy.ndarray: The Grad-CAM heatmap as a numpy array.
    """
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
    """
    Display the Grad-CAM visualization on top of the original image.

    Args:
        img_path (str): The path to the input image file.
        heatmap (numpy.ndarray): The heatmap generated by Grad-CAM.
        alpha (float, optional): The transparency of the heatmap overlay. Defaults to 0.3.
    """
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
    
    # Dictionary mapping class labels to their corresponding names
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
    
    
    # Import necessary libraries and modules
    data, X_train, y_train, X_val, X_test, y_test, y_val = readData()
    
    # Visualize some sample images from the dataset
    num_of_samples, num_classes, list_signs = visualization_of_image(data, X_train, y_train)
    print("lista segnaliiiiiii", list_signs)
    
    # Display the distribution of classes in the dataset
    datasetDistribution("C:/Users/luigi/OneDrive/Documenti/GitHub/Image-Perturbation-for-Visual-Security-Enhancement/FGSM_SIFAI/german-traffic-signs/train.p", num_classes, classes)
    
    # Preprocess the input images
    X_train = np.array(list(map(preprocess, X_train)))
    X_val = np.array(list(map(preprocess, X_val)))
    X_test = np.array(list(map(preprocess, X_test)))
    
    # View some processed images
    view_image_processed(X_train)
    
    # Reshape and preprocess the images
    X_train, X_test, X_val = Reshape_mapped_and_preprocessed_images(X_train, X_test, X_val)
    
    # Generate augmented data using data manipulation techniques
    datagen = manipulate_data(X_train, y_train)
    
    # Categorize the image labels
    y_train = to_categorical(y_train, 43)
    y_test = to_categorical(y_test, 43)
    y_val = to_categorical(y_val, 43)
    
    # Create an offline emissions tracker
    tracker = OfflineEmissionsTracker(country_iso_code="ITA")
    
    # Start tracking emissions
    # tracker.start()
    
    # Create the modified model using the Fast Gradient Sign Method (FGSM)
    model = fgsm.modified_model()
    
    ##############  FIT THE MODEL BASE TO TRAIN DATA ############# 
    # history = model.fit(datagen.flow(X_train, y_train, batch_size=50), steps_per_epoch=X_train.shape[0]/50, epochs=10, validation_data=(X_val, y_val), shuffle=1)
    # print(model.summary())
    # save_model(model, history)
    
    # Load the saved model and its history
    model, history = load_the_model()
    
    # Make predictions on the test data
    predictions = model.predict(X_test)
    
    # Load and preprocess an image from a file
    path = 'FGSM_SIFAI\Screenshot 2024-03-02 103033.jpg'
    image = cv2.imread(path)
    resized = cv2.resize(image, (32, 32))
    x = resized / 255.0
    
    # Apply Gaussian blur to the image
    x = cv2.GaussianBlur(x, (5, 5), 0)
    x = np.expand_dims(x, axis=0)
    
    # Get the last convolutional layer name in the model
    last_conv_layer_name = [layer.name for layer in model.layers if 'conv' in layer.name][-1]
    
    # Generate the heatmap using Grad-CAM
    heatmap = make_gradcam_heatmap(x, model, last_conv_layer_name)
    
    # Display the heatmap
    display_gradcam(path, heatmap)
    
    # Stop tracking emissions
    # tracker.stop()
    
    # Load the emissions data from a CSV file
    # emissions_csv = pd.read_csv("emissions.csv")
    # last_emissions = emissions_csv.tail(1)
    # emissions = last_emissions["emissions"] * 1000
    # energy = last_emissions["energy_consumed"]
    # cpu = last_emissions["cpu_energy"]
    # gpu = last_emissions["gpu_energy"]
    # ram = last_emissions["ram_energy"]
    # print(f"{emissions} Grams of CO2-equivalents")
    # print(f"{energy} Sum of cpu_energy, gpu_energy and ram_energy (kWh)")
    # print(f"{cpu} Energy used per CPU (kWh)")
    # print(f"{gpu} Energy used per GPU (kWh)")
    # print(f"{ram} Energy used per RAM (kWh)")
    
    # Evaluate the model's performance on the test data
    accuracy = model.evaluate(X_test, y_test)
    print('Accuracy:', accuracy)
    
    # Generate predictions and evaluate the model's performance using a confusion matrix
    y_pred, y_pred_classes, y_true, confusion_mtx = confusion_metrix(X_test, y_test, model)
    
    # Calculate and display additional metrics
    metriche(y_pred_classes, y_true)
    
    # Plot the training and validation metrics
    plot_metrics(history)
    
    # Load an image from a file
    img = load_image_from_file('FGSM_SIFAI\Screenshot 2024-03-02 103033.jpg')
    
    # Make a prediction on the image using the model
    print(predict(model, img))
    
    # Display an image from the test set
    img_rows, img_cols, channels = 32, 32, 3
    image = X_test[6000]
    plt.clf()
    plt.axis("off")
    plt.imshow(image.reshape((img_rows, img_cols, channels)))
    plt.savefig('FGSM_SIFAI\static\img3', bbox_inches='tight', pad_inches=0)
    
    # Get the label of the image
    image_label = y_test[3000]
    print("image_lable", image_label)
    
    # Generate adversarial perturbations for the image
    perturbations = adversarial_pattern(image.reshape((1, img_rows, img_cols, channels)), image_label, model).numpy()
    adversarial = image + perturbations * 0.1
    
    # Make predictions on the original and perturbed images
    print("Model prediction == ", list_signs[model.predict(image.reshape((1, img_rows, img_cols, channels))).argmax()])
    print("Prediction with Intrusion == ", list_signs[model.predict(adversarial).argmax()])
    
    # Display the perturbed image
    if channels == 1:
        plt.imshow(adversarial.reshape((img_rows, img_cols)))
    else:
        plt.imshow(adversarial.reshape((img_rows, img_cols, channels)))
    plt.show()
    
    ###### GENERATE ADVERSARIAL EXAMPLES FOR THE MODEL USING FGSM ######
    
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
    
    # Load the adversarial data from saved files
    x_adversarial_train = np.load('FGSM_SIFAI/dati_modificati/x_adversarial_train.npy')
    y_adversarial_train = np.load('FGSM_SIFAI/dati_modificati/y_adversarial_train.npy')
    
    x_adversarial_val = np.load('FGSM_SIFAI/dati_modificati/x_adversarial_val.npy')
    y_adversarial_val = np.load('FGSM_SIFAI/dati_modificati/y_adversarial_val.npy')
    
    x_adversarial_test = np.load('FGSM_SIFAI/dati_modificati/x_adversarial_test.npy')
    y_adversarial_test = np.load('FGSM_SIFAI/dati_modificati/y_adversarial_test.npy')
    
    # Combine the adversarial data with the original data
    x_combined_train = np.concatenate((X_train, x_adversarial_train), axis=0)
    y_combined_train = np.concatenate((y_train, y_adversarial_train), axis=0)
    
    x_combined_val = np.concatenate((X_val, x_adversarial_val), axis=0)
    y_combined_val = np.concatenate((y_val, y_adversarial_val), axis=0)
    
    x_combined_test = np.concatenate((X_test, x_adversarial_test), axis=0)
    y_combined_test = np.concatenate((y_test, y_adversarial_test), axis=0)
    
    # Create a robust model for defense against adversarial attacks
    defence_model = model_intrusion.create_robust_model()
    
    ###############  FIT THE DEFENSE MODEL TO THE COMBINED DATA  ############# 
    # combined_history = defence_model.fit(datagen.flow(x_combined_train, y_combined_train, batch_size=50), steps_per_epoch=x_combined_train.shape[0]/50, epochs=10, validation_data=(x_combined_val, y_combined_val), shuffle=1)
    # save_model2(defence_model, combined_history)
    
    # Load the saved defense model and its history
    defence_model, combined_history = load_the_model2()
    
    # Plot the training and validation metrics for the defense model
    plot_metrics(combined_history)
    
    # Evaluate the defense model's performance on the combined test data
    accuracy = defence_model.evaluate(x_combined_test, y_combined_test)
    print('Accuracy:', accuracy)
    
    # Generate predictions and evaluate the defense model's performance using a confusion matrix
    y_pred, y_pred_classes, y_true, confusion_mtx = confusion_metrix(x_combined_test, y_combined_test, defence_model)
    
    # Calculate and display additional metrics for the defense model
    metriche(y_pred_classes, y_true)
    
    # Generate an adversarial image for the 20km/h class
    image = X_train[1000]
    image_label = y_train[1000]
    perturbations = adversarial_pattern(image.reshape((1, img_rows, img_cols, channels)), image_label, model).numpy()
    adversarial = image + perturbations * 0.4
    
    # Make predictions on the original and perturbed images using the models
    print("Model Prediction on original image = ", list_signs[model.predict(image.reshape((1, img_rows, img_cols, channels))).argmax()])
    print("Defence Model Prediction on intrusion image = ", list_signs[defence_model.predict(adversarial).argmax()])
    
    # Display the perturbed image
    if channels == 1:
        plt.imshow(adversarial.reshape((img_rows, img_cols)))
    else:
        plt.imshow(adversarial.reshape((img_rows, img_cols, channels)))
    plt.show()
    
    # Test the models on the original and adversarial test data
    
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
    
    