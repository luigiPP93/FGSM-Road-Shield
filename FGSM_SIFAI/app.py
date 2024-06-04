from flask import Flask, render_template
from flask import request
import fgsm
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from flask import Flask, render_template, request, jsonify, url_for
from tensorflow.keras.utils import to_categorical


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'FGSM_SIFAI/static/img'

@app.route('/')
def index():
    """
    Renders the index.html template with a list of traffic sign names.

    Returns:
        The rendered index.html template with the list of traffic sign names.
    """
    # Leggi i nomi dei segnali dal file CSV
    df = pd.read_csv('FGSM_SIFAI/german-traffic-signs/signnames.csv')
    list_signs = df['SignName'].tolist()
    return render_template('index.html', list_signs=list_signs)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for predicting the class of an uploaded image.

    Returns:
        str: The predicted class of the image.
    """
    if request.method == 'POST':

        image1 = request.files['image1']
        #image2 = request.files['image2']
        
        filename = image1.filename
        image1.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print("path",filepath)
    
        # Fai quello che vuoi fare con le immagini qui, come stamparle sul terminale

        print("Image 1:", image1)
        #print("Image 2:", image2)
        
        # Leggi l'immagine dal percorso del file
        
        model,history=fgsm.load_the_model()
        img=fgsm.load_image_from_file(filepath)
        predizione = fgsm.predict(model,img)
        print("Predizione",predizione)
        #os.remove(filepath)
        
        return str(predizione)
    
def save_image(image, filepath):
    """
    Save the given image to the specified filepath.

    Args:
        image (PIL.Image.Image): The image to be saved.
        filepath (str): The filepath where the image will be saved.

    Raises:
        Exception: If there is an error during the image saving process.

    Returns:
        None
    """
    try:
        image.save(filepath)
        print(f"Immagine salvata correttamente in {filepath}")
    except Exception as e:
        print(f"Errore durante il salvataggio dell'immagine: {e}")
        
        
@app.route('/applyIntrusion', methods=['POST'])
def applyIntrusion():
    """
    Apply intrusion to an image and return the perturbed image and prediction.

    This function reads a CSV file containing traffic sign names, receives an image file and a perturbation value,
    applies perturbation to the image, blurs the perturbed image, saves the perturbed image, and returns the URL
    of the perturbed image and the predicted traffic sign name.

    Returns:
        A JSON object containing the following keys:
        - 'image_url': The URL of the perturbed image.
        - 'prediction': The predicted traffic sign name.

    Raises:
        None
    """
    df = pd.read_csv('FGSM_SIFAI/german-traffic-signs/signnames.csv')
    list_signs = df['SignName'].tolist()
    
    if request.method == 'POST':
        image1 = request.files.get('image1')
       # Ottieni il valore del livello di perturbazione come float
        add_pertubation = float(request.form.get("add_pertubation", "0.1"))  # Usa "0.1" come default
        print("Pertubation value: ", add_pertubation)
        
        filename = image1.filename
        adversarial_filename = "adversarial_img.png"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        adversarial_save_path = os.path.join(app.config['UPLOAD_FOLDER'], adversarial_filename)
        image1.save(save_path)
        
        img_rows, img_cols, channels = 32, 32, 3
        model, history = fgsm.load_the_model()  # Carica il tuo modello qui
        img = fgsm.load_image_from_file(save_path)  # Carica l'immagine per l'elaborazione
        
        # Indice della classe di esempio
        sign_index = int(request.form.get('signIndex'))  # Ottiene l'indice della classe come intero

        # Numero totale di classi
        num_class = 43
        # Crea l'etichetta in formato one-hot
        etichetta_one_hot = to_categorical(sign_index, num_class)

        print(etichetta_one_hot)
        
        perturbations = fgsm.adversarial_pattern(img.reshape((1, img_rows, img_cols, channels)), etichetta_one_hot, model).numpy()
        adversarial = img + perturbations * add_pertubation
        # Applica l'effetto di sfocatura a ciascun canale separatamente
        #adversarial_blurred = np.zeros_like(adversarial)
        for i in range(3):  # Presumendo che ci siano 3 canali
            adversarial[:, :, i] = cv2.GaussianBlur(adversarial[:, :, i], (5, 5), 0)

        # Usa l'immagine sfocata per il resto del tuo codice
        # Salva l'immagine
        plt.figure()
        
        if channels == 1:
            plt.imshow(adversarial.reshape((img_rows, img_cols)), cmap='gray')
        else:
            plt.imshow(adversarial.reshape((img_rows, img_cols, channels)))
        
        plt.axis('off')
        plt.savefig(adversarial_save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        image_url = url_for('static', filename='img/' + adversarial_filename)

        return jsonify({'image_url': image_url, 'prediction': list_signs[model.predict(adversarial).argmax()]})
    
@app.route('/predictWhitIntrusion', methods=['POST'])
def predictWhitIntrusion():
    """
    Endpoint for predicting with intrusion.

    This function handles the POST request to predict with intrusion. It loads the adversarial image from the uploaded folder,
    loads the model, and predicts the class of the image using the FGSM method. The predicted class is then returned as a string.

    Returns:
        str: The predicted class of the image.
    """
    if request.method == 'POST':
        adversarial_filename = "adversarial_img.png"  # Definisci il nome del file separatamente
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], adversarial_filename)  # Utilizza il nome del file per creare il percorso
        print("path", filepath)
        model, history = fgsm.load_the_model2()
        img = fgsm.load_image_from_file(filepath)
        predizione = fgsm.predict(model, img)
        print("Predizione", predizione)
        # os.remove(filepath)

        return str(predizione)
    
    
    
if __name__ == '__main__':
    import sys
    #print(sys.executable)
    app.run(debug=True)
