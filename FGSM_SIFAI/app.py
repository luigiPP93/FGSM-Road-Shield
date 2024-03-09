from flask import Flask, render_template
import pickle
from flask import request
import pandas as pd
import fgsm
import cv2
import os
import numpy as np
import base64
from flask import send_file
from PIL import Image
import matplotlib.pyplot as plt
from flask import jsonify

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'FGSM_SIFAI/imgTest'

@app.route('/')
def index():
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
   
    return render_template('index.html',list_signs=list_signs)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        model_type = request.form['modelType']
        image1 = request.files['image1']
        #image2 = request.files['image2']
        
        filename = image1.filename
        image1.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print("path",filepath)
    

        # Fai quello che vuoi fare con le immagini qui, come stamparle sul terminale
        print("Model Type:", model_type)
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
    try:
        image.save(filepath)
        print(f"Immagine salvata correttamente in {filepath}")
    except Exception as e:
        print(f"Errore durante il salvataggio dell'immagine: {e}")
        
        
@app.route('/applyIntrusion', methods=['POST'])
def applyIntrusion():
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
    if request.method == 'POST':
        model_type = request.form['modelType']
        image1 = request.files['image1']
        #label = request.form['label']
        #print("LABEL",label)
        #image2 = request.files['image2']
        
        filename = image1.filename
        image1.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print("path",filepath)
    

        # Fai quello che vuoi fare con le immagini qui, come stamparle sul terminale
        print("Model Type:", model_type)
        print("Image 1:", image1)
        #print("Image 2:", image2)
        
        
        # Leggi l'immagine dal percorso del file
        img_rows, img_cols, channels = 32, 32, 1
        model,history=fgsm.load_the_model()
        #img=filepath
        #img = cv2.imread(filepath)
        indice_classe = 3
        num_classi = 43

        # Creiamo un vettore di etichette di dimensioni 43
        etichetta = np.zeros(num_classi)
        etichetta[indice_classe] = 1
        img=fgsm.load_image_from_file(filepath)
        perturbations=fgsm.adversarial_pattern(img.reshape((1, img_rows, img_cols, channels)), etichetta,model).numpy()
        adversarial = img + perturbations * 0.3
        #predizione = fgsm.predict(model,img)
        #print("Predizione",predizione)
        #os.remove(filepath)
        print("Model prediction == ",list_signs[model.predict(img.reshape((1, img_rows, img_cols, channels))).argmax()])
        print("Prediction with Intrusion== ", list_signs[model.predict(adversarial).argmax()])
        
        if channels == 1:
            plt.imshow(adversarial.reshape((img_rows, img_cols)))
        
        else:
            plt.imshow(adversarial.reshape((img_rows, img_cols, channels)))

        # Salvare l'immagine
        plt.savefig('C:\\Users\\luigi\\OneDrive\\Documenti\\GitHub\\Image-Perturbation-for-Visual-Security-Enhancement\\FGSM_SIFAI\\static\\adversarial_image.png')
        
        image_path = 'C:\\Users\\luigi\\OneDrive\\Documenti\\GitHub\\Image-Perturbation-for-Visual-Security-Enhancement\\FGSM_SIFAI\\static\\adversarial_image.png'

        # Restituire il percorso dell'immagine come risposta JSON
        return jsonify({'image_url': image_path, 'prediction': str(list_signs[model.predict(adversarial).argmax()])})

if __name__ == '__main__':
    import sys
    #print(sys.executable)
    app.run(debug=True)
