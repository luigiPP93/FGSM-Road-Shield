from flask import Flask, render_template
from flask import request
import fgsm
import os
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, url_for

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'FGSM_SIFAI/static/img'

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
        model_type = request.form.get('modelType')
        image1 = request.files.get('image1')
        
        filename = image1.filename
        adversarial_filename = "adversarial_" + filename
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        adversarial_save_path = os.path.join(app.config['UPLOAD_FOLDER'], adversarial_filename)
        image1.save(save_path)
        
        img_rows, img_cols, channels = 32, 32, 1
        model, history = fgsm.load_the_model()  # Carica il tuo modello qui
        img = fgsm.load_image_from_file(save_path)  # Carica l'immagine per l'elaborazione
        
        indice_classe = 3  # esempio di classe target
        num_classi = 43
        etichetta = np.zeros(num_classi)
        etichetta[indice_classe] = 1
        
        perturbations = fgsm.adversarial_pattern(img.reshape((1, img_rows, img_cols, channels)), etichetta, model).numpy()
        adversarial = img + perturbations * 0.3

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
       
if __name__ == '__main__':
    import sys
    #print(sys.executable)
    app.run(debug=True)
