from flask import Flask, render_template
import pickle
from flask import request
import pandas as pd
import fgsm
import cv2
import os
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'FGSM_SIFAI/imgTest'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        model_type = request.form['modelType']
        image1 = request.files['image1']
        image2 = request.files['image2']
        
        filename = image1.filename
        image1.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print("path",filepath)
    

        # Fai quello che vuoi fare con le immagini qui, come stamparle sul terminale
        print("Model Type:", model_type)
        print("Image 1:", image1)
        print("Image 2:", image2)
        
        
        # Leggi l'immagine dal percorso del file
        
        model,history=fgsm.load_the_model()
        img=fgsm.load_image_from_file(filepath)
        predizione = fgsm.predict(model,img)
        print("Predizione",predizione)
        os.remove(filepath)
        
        return str(predizione)


if __name__ == '__main__':
    import sys
    #print(sys.executable)
    app.run(debug=True)
