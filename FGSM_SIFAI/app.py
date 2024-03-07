from flask import Flask, render_template
import pickle
from flask import request
import pandas as pd
import fgsm
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        model_type = request.form['modelType']
        image1 = request.files['image1']
        image2 = request.files['image2']

        # Fai quello che vuoi fare con le immagini qui, come stamparle sul terminale
        print("Model Type:", model_type)
        print("Image 1:", image1)
        print("Image 2:", image2)
        
        
        model,history=fgsm.load_the_model()
        #img=fgsm.img_process(image1)
        print(fgsm.predict(model,image1))

        return "Images received and processed."


if __name__ == '__main__':
    import sys
    #print(sys.executable)
    app.run(debug=True)
