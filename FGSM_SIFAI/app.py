from flask import Flask, render_template
import pickle
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')





if __name__ == '__main__':
    import sys
    #print(sys.executable)
    app.run(debug=True)
