import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
# Load the model
with open('lr_model.pkl','rb') as file :
    lr_model = pickle.load(file)

with open('scaling.pkl','rb') as file :
    scaler = pickle.load(file)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api() :
    data = request.json['data']
    print(data)
    scaled_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output = lr_model.predict(scaled_data)[0]
    return jsonify(output)

@app.route('/predict',methods=['POST'])
def predict() :
    data = [float(x) for x in request.form.values()]
    print(data)
    scaled_data = scaler.transform(np.array(data).reshape(1,-1))
    output = lr_model.predict(scaled_data)[0]
    return render_template('home.html', prediction_text = 'The house price prediction is {}'.format(output))

if __name__ == '__main__' :
    app.run(debug=True)
