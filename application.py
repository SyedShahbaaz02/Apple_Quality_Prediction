import flask
from flask import Flask, request, app,render_template
from flask import Response
import pickle
import numpy as np
import pandas as pd


application = Flask(__name__)
app=application

scaler=pickle.load(open("Model/applescaler.pkl", "rb"))
model = pickle.load(open("Model/appleclassifier.pkl", "rb"))

## Route for homepage

@app.route('/')
def index():
    return render_template('index.html')

## Route for Single data point prediction
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    result=""

    if request.method=='POST':

        Size=float(request.form.get("Size"))
        Weight = float(request.form.get('Weight'))
        Sweetness = float(request.form.get('Sweetness'))
        Crunchiness = float(request.form.get('Crunchiness'))
        Juiciness = float(request.form.get('Juiciness'))
        Ripeness = float(request.form.get('Ripeness'))
        Acidity = float(request.form.get('Acidity'))
      

        new_data=scaler.transform([[Size, Weight, Sweetness, Crunchiness,Juiciness,
       Ripeness, Acidity]])
        predict=model.predict(new_data)
       
        if predict[0] ==1 :
            result = 'Good'
        else:
            result ='Bad'
            
        return render_template('single_prediction.html',result=result)

    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")