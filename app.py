import pandas as pd
import pickle
import flask
import numpy as np
from flask import Flask,request,app,jsonify,render_template,url_for
from flask import Response
from flask_cors import CORS
app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')
# @app.route('/predict',methods = ['POST'])
# def predictapi():
#     data = request.json['data']
#     print(data)
#     newdata = [list(data.values())]
#     output = model.predict(newdata)[0]
#     print(output)
#     return str(output)
@app.route('/predict',methods = ['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_feature= [np.array(data)]
    output = model.predict(final_feature)[0]
    return render_template('home.html',prediction_text = "Air Foil Presser Is {}".format(output))
if __name__ == "__main__":
    app.run(debug=True)