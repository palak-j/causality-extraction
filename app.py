import prediction as T
import numpy as np
from flask import Flask, request, jsonify, render_template

from flask import Response

app = Flask(__name__)

pred = T.EntityExtraction()


@app.route('/')
def home():
    return render_template('index.html')

  

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    in_d = [x for x in request.form.values()]
    size_tag = "full_size"
    
    test_sen = in_d[0]
   
    output = pred.predict_tags(test_sen)

    return render_template('index.html', prediction_text='Predicted tags are:  $ {}'.format(output))


