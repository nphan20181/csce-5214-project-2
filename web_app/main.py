#!/usr/bin/env python
from flask import Flask, render_template, flash, request, redirect, url_for, jsonify, Markup
from house_price_model import HousePriceModel
import pickle
import pandas as pd
import numpy as np

rf_model = pickle.load(open('model.sav', 'rb'))           # load model
features = pickle.load(open('features.sav', 'rb'))        # load a list of feature name
df_states = pickle.load(open('df_states.sav', 'rb'))      # load temperature dataframe for state's lookup
 
app = Flask(__name__)
 
@app.route("/", methods=['POST', 'GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['POST', 'GET'])
def predict():
    # create an empty list that stores user's inputs
    input_fields = ['min_price', 'max_price', 'min_temp', 'max_temp', 'month', 'year']
    user_inputs = []

    # get user's input
    for field_name in input_fields:
        if request.method == 'POST':    # form submission
            user_inputs.append(float(request.form[field_name]))
        else:                           # get request
            user_inputs.append(float(request.args.get(field_name)))
    
    # get a list of state that have the specified temperature
    temp = df_states[(df_states['CityAvgYearlyTemp'] >= user_inputs[2]) & (df_states['CityAvgYearlyTemp'] <= user_inputs[3])]

    # perform one-hot-encodings
    temp = pd.concat([temp, pd.get_dummies(temp['Region'], prefix='Region_')], axis=1)
    temp = pd.concat([temp, pd.get_dummies(temp['Division'], prefix='Division_')], axis=1)

    # build inputs data frame for house price estimate
    inputs = pd.DataFrame(columns=features)
    for feature in features:
        if feature in list(temp.columns[4:]):
            inputs[feature] = temp[feature]

    # fill in month and year
    inputs['Month'] = user_inputs[4]
    inputs['Year'] = user_inputs[5]

    # fill missing values with 0
    inputs.fillna(0, inplace=True)

    # build outputs table
    outputs = pd.DataFrame(columns=['State', 'City', 'CityAvgYearlyTemp', 'PredictedPrice'])
    outputs['State'] = temp['State']
    outputs['City'] = temp['City']
    outputs['CityAvgYearlyTemp'] = inputs['CityAvgYearlyTemp']
    outputs['PredictedPrice'] = np.round(rf_model.model.predict(inputs), 2)

    # get a list of states that have the specified price
    outputs = outputs[(outputs['PredictedPrice'] >= user_inputs[0]) & (outputs['PredictedPrice'] <= user_inputs[1])]
    outputs.to_csv('static/outputs/outputs.csv', index=False)   # save predictions to csv

    return redirect(url_for('success', file_name='outputs.csv'))

@app.route('/success/<file_name>')
def success(file_name):
    # return url of output file
    url = 'http://' + request.host + '/static/outputs/' + file_name
    return url

# when running app locally
if __name__=='__main__':
      app.run(debug=True)