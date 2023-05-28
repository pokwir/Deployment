# Flask app to predict loan status using our pickled model from LoanPredictions.py

# import Flask and jsonify
from flask import Flask, request, jsonify

# import numpy
import numpy as np

# import pandas
import pandas as pd

# import Resource, Api and reqparser
from flask_restful import Api, Resource, reqparse

# import pickle
import pickle

app = Flask(__name__)
api = Api(app)

# load our pickled model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def greeting():
    return '''Welcome to my Loan Prediction API!\
        <br> To use this API, make a JSON post request to the predictLoan endpoint with the following fields:\
            <br> That is url/predictLoan'''

# json post route
@app.route('/predictLoan', methods=['POST', 'GET'])
def predictLoan():
    # get data from post request
    data = request.get_json(force = True)

    # create a dataframe from our json data
    df = pd.DataFrame(data, index=[0])

    # predict loan status
    prediction = model.predict(df)

    # return prediction
    return jsonify('Your Loan request is: Rejected' if prediction[0] == 'N' else 'Congratulations!!!. Your Loan is Approved.')
    # return jsonify({'prediction': prediction[0]})








    # data_df = pd.DataFrame.from_dict([data])

    # # predict loan status
    # prediction = model.predict(data_df)
    # prediction = prediction[0]

    # # return prediction
    # return jsonify({'prediction': prediction})


    # req_data = request.get_json(force=True)
    # if req_data:
    #     gender = req_data['Gender']
    #     married = req_data['Married']
    #     dependents = req_data['Dependents']
    #     education = req_data['Education']
    #     self_employed = req_data['Self_Employed']
    #     applicant_income = req_data['ApplicantIncome']
    #     coapplicant_income = req_data['CoapplicantIncome']
    #     loan_amount = req_data['LoanAmount']
    #     loan_amount_term = req_data['Loan_Amount_Term']
    #     credit_history = req_data['Credit_History']
    #     property_area = req_data['Property_Area']
    # # #create TotalIncome column as a sum of ApplicantIncome and CoapplicantIncome
    # # TotalIncome = applicant_income + coapplicant_income
    
    # # #create TotalIncome_log column as a log of TotalIncome
    # # TotalIncome_log = np.log(TotalIncome)
    # # #create LoanAmount_log
    # # LoanAmount_log = np.log(loan_amount)

    # # create a dict of features to be fed into our model
    # features = {
    #     'Gender' : [gender],
    #     'Married': [married],
    #     'Dependents': [dependents],
    #     'Education': [education],
    #     'Self_Employed': [self_employed],
    #     'ApplicantIncome': [applicant_income],
    #     'CoapplicantIncome': [coapplicant_income],
    #     'LoanAmount': [loan_amount],
    #     'Loan_Amount_Term': [loan_amount_term],
    #     'Credit_History': [credit_history],
    #     'Property_Area': [property_area]
    # }

    # # create a dataframe from the dict
    # features_df = pd.DataFrame(features)

    # # predict using our model
    # y_pred = model.predict(features_df)

    # return jsonify({'prediction' : y_pred[0]})

if __name__ == '__main__':
    app.run(debug=True, port=5000)