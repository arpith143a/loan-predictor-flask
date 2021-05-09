import numpy as np
from flask import Flask, request, jsonify, render_template
from preprocess import preprocess_new, transformation
import pickle

app = Flask(__name__)
model = pickle.load(open('finalized_model.sav', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    features = [x for x in request.form.values()]
    # print(features[2])
    preprocessed_features=preprocess_new(features)
    Gender=preprocessed_features[0]
    Married=preprocessed_features[1]
    Education=preprocessed_features[3]
    Self_Employed=preprocessed_features[4]
    Credit_History=preprocessed_features[9]
    Property_Urban=preprocessed_features[10]
    Property_Semiurban=preprocessed_features[11]
    Property_Rural=preprocessed_features[12]

    transformed_features=transformation.transform([[preprocessed_features[2],preprocessed_features[7],preprocessed_features[8],preprocessed_features[13]]])
    transformed_features=transformed_features.flatten()

    Dependents=transformed_features[0]
    LoanAmount=transformed_features[1]
    Loan_Amount_Term=transformed_features[2]
    TotalIncome=transformed_features[3]


    final_features = [[Gender,Married,Dependents,Education,Self_Employed,LoanAmount,Loan_Amount_Term,Credit_History,Property_Urban,Property_Semiurban,Property_Rural,TotalIncome]]
    prediction = model.predict(final_features)
    pred=['Approved' if prediction[0]==1 else "Not approved"]


    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(pred))


if __name__ == "__main__":
    app.run(debug=True)
