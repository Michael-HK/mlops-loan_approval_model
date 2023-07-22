import pickle
import pandas as pd
import request

# prediction function
def Loan_approval_predictor(to_predict_list):
    
    df = pd.DataFrame.from_dict(to_predict_list)       # to_predict_list is a dict
  
    numeric_cols = df['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    df_encode = df[['Education',' Dependents', 'Self_Employed', 'Gender', 'Married', 'Property_Area']]
  
    # categorical conversion
    le = pickle.load(open("label_encoder.pkl", "rb"))
    df['Education',' Dependents', 'Self_Employed', 'Gender', 'Married', 'Property_Area'] = df_encode.apply(le.transform)
  
    # scaling input value
    scaler = pickle.load(open("scaler_input.pkl", "rb"))
    df['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'] = numeric_cols.apply(scaler.transform)
    loaded_model = pickle.load(open("loan_model.pkl", "rb"))
    result = loaded_model.predict(df.to_numpy().tolist())
    return result[0]
 
@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
        predict_list = request.form.to_dict()
        result = Loan_approval_predictor(predict_list)       
        if int(result)== 1:
            prediction ='Loan Approved'
        else:
            prediction ="Loan denied! You don't meet the requirements"     
        return render_template(result.html, prediction = prediction)    
        return render_template(result.html, prediction = prediction)
