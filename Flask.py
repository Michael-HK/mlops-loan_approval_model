import pickle

# prediction function
def Loan_approval_predictor(to_predict_list):
    
    #to_predict = np.array(to_predict_list).reshape(1, len(to_predict_list))
    df = pd.DataFrame(to_predict_list)       # to_predict_list is a dict
  
    numeric_cols = df['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    df_encode = df[['Education',' Dependents', 'Self_Employed', 'Gender', 'Married', 'Property_Area']]
  
    # categorical conversion
    le = pickle.load(open("label_encoder.pkl", "rb"))
    df['Education',' Dependents', 'Self_Employed', 'Gender', 'Married', 'Property_Area'] = df_encode.apply(le.transform)
  
    # scaling input value
    df['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'] = numeric_cols.apply(scaler.transform)
    loaded_model = pickle.load(open("loan_model.pkl", "rb"))
    result = loaded_model.predict(df.values)
    return result[0]
 
@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        #to_predict_list = list(to_predict_list.values())
        #to_predict_list = list(map(int, to_predict_list))
        result = Loan_approval_predictor(to_predict_list)       
        if int(result)== 1:
            prediction ='Approved Loan'
        else:
            prediction ='Loan denied! You dont meet the requirements'     
        return render_template("result.html", prediction = prediction)    
        return render_template("result.html", prediction = prediction)
