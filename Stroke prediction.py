from flask import Flask, render_template, request
import joblib
import pickle
import os
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/result', methods=['POST', 'GET'])
def result():
    gender = int(request.form['gender'])
    age = int(request.form['age'])
    hypertension = int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])
    ever_married = int(request.form['ever_married'])
    work_type = int(request.form['work_type'])
    Residence_type = int(request.form['Residence_type'])
    avg_glucose_level = float(request.form['avg_glucose_level'])
    bmi = float(request.form['bmi'])
    smoking_status = int(request.form['smoking_status'])

    x = np.array([gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type,
                  avg_glucose_level, bmi, smoking_status]).reshape(1, -1)

    scaler_path = os.path.join('D:/Python Projects/Stroke Prediction', 'models/scaler.pkl')
    scaler = None
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)

    x = scaler.transform(x)

    model_path = os.path.join('D:/Python Projects/Stroke Prediction', 'models/decision_tree.sav')
    decision_tree = joblib.load(model_path)

    Y_pred = decision_tree.predict(x)

    # FOR no stroke risk prediction
    if Y_pred==0:
        return render_template('nostroke.html')
    else:
        return render_template('stroke.html')

if __name__ == '__main__':
    app.run(debug=True, port=5960)
#######################################################################################################################
# НЕ РАБОТАЕТ, ПОКАЗЫВАЕТ ТОЛЬКО ОДНО - ЧТО НЕТ ИНСУЛЬТА ДАЖЕ НЕ ДАННЫХ, КОТОРЫЕ РАЗМЕЧЕНЫ И ДОЛЖНЫ ПОКАЗЫВАТЬ ИНСУЛЬТ#
#####################################         И Я НЕ ЗНАЮ ПОЧЕМУ ТАК      #############################################
#######################################################################################################################