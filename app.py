from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    df=pd.read_csv('s_diabetes.csv')

    cdf=df.copy()

    X=cdf.drop(columns='Outcome',axis=1)

    y=cdf['Outcome']

    global sc
    sc=StandardScaler()
    X_std = sc.fit_transform(X)

    X_train,X_test,y_train,y_test=train_test_split(X_std,y,test_size=0.2,stratify=y,random_state=2)

    
    classifier=svm.SVC(kernel='linear')

    classifier.fit(X_train,y_train)
    pickle.dump(classifier, open("diabetes.pkl", "wb"))

    return render_template('home.html')
    
@app.route('/result', methods=['GET', 'POST'])
def predict():
    pregnancies = float(request.form['Pregnancies'])
    glucose = float(request.form['Glucose'])
    blood_pressure = float(request.form['BloodPressure'])
    skin_thickness = float(request.form['SkinThickness'])
    insulin = float(request.form['Insulin'])
    bmi = float(request.form['BMI'])
    DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
    age = float(request.form['Age'])

    model = pickle.load(open("diabetes.pkl", "rb"))

    n = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, DiabetesPedigreeFunction, age])
    input_data = n.reshape(1, -1)
    std_data = sc.transform(input_data)
    prediction = model.predict(std_data)
    if (prediction[0]==0):
        result = "You have  not diabetes"
    else:
        result = "diabetes"
    return render_template('result.html', result = result)
    
if __name__ == '__main__':
    app.run(debug= True)
