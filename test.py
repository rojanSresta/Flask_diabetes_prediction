import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df=pd.read_csv('s_diabetes.csv')

cdf=df.copy()

X=cdf.drop(columns='Outcome',axis=1)

y=cdf['Outcome']

sc=StandardScaler()
sc.fit(X)
sd = sc.transform(X)
X=sd
y=cdf['Outcome']


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=2)

from sklearn import svm
classifier=svm.SVC(kernel='linear')

classifier.fit(X_train,y_train)

from sklearn.metrics import accuracy_score

X_train_prediction=classifier.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,y_train)
training_data_accuracy

X_test_prediction=classifier.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,y_test)
test_data_accuracy


input_data = (2,121,70,32,95,39.1,0.886,23)

n = np.asarray(input_data)

input_data_reshape = n.reshape(1, -1)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    std_data = sc.transform(input_data_reshape)
    
    prediction=classifier.predict(std_data)

    print("not diabetic") if prediction[0]==0 else print("diabetic")
