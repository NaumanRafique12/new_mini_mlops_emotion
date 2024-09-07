 
import mlflow

from flask import Flask, render_template,request
import mlflow
from preprocessing_utility import normalize_text
import dagshub
import pickle
import pandas as pd
from mlflow.tracking import MlflowClient
# dagshub.init(repo_owner='NaumanRafique12', repo_name='mini-mlops-Project', mlflow=True)
# mlflow.set_tracking_uri('https://dagshub.com/NaumanRafique12/mini-mlops-Project.mlflow')
dagshub.init(repo_owner='noman.rafique', repo_name='new_mini_mlops_emotion', mlflow=True)

mlflow.set_tracking_uri('https://dagshub.com/noman.rafique/new_mini_mlops_emotion.mlflow')

logged_model = 'runs:/2d71df1644b545db9eaba0b1554d1886/Logistic Regression'
vectorizer = pickle.load(open('models/vectorizer.pkl','rb'))

# Load model as a PyFuncModel.
model = mlflow.pyfunc.load_model(logged_model)
# model = pickle.load(open(r"C:\Users\VU360solutions\Desktop\mlops\mini-mlops-project\models\model.pkl",'rb'))
 


features = vectorizer.transform(["text"])  # Vectorize the text
 

# Generate column names dynamically based on the number of columns in the features
num_columns = features.shape[1]
column_names = [str(i) for i in range(num_columns)]
data = pd.DataFrame(features.toarray(),columns=column_names)
prediction = model.predict(data)  # Predict using the model
print("Prediction:", prediction[0])  # Output the prediction
