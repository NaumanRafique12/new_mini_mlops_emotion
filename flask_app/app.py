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
app = Flask(__name__)

# # load model from model registry
def get_latest_model_version(model_name):
    client = MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["Production"])
    if not latest_version:
        latest_version = client.get_latest_versions(model_name, stages=["None"])
    return latest_version[0].version if latest_version else None



vectorizer = pickle.load(open('models/vectorizer.pkl','rb'))
# model = pickle.load(open(r"C:\Users\VU360solutions\Desktop\mlops\mini-mlops-project\models\model.pkl",'rb'))

logged_model = 'runs:/9d2ae11f5f9a42bfa47e12cebd10eab7/Logistic Regression'

# Load model as a PyFuncModel.
model = mlflow.pyfunc.load_model(logged_model)
 

 
@app.route('/')
def home():
    return render_template('index.html',result=None)

@app.route('/predict', methods=['POST'])
def predict():

    text = request.form['text']

    # clean
    text = normalize_text(text)
    
    features = vectorizer.transform([text])  # Vectorize the text
 

# Generate column names dynamically based on the number of columns in the features
    num_columns = features.shape[1]
    column_names = [str(i) for i in range(num_columns)]
    data = pd.DataFrame(features.toarray(),columns=column_names)
    result = model.predict(data)
    
    

    # show
    return render_template('index.html', result= result[0])

app.run(debug=True)