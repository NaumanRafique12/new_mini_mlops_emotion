import os
import dagshub
import mlflow

# Set environment variables for authentication
os.environ['DAGSHUB_USER'] = 'Aretec Noman'
os.environ['DAGSHUB_TOKEN'] = '29744f7f49906e9792e2cd23201ba82a6ca563fb'

# Initialize DagsHub and set MLflow tracking URI
dagshub.init(repo_owner='noman.rafique', repo_name='new_mini_mlops_emotion', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/noman.rafique/new_mini_mlops_emotion.mlflow')

# Start MLflow run
with mlflow.start_run():
    mlflow.log_param('parameter name', 'value')
    mlflow.log_metric('metric name', 1)
