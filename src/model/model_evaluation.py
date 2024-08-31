import os
import pickle
import json
import logging
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mlflow.models.signature import infer_signature

import numpy as np
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import mlflow
import seaborn as sns
import matplotlib.pyplot as plt 
import dagshub
dagshub.init(repo_owner='NaumanRafique12', repo_name='mini-mlops-Project', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/NaumanRafique12/mini-mlops-Project.mlflow')


def load_model(model_path: str):
    """Load the trained model from a file."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
            logging.info(f"Model loaded successfully from {model_path}")
            return model
    except Exception as e:
        logging.error(f"Failed to load model from {model_path}: {e}")
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully from {file_path}")
        return df
    except Exception as e:
        logging.error(f"Failed to load data from {file_path}: {e}")
        raise

def prepare_data(df: pd.DataFrame) -> tuple:
    """Prepare the feature matrix and target vector from the DataFrame."""
    try:
        X = df.drop("label",axis=1)
        y = df['label']
        logging.info("Data prepared successfully.")
        return X, y
    except Exception as e:
        logging.error(f"Failed to prepare data: {e}")
        raise

def evaluate_model(model, X_test:  pd.DataFrame, y_test:  pd.DataFrame) -> dict:
    
    """Evaluate the model and return performance metrics."""
    try:
        print("First")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        mlflow.set_experiment(experiment_name="LR in Pipeline")
        with mlflow.start_run(run_name="all_artifacts_file_model_png"):
            print("before1")
            
            # Evaluate the model
            accuracy = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            class_report = classification_report(y_test, y_pred)
            mlflow.set_tag("author","Noman")
            mlflow.set_tag("C","LR")
            mlflow.set_tag("solver","liblinear")
            mlflow.set_tag("penalty","l2")
            mlflow.log_artifacts(__file__)
            print("before2")
            
            signature = infer_signature(X_test, y_test)
            mlflow.sklearn.log_model(model,"Logistic Regression",signature=signature)
            X_test['output']=y_test
            data = mlflow.data.from_pandas(X_test)
            mlflow.log_input(data,"Testing Dataset")
            print("before3")
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=X_test.columns, yticklabels=X_test.columns)
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")

            # Save the plot as an image file
            plt.savefig("confusion_matrix.png")
            mlflow.log_artifact("confusion_matrix.png")
            print("second")
            
        print("before")

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }
        print("before")
        
        logging.info("Model evaluation metrics calculated successfully.")
        return metrics
    
    except Exception as e:
        logging.error(f"Failed to evaluate the model: {e}")
        raise

def save_metrics(metrics: dict, file_path: str):
    """Save the evaluation metrics to a JSON file."""
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logging.info(f"Metrics saved successfully to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save metrics to {file_path}: {e}")
        raise

def create_directory(directory_path: str):
    """Create a directory if it does not exist."""
    try:
        os.makedirs(directory_path, exist_ok=True)
        logging.info(f"Directory created at {directory_path} or already exists.")
    except Exception as e:
        logging.error(f"Failed to create directory {directory_path}: {e}")
        raise

def main():
    # Load the model
    model = load_model('./model/model.pkl')
    
    # Load the test data
    test_data = load_data('./data/features/test_tfidf.csv')
    
    # Prepare the data
    X_test, y_test = prepare_data(test_data)
    
    # Evaluate the model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Create directory for saving metrics
    create_directory("metrics")
    
    # Save the metrics
    save_metrics(metrics, 'metrics/metrics.json')

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"Program failed with error: {e}")
