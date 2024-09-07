import os
import pickle
import yaml
import logging
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from sklearn.linear_model import LogisticRegression
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_params(file_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(file_path, "r") as file:
            params = yaml.safe_load(file)
            logging.info(f"Parameters loaded successfully from {file_path}")
            return params
    except Exception as e:
        logging.error(f"Failed to load parameters from {file_path}: {e}")
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
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        logging.info("Data prepared successfully.")
        return X, y
    except Exception as e:
        logging.error(f"Failed to prepare data: {e}")
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> GradientBoostingClassifier:
    """Train a Gradient Boosting Classifier."""
    try:
        clf = LogisticRegression(C=1, solver='liblinear', penalty='l2')
        clf.fit(X_train, y_train)
        logging.info("Model trained successfully.")
        return clf
    except Exception as e:
        logging.error(f"Failed to train the model: {e}")
        raise

def save_model(model, file_path: str):
    """Save the trained model to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logging.info(f"Model saved successfully to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save the model to {file_path}: {e}")
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
    # Load parameters
    params = load_params("params.yaml")
    
    # Load processed training data
    train_data = load_data('./data/features/train_tfidf.csv')
    
    # Prepare data
    X_train, y_train = prepare_data(train_data)
    
    # Train the model
    clf = train_model(X_train, y_train, params['model_building'])
    
    # Create directory for saving the model
    create_directory("models")
    
    # Save the trained model
    save_model(clf, os.path.join("models", 'model.pkl'))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"Program failed with error: {e}")
