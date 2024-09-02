import os
import pandas as pd
import yaml
import logging
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
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
        df.fillna('', inplace=True)
        logging.info(f"Data loaded successfully from {file_path}")
        return df
    except Exception as e:
        logging.error(f"Failed to load data from {file_path}: {e}")
        raise

def apply_bow(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    """Apply Bag of Words (BoW) using CountVectorizer."""
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)
        X_train_bow = vectorizer.fit_transform(train_data['content'].values)
        X_test_bow = vectorizer.transform(test_data['content'].values)
        logging.info("Bag of Words applied successfully.")
        
        with open(os.path.join("models", 'vectorizer.pkl'), 'wb') as file:
            pickle.dump(vectorizer, file)
        logging.info("Bag of Words Vectrozier is saved.")
        
        return X_train_bow, train_data['sentiment'].values, X_test_bow, test_data['sentiment'].values
    except Exception as e:
        logging.error(f"Failed to apply Bag of Words: {e}")
        raise

def create_dataframe(X_bow, y) -> pd.DataFrame:
    """Create a DataFrame from the BoW features and labels."""
    try:
        df = pd.DataFrame(X_bow.toarray())
        df['label'] = y
        logging.info("DataFrame created successfully.")
        return df
    except Exception as e:
        logging.error(f"Failed to create DataFrame: {e}")
        raise

def save_data(df: pd.DataFrame, file_path: str):
    """Save DataFrame to a CSV file."""
    try:
        df.to_csv(file_path, index=False)
        logging.info(f"Data saved successfully to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save data to {file_path}: {e}")
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
    max_features = params['feature_engineering']['max_features']
    
    # Load processed training and test data
    train_data = load_data('./data/processed/train_processed.csv')
    test_data = load_data('./data/processed/test_processed.csv')
    
    # Apply Bag of Words (BoW)
    X_train_bow, y_train, X_test_bow, y_test = apply_bow(train_data, test_data, max_features)
    
    # Create DataFrames from BoW features
    train_df = create_dataframe(X_train_bow, y_train)
    test_df = create_dataframe(X_test_bow, y_test)
    
    # Create directory for saving feature data
    data_path = os.path.join("data", "features")
    create_directory(data_path)
    
    # Save the processed data
    save_data(train_df, os.path.join(data_path, "train_tfidf.csv"))
    save_data(test_df, os.path.join(data_path, "test_tfidf.csv"))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"Program failed with error: {e}")
