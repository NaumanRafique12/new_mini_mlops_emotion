import os
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_params(file_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(file_path, "r") as file:
            params = yaml.safe_load(file)
            logging.info(f"Successfully loaded parameters from {file_path}")
            return params
    except Exception as e:
        logging.error(f"Failed to load parameters from {file_path}: {e}")
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Successfully loaded data from {file_path}")
        return df
    except Exception as e:
        logging.error(f"Failed to load data from {file_path}: {e}")
        raise

def split_data(df: pd.DataFrame, test_size: float, random_state: int = 42) -> tuple:
    """Split the data into training and test sets."""
    try:
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)
        logging.info("Successfully split the data into training and test sets")
        return train_data, test_data
    except Exception as e:
        logging.error(f"Failed to split the data: {e}")
        raise

def save_data(df: pd.DataFrame, file_path: str):
    """Save data to a CSV file."""
    try:
        df.to_csv(file_path, index=False)
        logging.info(f"Successfully saved data to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save data to {file_path}: {e}")
        raise

def create_directory(directory_path: str):
    """Create a directory if it does not exist."""
    try:
        os.makedirs(directory_path, exist_ok=True)
        logging.info(f"Directory created at {directory_path} or already exists")
    except Exception as e:
        logging.error(f"Failed to create directory {directory_path}: {e}")
        raise

def main():
    # Load parameters
    params = load_params("params.yaml")
    test_size = params['data_splitting']['test_size']
    
    # Load data
    df = load_data("./data/external/data.csv")
    
    # Split data
    train_data, test_data = split_data(df, test_size)
    
    # Create directory for saving data
    data_path = os.path.join("data", 'raw')
    create_directory(data_path)
    
    # Save train and test data
    save_data(train_data, os.path.join(data_path, "train.csv"))
    save_data(test_data, os.path.join(data_path, "test.csv"))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"Program failed with error: {e}")
