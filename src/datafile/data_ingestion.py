import os
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data_from_url(url: str) -> pd.DataFrame:
    """Load data from a given URL."""
    try:
        df = pd.read_csv(url)
        logging.info(f"Successfully loaded data from {url}")
        return df
    except Exception as e:
        logging.error(f"Failed to load data from {url}: {e}")
        raise

def drop_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Drop a specific column from the DataFrame."""
    try:
        df.drop(columns=[column_name], inplace=True)
        logging.info(f"Successfully dropped column {column_name}")
        return df
    except KeyError as e:
        logging.error(f"Column {column_name} not found in DataFrame: {e}")
        raise
    except Exception as e:
        logging.error(f"Failed to drop column {column_name}: {e}")
        raise

def filter_and_replace(df: pd.DataFrame, column_name: str, filter_values: dict) -> pd.DataFrame:
    """Filter DataFrame based on specific values and replace them."""
    try:
        filtered_df = df[df[column_name].isin(filter_values.keys())]
        filtered_df[column_name].replace(filter_values, inplace=True)
        logging.info(f"Successfully filtered and replaced values in column {column_name}")
        return filtered_df
    except Exception as e:
        logging.error(f"Failed to filter and replace values in {column_name}: {e}")
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
    data_url = 'https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv'
    
    # Load data from URL
    df = load_data_from_url(data_url)
    
    # Drop the 'tweet_id' column
    df = drop_column(df, 'tweet_id')
    
    # Filter for 'happiness' and 'sadness' sentiments and replace them with 1 and 0 respectively
    filter_values = {'happiness': 1, 'sadness': 0}
    final_df = filter_and_replace(df, 'sentiment', filter_values)
    
    # Create directory for saving data
    data_path = os.path.join("data", 'external')
    create_directory(data_path)
    
    # Save the processed data
    save_data(final_df, os.path.join(data_path, "data.csv"))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"Program failed with error: {e}")
