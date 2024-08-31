import os
import re
import logging
import pandas as pd
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer

# Download necessary NLTK data
nltk.download('wordnet')
nltk.download('stopwords')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define preprocessing functions
def lemmatization(text: str) -> str:
    """Apply lemmatization to the input text."""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    logging.debug("Lemmatization applied.")
    return " ".join(text)

def remove_stop_words(text: str) -> str:
    """Remove stop words from the input text."""
    stop_words = set(stopwords.words("english"))
    text = [word for word in text.split() if word not in stop_words]
    logging.debug("Stop words removed.")
    return " ".join(text)

def removing_numbers(text: str) -> str:
    """Remove numbers from the input text."""
    text = ''.join([char for char in text if not char.isdigit()])
    logging.debug("Numbers removed.")
    return text

def lower_case(text: str) -> str:
    """Convert text to lower case."""
    text = [word.lower() for word in text.split()]
    logging.debug("Text converted to lower case.")
    return " ".join(text)

def removing_punctuations(text: str) -> str:
    """Remove punctuations and extra whitespaces from the text."""
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub('\s+', ' ', text).strip()
    logging.debug("Punctuations removed.")
    return text

def removing_urls(text: str) -> str:
    """Remove URLs from the input text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    text = url_pattern.sub(r'', text)
    logging.debug("URLs removed.")
    return text

def remove_small_sentences(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Remove sentences with less than 3 words from the DataFrame."""
    try:
        df[column_name] = df[column_name].apply(lambda x: np.nan if len(x.split()) < 3 else x)
        logging.info("Small sentences removed from DataFrame.")
    except Exception as e:
        logging.error(f"Failed to remove small sentences: {e}")
        raise
    return df

def normalize_text(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Apply all normalization steps to the DataFrame."""
    try:
        df[column_name] = df[column_name].apply(lower_case)
        df[column_name] = df[column_name].apply(remove_stop_words)
        df[column_name] = df[column_name].apply(removing_numbers)
        df[column_name] = df[column_name].apply(removing_punctuations)
        df[column_name] = df[column_name].apply(removing_urls)
        df[column_name] = df[column_name].apply(lemmatization)
        logging.info(f"Text normalization applied to column {column_name}.")
    except Exception as e:
        logging.error(f"Failed to normalize text: {e}")
        raise
    return df

def normalized_sentence(sentence: str) -> str:
    """Normalize a single sentence."""
    try:
        sentence = lower_case(sentence)
        sentence = remove_stop_words(sentence)
        sentence = removing_numbers(sentence)
        sentence = removing_punctuations(sentence)
        sentence = removing_urls(sentence)
        sentence = lemmatization(sentence)
        logging.debug("Sentence normalized.")
    except Exception as e:
        logging.error(f"Failed to normalize sentence: {e}")
        raise
    return sentence

def save_data(df: pd.DataFrame, file_path: str):
    """Save processed DataFrame to a CSV file."""
    try:
        df.to_csv(file_path, index=False)
        logging.info(f"Data saved to {file_path}")
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
    # Fetch the data
    train_data = pd.read_csv('./data/raw/train.csv')
    test_data = pd.read_csv('./data/raw/test.csv')

    # Normalize text data
    train_processed_data = normalize_text(train_data, 'content')
    test_processed_data = normalize_text(test_data, 'content')

    # Remove small sentences
    train_processed_data = remove_small_sentences(train_processed_data, 'content')
    test_processed_data = remove_small_sentences(test_processed_data, 'content')

    # Create directory for saving processed data
    data_path = os.path.join("data", 'processed')
    create_directory(data_path)

    # Save the processed data
    save_data(train_processed_data, os.path.join(data_path, "train_processed.csv"))
    save_data(test_processed_data, os.path.join(data_path, "test_processed.csv"))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"Program failed with error: {e}")
