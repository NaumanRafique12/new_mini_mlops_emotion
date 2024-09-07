import json
import mlflow
import logging
import dagshub
# dagshub.init(repo_owner='NaumanRafique12', repo_name='mini-mlops-Project', mlflow=True)
# mlflow.set_tracking_uri('https://dagshub.com/NaumanRafique12/mini-mlops-Project.mlflow')
import dagshub
dagshub.init(repo_owner='noman.rafique', repo_name='new_mini_mlops_emotion', mlflow=True)

mlflow.set_tracking_uri('https://dagshub.com/noman.rafique/new_mini_mlops_emotion.mlflow')

# logging configuration
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model info: %s', e)
        raise
# Adding the versions code 
import json
import os

# Function to read the JSON file
def read_json(file_path):
    # Check if the file exists and is not empty
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, "r") as f:
            data = json.load(f)
    else:
        # If the file doesn't exist or is empty, return an empty dictionary
        data = {}
    return data

# Function to add a new record to the dictionary and save it
def add_record(file_path, key, value):
    # Read the current data
    data = read_json(file_path)
    
    # Add the new record
    data[key] = value
    
    # Save the updated data back to the file
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    
    print(f"Record added: {key}: {value}")


def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)

        # Transition the model to "Staging" stage
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Production"
        )
        
        # Path to your JSON file
        #  = r"../../reports/versions.json"
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', 'reports', 'versions.json')


        # Add a new record (Example: V3: id3)
        add_record(file_path, model_version.version, model_info['run_id'])
        
        logger.debug(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')
        
    except Exception as e:
        logger.error('Error during model registration: %s', e)
        raise

def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)

        model_name = "my_model"
        register_model(model_name, model_info)
    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()