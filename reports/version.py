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

# Path to your JSON file
file_path = r"reports\versions.json"

# Add a new record (Example: V3: id3)
add_record(file_path, "V5", "id3")
