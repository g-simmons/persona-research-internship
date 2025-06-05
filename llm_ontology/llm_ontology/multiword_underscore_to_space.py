#!/usr/bin/env python3

# from datasets import load_dataset
import json
import os
from pathlib import Path
from joblib import Parallel, delayed

def get_file_string(file_name: str, folder_path: str) -> str:
    for file in os.listdir(folder_path):
        if file == file_name:
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r") as file:
                return file.read()

def create_multiword_files(pre_processed_path: str, post_processed_name: str):
    # Get the parent directory of pre_processed_path
    parent_dir = os.path.dirname(pre_processed_path)
    print(parent_dir)
    post_proccesed_path = os.path.join(parent_dir, post_processed_name)
    print(post_proccesed_path)
    
    # Create directory only if it doesn't exist
    if not os.path.exists(post_proccesed_path):
        os.makedirs(post_proccesed_path)
    
    for file in os.listdir(pre_processed_path):
        content = get_file_string(file, pre_processed_path)
        processed_content = content.replace('_', ' ')
        # Add 'multi' to the filename
        file_name, file_ext = os.path.splitext(file)
        new_file_name = f"{file_name}_multi{file_ext}"
        new_file_path = os.path.join(post_proccesed_path, new_file_name)
        
        # Only write if the file doesn't exist
        if not os.path.exists(new_file_path):
            with open(new_file_path, 'w') as f:
                f.write(processed_content)

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    DATA_DIR = script_dir / "../data"
    ONTOLOGY_TERMS_PATH = DATA_DIR / "term_frequencies" / "ontology-terms"
    create_multiword_files(ONTOLOGY_TERMS_PATH, "processed-ontology-terms")