import requests
import json
import os
from collections import OrderedDict
"""
import infini_gram
from infini_gram.engine import InfiniGramEngine

# from transformers import AutoTokenizer

# Ensure the virtual environment path is included in the Python path
import sys

# sys.path.append('/home/logan/.venv/lib/python3.12/site-packages')

try:
    import infini_gram
    from infini_gram.engine import InfiniGramEngine

    print("infini-gram imported successfully")
except ImportError:
    print("infini-gram is not installed")"""

"""try:
    import infinigram
    from infinigram.engine import InfiniGramEngine
    from transformers import AutoTokenizer
except ImportError:
    print("infini-gram is not installed")"""


def get_term_frequency(term):
    url = "https://api.infini-gram.io/"
    headers = {"Content-Type": "application/json"}
    payload = {"index": "v4_piletrain_llama", "query_type": "count", "query": term}

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()['count']


def get_term_dict(terms):
    term_dict = {}
    for term in terms:
        term_dict[term] = get_term_frequency(term)
    return term_dict


def get_terms_from_file_string(file_string):
    return file_string.split("\n")


def get_file_string(file_name, folder_path):
    for file in os.listdir(folder_path):
        if file == file_name:
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r") as file:
                return file.read()


def add_frequencies_to_json(file_name, word_list, chunk_size=25):
    folder_path = "./data/term_frequencies"
    progress_file = os.path.join(folder_path, f"progress-{file_name}.txt")
    json_file_path = os.path.join(folder_path, f"{file_name}-frequencies.json")
    print(json_file_path)
    print(progress_file)
    # progress_file = f"progress-{file_name}.txt"

    try:
        if os.path.exists(json_file_path):
            with open(json_file_path) as f:
                data = json.load(f)
        else:
            data = {}
        if file_name not in data:
            data[file_name] = {}

        if os.path.exists(progress_file):
            with open(progress_file, "r") as f:
                last_index = int(f.read())
            counter = last_index
        else:
            last_index = 0
            counter = 0
        for i in range(last_index, len(word_list), chunk_size):
            chunk = word_list[i : i + chunk_size]
            term_dict = get_term_dict(chunk)
            data[file_name].update(term_dict)
            with open(progress_file, "w") as f:
                json.dump(data, f, indent=4)
            with open(progress_file, "w") as f:
                f.write(str(i + chunk_size))
            print(f"words, {i}-{i+chunk_size} added to {json_file_path}")
            counter = i + chunk_size
        if counter < len(word_list):
            print(f"Adding words, {counter}-{len(word_list)} to {json_file_path}")
            chunk = word_list[counter : len(word_list)]
            term_dict = get_term_dict(chunk)
            data[file_name].update(term_dict)
            with open(json_file_path, "w") as f:
                json.dump(data, f, indent=4)
            with open(progress_file, "w") as f:
                f.write(str(i + chunk_size))
            print(f"words, {i}-{len(word_list)} added to {json_file_path}")
        print("All word frequency pairs added to {json_file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")


def add_file_frequencies_to_json(file_name, folder_path):
    word_list = get_terms_from_file_string(get_file_string(file_name, folder_path))
    unique_list = list(OrderedDict.fromkeys(word_list))
    add_frequencies_to_json(file_name, unique_list, 200)


add_file_frequencies_to_json("mp.txt", "./data/term_frequencies/ontology-terms")
