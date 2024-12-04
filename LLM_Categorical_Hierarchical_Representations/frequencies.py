#from datasets import load_dataset
import requests
import json
import os
from collections import OrderedDict

import infini_gram
from infini_gram.engine import InfiniGramEngine
from transformers import AutoTokenizer
from joblib import Parallel, delayed

ONTOLOGY_TERMS_PATH = './data/term_frequencies/ontology-terms'
FOLDER_PATH = './data/term_frequencies'

def get_term_frequency_package(term):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", add_bos_token=False, add_eos_token=False)
    engine = InfiniGramEngine(index_dir='/mnt/bigstorage/infinigram_indices/v4_pileval_llama', eos_token_id=tokenizer.eos_token_id)
    input_ids = tokenizer.encode(term)
    result = engine.count(input_ids=input_ids)
    input_ids = tokenizer.encode(term)
    return result['count']



def get_term_frequency_api(term):
    url = 'https://api.infini-gram.io/'
    headers = {'Content-Type': 'application/json'}
    payload = {
        'index': 'v4_pileval_llama',
        'query_type': 'count',
        'query': term
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()['count']

def get_term_dict_timed(terms):
    term_dict = {}
    from datetime import datetime

    # Get the current timestamp
    earlier = datetime.now()
    early_timestamp = earlier.timestamp()
    for term in terms:
        term_dict[term] = get_term_frequency_package(term)
        #print(f'{term} pkg:', get_term_frequency_package(term))
        #print(f'{term} api:', get_term_frequency_api(term))
    now = datetime.now()
    timestamp = now.timestamp()
    print(f'time taken: {timestamp - early_timestamp}')
    return term_dict

def get_term_dict(terms):
    term_dict = {}
    from datetime import datetime

    for term in terms:
        term_dict[term] = get_term_frequency_package(term)
    return term_dict

def get_terms_from_file_string(file_string):
    return file_string.split('\n')

def get_file_string(file_name, folder_path):
    for file in os.listdir(folder_path):
        if file == file_name:
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as file:
                return file.read()

def add_frequencies_to_json(file_name, word_list, chunk_size=25, collection_type = None):
    progress_file = os.path.join(FOLDER_PATH, f"progress-{file_name}.txt")
    json_file_path = os.path.join(FOLDER_PATH, f"{file_name}-frequencies.json")
    all_files_progress_path = os.path.join(FOLDER_PATH, "completed-frequencies.txt")
    print(json_file_path)
    print(progress_file)
    print("word-list length: ")
    print(len(word_list))

    try: 
        if os.path.exists(json_file_path):
            with open(json_file_path) as f:
                data = json.load(f)
        else:
            data = {}
        if file_name not in data:
            data[file_name] = {}

        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                last_index = int(f.read())
            counter = last_index
        else:
            last_index = 0
            counter = 0
        for i in range(last_index, len(word_list) - chunk_size, chunk_size):
            chunk = word_list[i:i + chunk_size]
            term_dict = get_term_dict(chunk)
            data[file_name].update(term_dict)
            with open(json_file_path, 'w') as f:
                json.dump(data, f, indent=4)
                f.flush()
            with open(progress_file, 'w') as f:
                f.write(str(i + chunk_size))
                f.flush()
            print(f'words, {i}-{i+chunk_size} added to {json_file_path}')
            counter = i + chunk_size
        if counter < len(word_list):
            print(f'Adding words, {counter}-{len(word_list)} to {json_file_path}')
            print(word_list[len(word_list) - 1])
            chunk = word_list[counter:len(word_list)]
            term_dict = get_term_dict(chunk)
            data[file_name].update(term_dict)
            with open(json_file_path, 'w') as f:
                json.dump(data, f, indent=4)
            with open(progress_file, 'w') as f:
                f.write(str(counter + chunk_size))
                f.flush()
            print(f'words, {counter}-{len(word_list)} added to {json_file_path}')
        print(f'All word frequency pairs added to {json_file_path}')
        with open(all_files_progress_path, 'a') as f:
            f.write(f'{file_name}\n')
            f.flush()

    
    except Exception as e:
        print(f"An error occurred here: {e}")

def add_file_frequencies_to_json(file_name, folder_path, collection_type = None):
    word_list = get_terms_from_file_string(get_file_string(file_name, folder_path))
    for i in range(len(word_list)):
        if word_list[i] == '': #can make more efficient but likely unneccesary
            word_list.pop(i)
    unique_list = list(OrderedDict.fromkeys(word_list))
    add_frequencies_to_json(file_name, unique_list, 1000, collection_type)

def get_all_frequencies(folder_path):
    #for file in file_list:
        #add_file_frequencies_to_json(file, folder_path)
    progress_file = os.path.join(folder_path, "completed-frequencies.txt")
    if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                completed_files = f.read().split('\n')
                completed_set = set(completed_files)
            uncompleted_files = []
            file_list = get_file_names(ONTOLOGY_TERMS_PATH)
            for file in file_list:
                if file not in completed_set:
                    uncompleted_files.append(file)
            Parallel(n_jobs=4)(delayed(add_file_frequencies_to_json)(file, folder_path) for file in uncompleted_files)
    
    #results = Parallel(n_jobs=4)(delayed(add_file_frequencies_to_json)(file, folder_path) for file in file_list)
def get_file_names(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

#add_file_frequencies_to_json('mp.txt', './data/term_frequencies/ontology-terms')
get_all_frequencies(FOLDER_PATH)
#delete later:
def test_freq_correct(file_name, folder_path):
    word_list = get_terms_from_file_string(get_file_string(file_name, folder_path))
    for i in range(len(word_list)):
        if word_list[i] == '': #can make more efficient but likely unneccesary
            word_list.pop(i)
    unique_list = list(OrderedDict.fromkeys(word_list))
    for elem in unique_list:
        print(f'{elem}: {get_term_frequency_package(elem)}')

#test_freq_correct('wordnet.txt', './data/term_frequencies/ontology-terms')