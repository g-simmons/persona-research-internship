#!/usr/bin/env python3

# from datasets import load_dataset
import requests
import json
import os
from collections import OrderedDict
import argparse
from pathlib import Path

import infini_gram
from infini_gram.engine import InfiniGramEngine
from transformers import AutoTokenizer
from joblib import Parallel, delayed


def get_term_frequency_package(term: str, index_dir: str) -> int:
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf", add_bos_token=False, add_eos_token=False
    )
    engine = InfiniGramEngine(
        index_dir=index_dir,
        eos_token_id=tokenizer.eos_token_id,
    )
    input_ids = tokenizer.encode(term)
    result = engine.count(input_ids=input_ids)
    input_ids = tokenizer.encode(term)
    return result["count"]


def get_term_frequency_api(term: str) -> int:
    url = "https://api.infini-gram.io/"
    headers = {"Content-Type": "application/json"}
    payload = {"index": "v4_pileval_llama", "query_type": "count", "query": term}

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["count"]


def get_term_dict_timed(terms: list[str], index_dir: str) -> dict[str, int]:
    term_dict = {}
    from datetime import datetime

    # Get the current timestamp
    earlier = datetime.now()
    early_timestamp = earlier.timestamp()
    for term in terms:
        term_dict[term] = get_term_frequency_package(term, index_dir)
    now = datetime.now()
    timestamp = now.timestamp()
    print(f"time taken: {timestamp - early_timestamp}")
    return term_dict


def get_term_dict(terms: list[str], index_dir: str) -> dict[str, int]:
    term_dict = {}
    from datetime import datetime

    for term in terms:
        term_dict[term] = get_term_frequency_package(term, index_dir)
    return term_dict


def get_terms_from_file_string(file_string: str) -> list[str]:
    return file_string.split("\n")


def get_file_string(file_name: str, folder_path: str) -> str:
    for file in os.listdir(folder_path):
        if file == file_name:
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r") as file:
                return file.read()


def add_frequencies_to_json(
    file_name: str,
    word_list: list[str],
    folder_path: str,
    index_dir: str,
    chunk_size: int = 25,
    collection_type: str | None = None
) -> None:
    progress_file = os.path.join(folder_path, f"progress-{file_name}.txt")
    json_file_path = os.path.join(folder_path, f"{file_name}-frequencies.json")
    all_files_progress_path = os.path.join(folder_path, "completed-frequencies.txt")

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
        elif len(data[file_name].keys()) != 0:
            last_index = len(data[file_name].keys())
            counter = last_index
            with open(progress_file, "w") as f:
                f.write(str(counter))
                f.flush()
        else:
            last_index = 0
            counter = 0
        for i in range(last_index, len(word_list) - chunk_size, chunk_size):
            chunk = word_list[i : i + chunk_size]
            term_dict = get_term_dict(chunk, index_dir)
            data[file_name].update(term_dict)
            with open(json_file_path, "w") as f:
                json.dump(data, f, indent=4)
                f.flush()
            with open(progress_file, "w") as f:
                f.write(str(i + chunk_size))
                f.flush()
            print(f"words, {i}-{i+chunk_size} added to {json_file_path}")
            counter = i + chunk_size
        if counter < len(word_list):
            # print(f'Adding words, {counter}-{len(word_list)} to {json_file_path}')
            chunk = word_list[counter : len(word_list)]
            term_dict = get_term_dict(chunk, index_dir)
            data[file_name].update(term_dict)
            with open(json_file_path, "w") as f:
                json.dump(data, f, indent=4)
            with open(progress_file, "w") as f:
                f.write(str(counter + chunk_size))
                f.flush()
            print(f"words, {counter}-{len(word_list)} added to {json_file_path}")
        print(f"All word frequency pairs added to {json_file_path}")
        with open(all_files_progress_path, "a") as f:
            f.write(f"{file_name}\n")
            f.flush()

    except Exception as e:
        print(f"An error occurred here: {e}")


def add_file_frequencies_to_json(
    file_name: str,
    ontology_terms_path: str, 
    folder_path: str,
    index_dir: str,
    collection_type: str | None = None
) -> None:
    print(file_name, ontology_terms_path)
    word_list = get_terms_from_file_string(
        get_file_string(file_name, ontology_terms_path)
    )
    i = 0
    while i < len(
        word_list
    ):  # using while loop so length of word_list changes appropriately with word_list.pop()
        if word_list[i] == "":  # can make more efficient but likely unneccesary
            word_list.pop(i)
        else:
            i += 1
    unique_list = list(OrderedDict.fromkeys(word_list))
    add_frequencies_to_json(file_name, unique_list, folder_path, index_dir, 1000, collection_type)


def get_all_frequencies(ontology_terms_path: str, folder_path: str, index_dir: str) -> None:
    progress_file = os.path.join(folder_path, "completed-frequencies.txt")
    file_list = sorted(get_file_names(ontology_terms_path))
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            progress_file_content = f.read()
            if progress_file_content:
                completed_files = progress_file_content.split("\n")
            else:
                completed_files = []
            completed_set = set(completed_files)
        uncompleted_files = []
        for file in file_list:
            if file not in completed_set:
                uncompleted_files.append(file)
    else:
        uncompleted_files = file_list
    print('file-list', uncompleted_files)
    Parallel(n_jobs=16)(
        delayed(add_file_frequencies_to_json)(file, ontology_terms_path, folder_path, index_dir)
        for file in uncompleted_files
    )



def get_file_names(path: str) -> list[str]:
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

def get_selected_frequencies(ontology_terms_path: str, folder_path: str, index_dir: str, *args: str) -> None:
    if not args:
        get_all_frequencies(ontology_terms_path, folder_path, index_dir)
    else:
        file_list = args
        Parallel(n_jobs=16)(
            delayed(add_file_frequencies_to_json)(
                file, ontology_terms_path, folder_path, index_dir
            )
            for file in file_list
        )


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    DATA_DIR = script_dir / "../data"
    ONTOLOGY_TERMS_PATH = DATA_DIR / "term_frequencies" / "ontology-terms"
    TERM_FREQUENCIES_PATH = DATA_DIR / "term_frequencies"
    DEFAULT_INDEX_DIR = "/mnt/bigstorage/infinigram_indices/v4_pileval_llama"

    parser = argparse.ArgumentParser(description="Process term frequencies")
    parser.add_argument(
        "--ontology-terms-path",
        default=ONTOLOGY_TERMS_PATH,
        help="Path to ontology terms directory",
    )
    parser.add_argument(
        "--folder-path",
        default=TERM_FREQUENCIES_PATH,
        help="Path to output folder",
    )
    parser.add_argument(
        "--index-dir",
        default=DEFAULT_INDEX_DIR,
        help="Path to the infinigram index directory",
    )
    parser.add_argument(
        "files", nargs="*", help="Optional list of specific files to process"
    )

    args = parser.parse_args()
    get_selected_frequencies(args.ontology_terms_path, args.folder_path, args.index_dir, *args.files)
