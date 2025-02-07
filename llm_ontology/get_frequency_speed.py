from frequencies import get_term_frequency_package, get_term_frequency_api
import argparse
import json
import datetime

def get_keys_from_json(file_path, num_keys=50, file_name = 'wordnet.txt'):
    with open(file_path, "r") as f:
        data = json.load(f)

    keys = list(data[file_name])[:num_keys]
    return keys

def measure_time_and_speed(func, term_arr):
  start_time = datetime.datetime.now()

  for term in term_arr:
    func(term)

  end_time = datetime.datetime.now()
  time_difference = end_time - start_time
  elapsed_seconds = time_difference.total_seconds()

  return elapsed_seconds

term_arr = ['entity', 'physical_entity', 'abstraction','thing','object','causal_agent','matter','process','psychological_feature', 'attribute']

if __name__ == "__main__":
    file_path = '/home/logan/persona-research-internship/data/term_frequencies/wordnet.txt-frequencies.json'
    
    term_arr = get_keys_from_json(file_path)


    #get_term_frequency_api(term_arr[0])
    #get_term_frequency_package(term_arr[0])

    api_time = measure_time_and_speed(get_term_frequency_api, term_arr)
    print(f'Speed of API per {len(term_arr)} terms: {api_time:.4f} seconds')

    package_time = measure_time_and_speed(get_term_frequency_package, term_arr)
    print(f'Speed of package per {len(term_arr)} terms: {package_time:.4f} seconds') 