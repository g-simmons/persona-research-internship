import pandas as pd
import json
import os 

jsonl_file_path = "/Users/riab/Documents/persona-research-internship/data/dim_reduced_movie_reviews/all_data_2.jsonl"

file_data = [] # initialize list to store data 

# read the jsonl file and append to the list
with open(jsonl_file_path, 'r') as f:
    for line in f:
        json_file_data = json.loads(line)
        file_data.append(json_file_data)

# create a pandas DataFrame from the list 
df = pd.DataFrame(file_data)

directory_to_save_file = "/Users/riab/Documents/persona-research-internship/data/csv_converted_data"
os.makedirs(directory_to_save_file, exist_ok=True)

# save the DataFrame to a csv file in directory created above 
csv_file_path = os.path.join(directory_to_save_file, 'all_data_2.csv')
df.to_csv(csv_file_path, index=False)