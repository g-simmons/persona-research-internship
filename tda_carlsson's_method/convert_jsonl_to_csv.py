# import pandas as pd
# import json
# import os 

# jsonl_file_path = "/Users/riab/Documents/persona-research-internship/data/llm_edited_reviews_embedded/llm_edited_reviews_embedded.jsonl"

# file_data = [] # initialize list to store data 

# # read the jsonl file and append to the list
# with open(jsonl_file_path, 'r') as f:
#     for line in f:
#         json_file_data = json.loads(line)
#         file_data.append(json_file_data)

# # create a pandas DataFrame from the list 
# df = pd.DataFrame(file_data)["full_prompt_embedding"]
# print(type(df["full_prompt_embedding"].values[0]))

# print(df.dtype)
# directory_to_save_file = "/Users/riab/Documents/persona-research-internship/data/csv_converted_data"
# os.makedirs(directory_to_save_file, exist_ok=True)



# # save the DataFrame to a csv file in directory created above 
# csv_file_path = os.path.join(directory_to_save_file, 'llm_edited_reviews_embedded.csv')
# df.to_csv(csv_file_path, index=False, header=["full_prompt_embedding"])

import pandas as pd
import json
import os 
import numpy as np

jsonl_file_path = "/Users/riab/Documents/persona-research-internship/data/llm_edited_reviews_embedded/llm_edited_reviews_embedded.jsonl"

file_data = [] # initialize list to store data 

# read the jsonl file and append to the list
with open(jsonl_file_path, 'r') as f:
    for line in f:
        json_file_data = json.loads(line)
        file_data.append(json_file_data)

# create a pandas DataFrame from the list 
orig = pd.DataFrame(file_data)

# # ["full_prompt_embedding"]
df = pd.DataFrame(orig["full_prompt_embedding"].tolist())
#df["sentiment"] = 1 # TODO compute actual sentiment scores

#df.to_csv("../data/llm_edited_reviews_embedded/llm_edited_reviews_embedded.csv", index=False)

print(df["sentiment"])