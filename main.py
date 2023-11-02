import os
import pandas as pd
import torch
import torchvision
import pandera as pa
import pandera.extensions as extensions

import json
# Load model directly
from llama_index.embeddings import HuggingFaceEmbedding
from typing import Dict, List, Tuple, NamedTuple
# File path for prompts
file_path = "prompts.jsonl"

# Read the prompts into a DataFrame
input_df = pd.read_json(file_path, lines=True)

# Load model from HuggingFace
embed_model = HuggingFaceEmbedding(model_name="bert-base-uncased")
# Initialize a list to store the embeddings
embeddings = []

# Process each prompt and obtain embeddings
for index, row in input_df.iterrows():
    prompt = row['prompt']
    # Get embeddings for the prompt
    prompt_embeddings = embed_model.get_text_embedding(prompt)
    #Store the embeddings (up to 5)
    prompt_data = {"prompt": prompt, "embeddings": prompt_embeddings}
    embeddings.append(prompt_data)

# Schema Input DataFrame
input_schema = pa.DataFrameSchema({
    "prompt": pa.Column(pa.String, required=True),
})
# Validate prompts
validate_input = input_schema.validate(input_df)
print(validate_input)

# Save the data in the JSONL format
output_file = 'prompt_embeddings.jsonl'
try:
    with open(output_file, 'w') as jsonl_file:
        for item in embeddings:
            jsonl_file.write(json.dumps(item) + '\n')
except Exception as e:
    print(f"An error occurred: {e}")



# Read output file into DataFrame
output_df = pd.read_json(output_file, lines=True)

# Define a custom check function to validate the dtype of the column
def check_dtype(val, expected_dtype):
    return val.dtype == expected_dtype

# Define a schema for the DataFrame
output_schema = pa.DataFrameSchema({
    "prompt": pa.Column(pa.String, required=True),
    "embeddings": pa.Column(
        List[float]
    ),
})

validate_output = output_schema.validate(output_df)
print(validate_output)

print(f"Saved to {output_file}")