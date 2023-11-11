import cohere
import torch
import torchvision
import pandas 
import pandera
import json
from schemas import expected_input_schema, expected_output_schema
from llama_index.embeddings import HuggingFaceEmbedding

# Local embedding 
# Load model from HuggingFace
embed_model = HuggingFaceEmbedding(model_name="bert-base-uncased")

# Cloud based embedding

# Cohere Key:
co = cohere.Client("51F5llrM1i81Mp4A2HJ0TpApR4FF9Lpn12Nc90pN")

# raises error flag if schema doesn't match 
def error_flag(dict,schema,str):
  try:
    schema.validate(dict, lazy=True)
  except pandera.errors.SchemaErrors as err:
    print(str)
    print(err)


# The output Schema has an a column that consists of a list of floats...
# There is no way to check the values of the data by using pandera.Check()
# So I made a brute force checker that looks at values in the cohere embedding scores: (10>= x >= -10)

def embedding_checker(embeddings):

  for i in range(0, len(embeddings)):
    test = pandas.DataFrame({
      "prompt_embedding" : embeddings[i]
    })

    test_schema = pandera.DataFrameSchema(
      {
        "prompt_embedding": pandera.Column(float, pandera.Check.less_than_or_equal_to(10),pandera.Check.greater_than_or_equal_to(-10))
      }
    )
    error_flag(test, test_schema, f"PROMPT EMBEDDING INFORMATION ERROR {i}")


# Find Json from input path
def json_input_directory() -> dict:
    # directory/folder path
    dir_path = input("Enter directory for Json file: ")
    raw_dir_path = r'{}'.format(dir_path)
    # puts Json Data in varaible
    dictionary = pandas.read_json(raw_dir_path, lines=True)

    #Implement Panderas checking the "dictionary" var in this function

    return dictionary
   

# Put Json Data in list
def json_to_list(dict) -> list:
  # create list
  list = []
  # check if "Prompts" dictionary in Json exists
  if "prompt" in dict:
      for value in dict["prompt"]:
         # add to list to use for Cohere
         list.append(value)
  return list


# Creates json for output
def json_output(embedding_list, prompt_list) -> dict:
  #sets up the dictionary frame

  df = pandas.DataFrame({'prompt': prompt_list,'prompt_embedding': embedding_list})
  # Save the DataFrame to a JSONL file
  df.to_json('output.jsonl', orient='records', lines=True)
  
  return df

# Create local embedding function
def local_embedding(dict):
   # Initialize a list to store the embeddings
    embeddings = []
    for key in dict:
       prompt_length = len(dict[key])
       for i in range(0, prompt_length):
          # Get embeddings for the prompt
          prompt = dict[key][i]
          prompt_embedding = embed_model.get_text_embedding(prompt)
          prompt_data = {"prompt": prompt, "prompt_embedding":prompt_embedding}
          # Store embedding for each prompt into a list
          embeddings.append(prompt_data)

    # Save the data in the JSONL format
    output_file = 'output.jsonl'
    try:
        with open(output_file, 'w') as jsonl_file:
            for item in embeddings:
                jsonl_file.write(json.dumps(item) + '\n')
    except Exception as e:
        print(f"An error occurred: {e}")

# Main ----------------------------------------------------------------------------------------

# Save jsonl input into dictionary
dict = json_input_directory()
error_flag(dict, expected_input_schema, "INPUT SCHEMA ERROR")

# Prompt to choose between local and cloud embedding
user_choice = input("Choose between local and cloud embedding: ")
if(user_choice == 'local'):
  local_embedding(dict)
elif(user_choice == 'cloud'):
  # Cloud embedding
  list = json_to_list(dict)

  # Cohere Cloud Embedding
  response = co.embed(
    list, #list generated from function json_to_list
    model='small', 
  )
  output_dict = json_output(response.embeddings, list)


