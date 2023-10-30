import cohere
import pandas 
import pandera


# Cloud based embedding

# Cohere Key:
co = cohere.Client("51F5llrM1i81Mp4A2HJ0TpApR4FF9Lpn12Nc90pN")

# Sets up the wanted schema for the input "test.json"
# Only checks for strings inside the list
expected_input_schema = pandera.DataFrameSchema(
   {
    "Prompts": pandera.Column(str)
   }
)
# Sets up the wanted schema for the output "output.jsonl"

expected_output_schema = pandera.DataFrameSchema(
   {
    "prompt": pandera.Column(str),
    "prompt_embedding": pandera.Column(list[float])
   }
)

# raises error flag if schema doesn't match 
def error_flag(dict,schema,str):
  try:
    schema.validate(dict, lazy=True)
  except pandera.errors.SchemaErrors as err:
    print(str)
    print(err)


#For some reason, we can't run a checker on the list[float] above... so we need this function
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
    dictionary = pandas.read_json(raw_dir_path)

    #Implement Panderas checking the "dictionary" var in this function

    return dictionary
   

# Put Json Data in list
def json_to_list(dict) -> list:
  # create list
  list = []
  # check if "Prompts" dictionary in Json exists
  if "Prompts" in dict:
      for value in dict["Prompts"]:
         # add to list to use for Cohere
         list.append(value)
  return list


# Creates json for output
def json_output(embedding_list, prompt_list) -> dict:
  #sets up the dictionary frame

  df = pandas.DataFrame({'prompt': prompt_list,'prompt_embedding': embedding_list})
  print(df)
  # Save the DataFrame to a JSONL file
  df.to_json('output.jsonl', orient='records', lines=True)
  
  return df


# Main ----------------------------------------------------------------------------------------


dict = json_input_directory()

error_flag(dict, expected_input_schema, "INPUT SCHEMA ERROR")

list = json_to_list(dict)

# Cohere Cloud Embedding
response = co.embed(
  list, #list generated from function json_to_list
  model='small', 
)

output_dict = json_output(response.embeddings, list)


error_flag(output_dict, expected_output_schema, "OUTPUT SCHEMA ERROR")











