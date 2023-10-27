import cohere
import pandas 
import json

# Cloud based embedding

# Cohere Key:
co = cohere.Client("51F5llrM1i81Mp4A2HJ0TpApR4FF9Lpn12Nc90pN")


# Find Json from input path
def json_input_directory() -> dict:
    # directory/folder path
    dir_path = input("Enter directory for Json file: ")
    raw_dir_path = r'{}'.format(dir_path)
    # puts Json Data in varaible
    dictionary = pandas.read_json(raw_dir_path)

    
    return dictionary

# Creates json for output
def json_output(list):
    dict = {}
    for i in range(0, len(list)):
      string = "Prompt: " + str(i)
      dict.update({string:list[i]})
    
#method using import Json
    # # Serializing json
    # json_object = json.dumps(dict, indent=4)
    
    # # Writing to sample.json
    # with open("sample.json", "w") as outfile:
    #     outfile.write(json_object)


# Method using import Panda
    # Convert the list to a DataFrame
      df = pandas.DataFrame(dict)

      # Specify the output file name
      output_file = "output.json"

      # Save the DataFrame as a JSON file
      df.to_json(output_file, orient='records', lines=True)
        
   

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


# Main ----------------------------------------------------------------------------------------

dict = json_input_directory()
list = json_to_list(dict)

# Cohere Cloud Embedding
response = co.embed(
  list, #list generated from function json_to_list
  model='small', 
)

json_output(response.embeddings)



https://github.com/g-simmons/persona-research-internship.git