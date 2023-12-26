import cohere
import pandas as pd
import click
# from schemas import expected_input_schema, expected_output_schema
from llama_index.embeddings import HuggingFaceEmbedding
from catalog import Catalog

def process_data(file_name) -> pd.DataFrame:
    """
    Process an Excel file into a Pandas DataFrame

    Args:
        file_path (str): Path to the Excel file.
    """
    df = pd.read_excel(file_name)
    return df


def local_embedding(prompt) -> list:
    """
    Get text embeddings using HuggingFace BERT model

    Args:
        - prompt: prompt that need to be embedded
    """
    # Local embedding 
    # Load model from HuggingFace
    embed_model = HuggingFaceEmbedding(model_name="bert-base-uncased")
    prompt_embedding = embed_model.get_text_embedding(prompt)
    return prompt_embedding

# Cloud based embedding
# Cohere Key:
co = cohere.Client("51F5llrM1i81Mp4A2HJ0TpApR4FF9Lpn12Nc90pN")

def cloud_embedding(li) -> list:
  """
  Get text embeddings using Cohere
  """
  response = co.embed(
  li, #list generated from function json_to_list
  model='small')
  return response.embeddings[0]

def embedding_data(embedding_method, df) -> pd.DataFrame:
    """
    Store prompt embedding data into a new Excel file

    Args:
     - embedding_method: Local/Cloud/OpenAI Embedding
    """
    # Full prompt embedding
    for row in range(1,100):
        inp = [df.loc[row,"full_prompt"]]
        response = embedding_method(inp)
        df.loc[row,"full_prompt_embedding"] = str(response)
    # Edited Direction Embeddings:
    # creates original embedding for areas of the spreadsheet where edits weren't made
    original_input = [df.loc[1,"original_direction"]]
    original_response = embedding_method(original_input)
    original_embedding = str(original_response)
    for row in range(1,100):
        # if out of the edit window of directoin embeddings (1-33) use the original
        if row < 33:
            df.loc[row,"edited_direction_embedding"] = original_embedding
        else:
            inp = [df.loc[row,"edited_direction"]]
            response = embedding_method(inp)
            df.loc[row,"edited_direction_embedding"] = str(response)
    # Edited Action Embeddings:
    # creates original embedding for areas of the spreadsheet where edits weren't made
    original_input = [df.loc[1,"original_acting"]]
    original_response = embedding_method(original_input)
    original_embedding = str(original_response)
    for row in range(1,100):
        if row > 33 and row < 67:
            inp = [df.loc[row,"edited_acting"]]
            response = embedding_method(inp)
            df.loc[row,"edited_acting_embedding"] = str(response)
        else:
            df.loc[row,"edited_acting_embedding"] = original_embedding
    # Edited Cinematography Embeddings:
    # creates original embedding for areas of the spreadsheet where edits weren't made
    original_input = [df.loc[1,"original_cinematography"]]
    original_response = embedding_method(original_input)
    original_embedding = str(original_response)
    for row in range(1,100):
        if row > 66 and row < 100:
            inp = [df.loc[row,"edited_cinematography"]]
            response = embedding_method(inp)
            df.loc[row,"edited_cinematography_embedding"] = str(response)
        else:
            df.loc[row,"edited_cinematography_embedding"] = original_embedding
    # Store as "FINALOUTPUT + embedding_method.xlsx"
    df.to_excel(f"FINALOUTPUT_{embedding_method.__name__}.xlsx")
    return df

@click.command()
@click.option("--em", type=str, help="Embedding method (Local/Cloud/OpenAI)")
@click.option("--n", type=str, help="Path to prompt Excel file")

def main(em, n):
    catalog = Catalog()
    try:
        # Return a DataFrame after processing data
        df = catalog._process_excel(n)
        embedding_data(em, df)
        click.echo(f"Finished processing {n}")
    except Exception as e:
        click.echo(f"Error processing file: {str(e)}", err=True)


if __name__ == '__main__':
    main()
