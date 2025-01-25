import cohere
from typing import Callable
import pandas as pd
import click
# from schemas import expected_input_schema, expected_output_schema
from llama_index.embeddings import HuggingFaceEmbedding
from catalog import Catalog

from enum import Enum

def process_data(file_name) -> pd.DataFrame:
    """
    Process an Excel file into a Pandas DataFrame

    Args:
        file_path (str): Path to the Excel file.
    """
    df = pd.read_excel(file_name)
    return df


def local_embedding(prompt:str) -> list:
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
# TODO move to .env file
co = cohere.Client("51F5llrM1i81Mp4A2HJ0TpApR4FF9Lpn12Nc90pN")

def cloud_embedding(li) -> list:
  """
  Get text embeddings using Cohere
  """
  response = co.embed(
  li, #list generated from function json_to_list
  model='small')
  return response.embeddings[0]

def get_embed_fn(embedding_method: str) -> Callable:
    embedding_methods = {
        'local': local_embedding,
        'cloud': cloud_embedding,
    }
    embedding_method = embedding_methods[embedding_method] 
    assert embedding_method is not None, "Embedding method not found"
    assert callable(embedding_method), "Embedding method is not callable"
    return embedding_method


def embed_movie_reviews(embedding_method: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Store prompt embedding data into a new Excel file

    Args:
     - embedding_method: Local/Cloud/OpenAI Embedding
    """
    # Get embedding method
    embed_fn = get_embed_fn(embedding_method) # type: Callable

    # Full prompt embedding
    for row in range(1,100):
        inp = [df.loc[row,"full_prompt"]]
        response = embed_fn(inp)
        df.loc[row,"full_prompt_embedding"] = str(response)
    # Edited Direction Embeddings:
    # creates original embedding for areas of the spreadsheet where edits weren't made
    original_input = [df.loc[1,"original_direction"]]
    original_response = embed_fn(original_input)
    original_embedding = str(original_response)
    for row in range(1,100):
        # if out of the edit window of directoin embeddings (1-33) use the original
        if row < 33:
            df.loc[row,"edited_direction_embedding"] = original_embedding
        else:
            inp = [df.loc[row,"edited_direction"]]
            response = embed_fn(inp)
            df.loc[row,"edited_direction_embedding"] = str(response)
    # Edited Action Embeddings:
    # creates original embedding for areas of the spreadsheet where edits weren't made
    original_input = [df.loc[1,"original_acting"]]
    original_response = embed_fn(original_input)
    original_embedding = str(original_response)
    for row in range(1,100):
        if row > 33 and row < 67:
            inp = [df.loc[row,"edited_acting"]]
            response = embed_fn(inp)
            df.loc[row,"edited_acting_embedding"] = str(response)
        else:
            df.loc[row,"edited_acting_embedding"] = original_embedding
    # Edited Cinematography Embeddings:
    # creates original embedding for areas of the spreadsheet where edits weren't made
    original_input = [df.loc[1,"original_cinematography"]]
    original_response = embed_fn(original_input)
    original_embedding = str(original_response)
    for row in range(1,100):
        if row > 66 and row < 100:
            inp = [df.loc[row,"edited_cinematography"]]
            response = embed_fn(inp)
            df.loc[row,"edited_cinematography_embedding"] = str(response)
        else:
            df.loc[row,"edited_cinematography_embedding"] = original_embedding

    return df

@click.command()
@click.option("--embedding_method", type=str, help="Embedding method (Local/Cloud/OpenAI)")
@click.option("--input_path", type=str, help="Path to prompt Excel file")
def main(embedding_method: str, input_path: str):
    catalog = Catalog()
    try:
        # Return a DataFrame after processing data
        df = catalog._process_excel(input_path)
        embed_movie_reviews(embedding_method, df)
        click.echo(f"Finished processing {input_path}")
    except Exception as e:
        click.echo(f"Error processing file: {str(e)}", err=True)


if __name__ == '__main__':
    main()
