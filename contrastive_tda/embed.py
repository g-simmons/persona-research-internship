from typing import Iterable, List, Optional
import os
import cohere
from dotenv import load_dotenv
from llama_index.embeddings import HuggingFaceEmbedding


def init_cohere_api():
    load_dotenv()
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if cohere_api_key is None:
        raise ValueError("Cohere API key not found. Set COHERE_API_KEY in .env file.")
    co = cohere.Client(cohere_api_key)
    return co
    
def _embed_local(items: Iterable, embed_field: str, embedding_model: str) -> List:
    model = HuggingFaceEmbedding(model_name="bert-base-uncased")
    embeddings = [model.get_text_embedding(getattr(item, embed_field)) for item in items]
    return embeddings

def _embed_cloud(items: Iterable, embed_field: str) -> List[List[float]]:
    co = init_cohere_api()
    response = co.embed(texts=[getattr(item,embed_field) for item in items], model='small')
    return response.embeddings

def validate_embed_field(items: Iterable, embed_field: str):
    """
    Validate that all items have the given embed_field
    """
    if not all(hasattr(item, embed_field) for item in items):
        raise ValueError(
            f"Expected all items to have an attribute {embed_field}"
        )

def embed(items: Iterable, embed_field: str, embed_method: str, embed_model: Optional[str] = None) -> List:
    """
    Embed a list of items using the given embed_field

    Args:
        items (Iterable[BaseModel]): A list of items to embed
        embed_field (str): The name of the field to embed
        embed_model (str): The name of the embedding model to use (for local)
    """
    validate_embed_field(items, embed_field)
    
    if embed_method == "local":
        return _embed_local(items, embed_field, embed_model)
    
    elif embed_method == "cloud":
        return _embed_cloud(items, embed_field)
    
    else:
        raise ValueError(f"Invalid embed method {embed_method}")
    
