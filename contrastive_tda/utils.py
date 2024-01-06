from pydantic import BaseModel
from typing import Dict, Type, Any
import pandas as pd

def pandas_dtypes_to_pydantic_types(df: pd.DataFrame) -> Dict[str, Type]:
    """
    Maps pandas DataFrame dtypes to corresponding Python types for Pydantic model.

    :param df: pandas DataFrame.
    :return: A dictionary mapping field names to Python types.
    """
    type_mapping = {
        'int64': int,
        'float64': float,
        'bool': bool,
        'object': str,  # Mapping for 'object' dtype, commonly used for strings
        # Add other pandas dtypes mappings if necessary
    }

    # Default case for any unexpected dtypes
    return {col: type_mapping.get(str(df[col].dtype), Any) for col in df.columns}

def create_pydantic_model(field_types: Dict[str, Type], model_name: str) -> tuple[Type[BaseModel], str]:
    """
    Dynamically creates a Pydantic model based on a dictionary mapping field names to types.
    """
    class_body = '\n    '.join(f"{name}: {typ.__name__}" for name, typ in field_types.items())
    class_def = f"class {model_name}(BaseModel):\n    {class_body}"
    local_namespace = {}
    exec(class_def, globals(), local_namespace)
    return local_namespace[model_name], class_def

def df_to_pydantic_model(df: pd.DataFrame, model_name: str) -> tuple[Type[BaseModel], str]:
    """
    Converts a pandas DataFrame to a Pydantic model.

    :param df: pandas DataFrame.
    :param model_name: Name of the Pydantic model.
    :return: A tuple containing the Pydantic model and its class definition.
    """
    field_types = pandas_dtypes_to_pydantic_types(df)
    model, class_def = create_pydantic_model(field_types,model_name)
    model.__name__ = model_name
    return model, class_def