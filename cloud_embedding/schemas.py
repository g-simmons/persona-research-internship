
import pandas 
import pandera

# Schemas for Embeddings: 



# Sets up the wanted schema for the input "test.jsonl"
# Only checks for strings inside the list
expected_input_schema = pandera.DataFrameSchema(
   {
    "prompt": pandera.Column(str,nullable=False)
   }
)
# Sets up the wanted schema for the output "output.jsonl"

expected_output_schema = pandera.DataFrameSchema(
   {
    "prompt": pandera.Column(str, nullable=False),
    "prompt_embedding": pandera.Column(list[float], nullable=False)
   }
)

