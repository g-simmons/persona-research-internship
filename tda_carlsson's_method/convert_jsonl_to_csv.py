# In this file, the dvc data files that are in jsonl format are converted to csv

import pandas as pd

file_path = '/Users/riab/persona-research-internship/data/llm_edited_reviews/llm_edited_reviews.jsonl'
df = pd.read_json(file_path, lines = True)
embeddings = df["embeddings"]
pd.DataFrame(embeddings).to_csv('llm_edited_reviews.csv')