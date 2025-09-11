import torch
import numpy as np
import pandas as pd
import altair as alt
import json

# Define models and steps
models = ["70M", "160M", "1.4B"]
steps = list(range(1000, 143000, 2000))

all_scores = []

# Load index-to-term mapping once outside the loop
with open("llm_ontology/data/index_to_term_pythia.json", "r") as f:
    index_to_term = json.load(f)
    index_to_term = {int(k): v for k, v in index_to_term.items()}

# Instead of chosen_terms, we use all terms
all_indices = list(index_to_term.keys())

# Iterate over models and steps
for model in models:
    for step in steps:
        base_path = f"/mnt/bigstorage/raymond/heatmaps-pythia/{model}/{model}-step{step}"
        
        adj = torch.load(f"{base_path}-1.pt", weights_only=False)
        cos = torch.load(f"{base_path}-2.pt", weights_only=False)
        hier = torch.load(f"{base_path}-3.pt", weights_only=False)
        linear = torch.load(f"{base_path}-4.pt", weights_only=False).numpy()

        # Compute per-term scores (sum over columns)
        causal_sep_scores = np.sum((adj - cos) ** 2, axis=1)
        hierarchy_scores = np.sum((hier - cos) ** 2, axis=1)
        linear_rep_scores = linear  # Each value directly represents the score per term
        
        # Create a DataFrame for all terms
        df_scores = pd.DataFrame({
            "index": all_indices,
            "term": [index_to_term[i] for i in all_indices],
            "causal_sep_score": [causal_sep_scores[i] for i in all_indices],
            "hierarchy_score": [hierarchy_scores[i] for i in all_indices],
            "linear_rep_score": [linear_rep_scores[i] for i in all_indices],
            "step": step,
            "model": model
        })
        
        all_scores.append(df_scores)

# Combine all results into one DataFrame
df_all_scores = pd.concat(all_scores, ignore_index=True)

# Save the DataFrame to a CSV file (adjust the file path as needed)
output_path = "llm_ontology/data/term_scores_all.csv"
df_all_scores.to_csv(output_path, index=False)
print(f"Term scores saved to {output_path}")
