import torch
import numpy as np
import pandas as pd
import altair as alt
import json
# Jaxtyping imports
from jaxtyping import Float, Int

# Define models, steps, and chosen terms
models = ["70M", "160M", "1.4B"]
steps = list(range(1000, 143000, 2000))
chosen_terms = ["communication.n.02", "measure.n.02", "set.n.02", "location.n.01", "solid.n.01"]

# Load the index-to-term mapping only once
with open("llm_ontology/data/index_to_term_pythia.json", "r") as f:
    index_to_term = json.load(f)
index_to_term = {int(k): v for k, v in index_to_term.items()}

# Filter indices for only the chosen terms
filtered_indices = [i for i, term in index_to_term.items() if term in chosen_terms]

all_scores = []

# Iterate over models and steps
for model in models:
    for step in steps:
        base_path = f"/mnt/bigstorage/raymond/heatmaps-pythia/{model}/{model}-step{step}"
        
        # Load tensors from disk
        adj: Float[torch.Tensor, "vocab_size embedding_dim"] = torch.load(f"{base_path}-1.pt", weights_only=False)
        cos: Float[torch.Tensor, "vocab_size embedding_dim"] = torch.load(f"{base_path}-2.pt", weights_only=False)
        hier: Float[torch.Tensor, "vocab_size embedding_dim"] = torch.load(f"{base_path}-3.pt", weights_only=False)
        linear: Float[np.ndarray, "vocab_size"] = torch.load(f"{base_path}-4.pt", weights_only=False).numpy()

        # Compute per-term scores for the chosen indices by slicing the tensors
        causal_sep_scores: Float[np.ndarray, "num_terms"] = np.sum((adj[filtered_indices] - cos[filtered_indices]) ** 2, axis=1)
        hierarchy_scores: Float[np.ndarray, "num_terms"] = np.sum((hier[filtered_indices] - cos[filtered_indices]) ** 2, axis=1)
        linear_rep_scores: Float[np.ndarray, "num_terms"] = linear[filtered_indices]  # Each value represents the score per term

        # Store scores in a DataFrame for the chosen terms only
        df_scores = pd.DataFrame({
            "index": filtered_indices,
            "term": [index_to_term[i] for i in filtered_indices],
            "causal_sep_score": causal_sep_scores,
            "hierarchy_score": hierarchy_scores,
            "linear_rep_score": linear_rep_scores,
            "step": step,
            "model": model
        })

        all_scores.append(df_scores)

# Combine all results into one DataFrame
df_all_scores = pd.concat(all_scores, ignore_index=True)

# Convert to long format for visualization
df_melted = df_all_scores.melt(
    id_vars=["step", "term", "model"], 
    value_vars=["causal_sep_score", "hierarchy_score", "linear_rep_score"],
    var_name="score_type", value_name="score"
)

# Create a chart with rows representing score types and columns representing models.
# Color now represents the different terms.
final_chart = alt.Chart(df_melted).mark_line().encode(
    x=alt.X('step:Q', title='Training Steps'),
    y=alt.Y('score:Q', title='Ontology Score'),
    color=alt.Color('term:N', title='Term'),
    tooltip=['step', 'score', 'term', 'score_type', 'model']
).facet(
    row=alt.Row('score_type:N', title='Score Type'),
    column=alt.Column('model:N', title='Model')
).resolve_scale(
    y='independent'
).interactive()

# Save the final chart to an HTML file
final_chart.save("plot_term_scores.html")