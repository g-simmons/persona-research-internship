#!/bin/bash

# Create target directory
mkdir -p llm_persona_website/model_score_plots_nonmulti

# Array of source and destination files
declare -a files=(
    "causal_sep_scores.html"
    "hierarchy_scores.html"
    "linear_rep_scores.html"
    "combined_scores.html"
)

# Copy each file
for file in "${files[@]}"; do
    src="LLM_Categorical_Hierarchical_Representations/model_score_plots_nonmulti/$file"
    dst="llm_persona_website/model_score_plots_nonmulti/$file"
    
    if [ -f "$src" ]; then
        cp "$src" "$dst"
        echo "Successfully copied $src to $dst"
    else
        echo "Error: Could not find $src"
    fi
done 