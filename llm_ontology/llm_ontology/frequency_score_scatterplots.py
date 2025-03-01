#!/usr/bin/env python3

import json
import logging
import pathlib
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from utils import savefig, figname_from_fig_metadata
from ontology_scores import causal_sep_score_simple as causal_sep_score

# Set up logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Global paths
BIGSTORAGE_DIR = pathlib.Path("/mnt/bigstorage")

def save_scatterplot(adj: torch.Tensor | np.ndarray, cos: torch.Tensor | np.ndarray, row_terms_path: pathlib.Path, term_freq_path: pathlib.Path) -> None:
    """Create and save a scatterplot comparing term frequencies to causal separability scores.
    
    Args:
        adj: Adjacency matrix tensor
        cos: Cosine similarity matrix tensor 
        row_terms_path: Path to file containing row terms
        term_freq_path: Path to JSON file containing term frequencies
    """
    # Load terms and frequencies
    with open(row_terms_path, "r") as f:
        row_terms = [line.strip() for line in f.readlines()]
    
    with open(term_freq_path, 'r') as f:
        term_freq = json.load(f)

    # Calculate causal separability score
    score = causal_sep_score(adj, cos)
    
    # Map frequencies to scores
    scores = {}  # freq: score
    for i, term in enumerate(row_terms):
        diff = cos[i] - adj[i]
        scores[term_freq[term]] = float(np.linalg.norm(diff))

    # Create scatterplot
    freqs = list(scores.keys())
    term_scores = list(scores.values())
    
    # Set up the plot style
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Create scatterplot with seaborn
    sns.scatterplot(x=freqs, y=term_scores, alpha=0.6)
    plt.xlabel("Pretraining Term Frequency")
    plt.ylabel("Term Causal Separability Score")

    # Save plot
    metadata = {
        "type": "frequency_score_scatterplot",
        "title": "Term Frequency vs Causal Separability Score"
    }
    figure_name = figname_from_fig_metadata(metadata)
    
    script_dir = pathlib.Path(__file__).parent
    figures_dir = script_dir.parent / "figures"
    
    savefig(
        fig=plt.gcf(),
        figures_dir=figures_dir,
        figure_name=figure_name,
        formats=["png"],
        overwrite=True
    )
    plt.clf()

def main() -> None:
    param_model = "160M"
    step = "step143000"

    adj = torch.load(str(BIGSTORAGE_DIR / f"raymond/heatmaps-pythia/{param_model}/{param_model}-{step}-1.pt"), weights_only=False)
    cos = torch.load(str(BIGSTORAGE_DIR / f"raymond/heatmaps-pythia/{param_model}/{param_model}-{step}-2.pt"), weights_only=False)

    script_dir = pathlib.Path(__file__).parent
    row_terms_path = script_dir / "wordnet_row_terms.txt"
    term_freq_path = script_dir / "term_frequencies/wordnet-frequencies.json"

    save_scatterplot(adj, cos, row_terms_path, term_freq_path)

if __name__ == "__main__":
    main()
