#!/usr/bin/env python3

import json
import logging
import pathlib
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
#from utils import savefig, figname_from_fig_metadata
from utils import savefig, figname_from_fig_metadata
#from .ontology_scores import causal_sep_score_simple as causal_sep_score

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

def save_scatterplot(adj: torch.Tensor | np.ndarray, cos: torch.Tensor | np.ndarray, row_terms_path: pathlib.Path, term_freq_path: pathlib.Path, model_name: str = "", param_model: str = "") -> None:
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
        outer_dict = json.load(f)
        term_freq = outer_dict[list(outer_dict.keys())[0]] #overly complicated, may just modify the structure of the frequencies.json files

    # Calculate causal separability score
    #score = causal_sep_score(adj, cos)
    
    # Map frequencies to scores
    scores = {}  # freq: score
    for i, term in enumerate(row_terms):
        diff = cos[i] - adj[i]
        #print(term_freq)
        #print(row_terms)
        scores[term_freq[term]] = float(np.linalg.norm(diff))

    # Create scatterplot
    freqs = list(scores.keys())
    term_scores = list(scores.values())
    
    # Set up the plot style
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Create scatterplot with seaborn
    sns.scatterplot(x=freqs, y=term_scores, alpha=0.6)
    plt.xscale('log')
    plt.xlabel("Pretraining Term Frequency")
    plt.ylabel("Term Causal Separability Score")

    # Save plot
    metadata = {
        "type": "frequency_score_scatterplot",
        "title": "Term Frequency vs Causal Separability Score - {model_name}"
    }
    #figure_name = figname_from_fig_metadata(metadata)
    figure_name = f"frequency_score_scatterplot_{model_name}_{param_model}.png"
    
    script_dir = pathlib.Path(__file__).parent
    figures_dir = script_dir.parent / "figures"
    
    savefig(
        fig=plt.gcf(),
        figures_dir=figures_dir,
        figure_name=figure_name,
        formats=["png"]
    )
    plt.clf()

def main() -> None:
    param_model = "2.8B"
    param_model_olmo = "7B"
    step = "step143000"
    step_olmo = "step150000-tokens664B"
    model_name = "pythia"
    adj_p = torch.load(str(BIGSTORAGE_DIR / f"raymond/heatmaps-{model_name}/{param_model}/{param_model}-{step}-1.pt"), weights_only=False)
    cos_p = torch.load(str(BIGSTORAGE_DIR / f"raymond/heatmaps-{model_name}/{param_model}/{param_model}-{step}-2.pt"), weights_only=False)
    model_name = "olmo"
    adj_o = torch.load(str(BIGSTORAGE_DIR / f"raymond/heatmaps-{model_name}/{param_model_olmo}/{step_olmo}-1.pt"), weights_only=False)
    cos_o = torch.load(str(BIGSTORAGE_DIR / f"raymond/heatmaps-{model_name}/{param_model_olmo}/{step_olmo}-2.pt"), weights_only=False)

    script_dir = pathlib.Path(__file__).parent.parent / "data/"
    row_terms_path = script_dir / "owl_row_terms/wordnet_row_terms.txt"
    term_freq_path = script_dir / "term_frequencies/wordnet.txt-frequencies.json"

    save_scatterplot(adj_p, cos_p, row_terms_path, term_freq_path, model_name="pythia", param_model=param_model)
    save_scatterplot(adj_o, cos_o, row_terms_path, term_freq_path, model_name="olmo", param_model=param_model_olmo)

if __name__ == "__main__":
    main()
