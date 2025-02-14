#!/usr/bin/env python3

import torch
from ontology_scores import *
import matplotlib.pyplot as plt



def save_scatterplot(adj: torch.Tensor, cos: torch.Tensor, row_terms_txt_dir: str, term_freq_json_dir: str):
    size = cos.shape
    with open(row_terms_txt_dir, "r") as f:
        row_terms = f.readlines()
        row_terms = list(map(lambda x: x[:-1], row_terms))
    
    term_freq = {}
    with open(term_freq_json_dir, 'r') as f:
        term_freq = json.load(f)

    # 0_diag Hadamard product equivalent
    # TODO call causal separability score function here
    for i in range(size[0]):
        cos[i][i] = 0
    
    # term scores calculation
    scores = {}     # freq: score
    for i in range(size[0]):
        diff = cos[i] - adj[i]
        scores[term_freq[row_terms[i]]] = np.linalg.norm(diff)


    # scatterplot
    freqs = list(scores.keys())
    term_scores = list(scores.values())
    plt.scatter(freqs, term_scores)
    plt.xlabel("Pretraining Term Frequency")
    plt.ylabel("Term Causal Separability Score")

    # TODO get a filename from get_figname_from_fig_metadata
    # TODO call savefig with the chart and filename
    plt.savefig("scatterplotTEST.png")

    plt.clf()




param_model = "160M"
step = "step143000"

adj = torch.load(f"/mnt/bigstorage/raymond/heatmaps/{param_model}/{param_model}-{step}-1.pt")
cos = torch.load(f"/mnt/bigstorage/raymond/heatmaps/{param_model}/{param_model}-{step}-2.pt")

save_scatterplot(adj, cos, "wordnet_row_terms.txt", "term_frequencies/wordnet-frequencies.json")
