#!/usr/bin/env python3

from obo_ontology_heatmaps import *
import matplotlib.pyplot as plt
import json
import torch
import ontology_class


def save_scatterplot(adj: torch.Tensor, cos: torch.Tensor, row_terms_txt_dir: str, ontology: ontology_class.Onto):
    size = cos.shape
    with open(row_terms_txt_dir, "r") as f:
        row_terms = f.readlines()
        row_terms = list(map(lambda x: x[:-1], row_terms))
    

    # 0_diag Hadamard product equivalent
    for i in range(size[0]):
        cos[i][i] = 0
    
    # term scores calculation
    depth = []
    scores = []
    for i in range(size[0]):
        diff = cos[i] - adj[i]
        depth.append(ontology.get_synset(row_terms[i]).get_depth())
        scores.append(np.linalg.norm(diff))
        print('adfjlhk')


    # scatterplot
    plt.scatter(depth, scores)
    plt.xlabel("Term Depth")
    plt.ylabel("Term Causal Separability Score")
    plt.savefig("scatterplotDEPTHTEST.png")

    plt.clf()


save_ontology_hypernym("70M", "step99000", "owl/cl.owl")
mats = get_mats("70M", "step99000")

adj = mats[0]
cos = mats[1]

save_scatterplot(adj, cos, "cl_row_terms.txt", ontology_class.Onto("owl/cl.owl"))