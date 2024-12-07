#!/usr/bin/env python3


import os
import json
import networkx as nx
# from nltk.corpus import wordnet as wn
from transformers import AutoTokenizer

import inflect

import huggingface_hub

import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import hierarchical as hrc
import warnings
warnings.filterwarnings('ignore')

import ontology_class
import obo_ontology_heatmaps

import altair as alt
import pandas as pd



# returns the synsets that make it through the filter, in order of heatmap row
def get_mat_dim(params: str, step: str, filter: int):
    device = torch.device("cpu")
    # tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    tokenizer = AutoTokenizer.from_pretrained(
        f"EleutherAI/pythia-{params}-deduped",
        revision=f"{step}",
        cache_dir=f"/mnt/bigstorage/raymond/huggingface_cache/pythia-{params}-deduped/{step}"
    )

    g = torch.load(f'/mnt/bigstorage/raymond/{params}-unembeddings/{step}').to(device) # 'FILE_PATH' in store_matrices.py

    # g = torch.load('FILE_PATH').to(device)

    vocab_dict = tokenizer.get_vocab()
    new_vocab_dict = {}
    for key, value in vocab_dict.items():
        new_key = key.replace(" ", "_")
        new_vocab_dict[new_key] = value
    vocab_dict = new_vocab_dict

    vocab_list = [None] * (max(vocab_dict.values()) + 1)
    for word, index in vocab_dict.items():
        vocab_list[index] = word


    cats, G, sorted_keys = hrc.get_categories_ontology(filter)
    return cats

# saves row terms to text file in `owl_row_terms` directory
def save_row_terms(ontology_dir: str, params: str, step: str, filter: int, multi: bool):
    ontology_name = ontology_dir.split("/")[-1][:-4]

    obo_ontology_heatmaps.save_ontology_hypernym(params, step, ontology_dir, multi)
    cats = get_mat_dim(params, step, filter)  #50
    if len(cats) > 30:      #number of rows(number of synsets that make it past the filter)
        with open(f"owl_row_terms/{ontology_name}_row_terms.txt", "w") as f:
            for term in list(cats.keys()):
                f.write(term + "\n")


# based off the row terms and heatmaps, returns the depth, scores, terms, and term classes
def get_data(adj: torch.Tensor, cos: torch.Tensor, row_terms_txt_dir: str, ontology: ontology_class.Onto):
    size = cos.shape
    with open(row_terms_txt_dir, "r") as f:
        row_terms = f.readlines()
        row_terms = list(map(lambda x: x[:-1], row_terms))

    # term_freq = {}
    # with open(term_freq_json_dir, 'r') as f:
    #     term_freq = json.load(f)
    

    # 0_diag Hadamard product equivalent
    for i in range(size[0]):
        cos[i][i] = 0
    
    # term scores calculation
    depth = []
    scores = []
    terms = []
    term_classes = []
    # freqs = []
    for i in range(size[0]):
        diff = cos[i] - adj[i]
        depth.append(ontology.get_synset(row_terms[i]).get_depth())
        scores.append(np.linalg.norm(diff))
        terms.append(row_terms[i])
        term_classes.append(ontology.get_synset(row_terms[i]).get_ontology_class())
        # freqs.append(term_freq[row_terms[i]])

    return depth, scores, terms, term_classes

# saves a depth scatterplot, you need to have already saved the row terms
def save_depth_scatterplot(ontology_dir: str, params, step, filter, multi):
    ontology_name = ontology_dir.split("/")[-1][:-4]
    term_txt_dir = "owl_row_terms/" + ontology_name + "_row_terms.txt"

    obo_ontology_heatmaps.save_ontology_hypernym(params, step, ontology_dir, multi)
    mats = obo_ontology_heatmaps.get_mats(params, step, multi, filter)
    adj = mats[0]
    cos = mats[1]

    depth, scores, terms, term_classes = get_data(adj, cos, term_txt_dir, ontology_class.Onto(ontology_dir))
    df = pd.DataFrame({'Depth': depth, 'Score': scores, 'Term': terms, "Term Class": term_classes})

    print(df)

    chart = alt.Chart(df).mark_circle(size=60).encode(
        x='Depth',
        y='Score',
        color = 'Term Class',
        tooltip=['Term', 'Depth', 'Score', 'Term Class']
    ).interactive()

    chart.save(f"depth_scatterplots_3_html/{ontology_name}_depth_scatterplot.html")
    chart.save(f"depth_scatterplots_3_png/{ontology_name}_depth_scatterplot.png")



if __name__ == "__main__":
    # SETTINGS
    params = "70M"          # chosen because it is the smallest model size, so it is the fastest to run 
    step = "step143000"     # chosen because it is the latest step, and therefore will have the "best" term scores
    filter = 1              # is the threshold for how many lemmas a synset must have to be part of the heatmap
    multi = False           # whether or not to include multi-word lemmas.


    ontology_dir = "/mnt/bigstorage/raymond/owl/aism.owl"

    # save_row_terms(ontology_dir, params, step, filter, multi)   # only need to run this if you choose new settings, otherwise they are already saved into `owl_row_terms` directory
    save_depth_scatterplot(ontology_dir, params, step, filter, multi)

