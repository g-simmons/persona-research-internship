#!/usr/bin/env python3

import logging
logger = logging.getLogger(__name__)

# Set up file handler
file_handler = logging.FileHandler('obo_depth_scatterplots.log')
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Set up stdout handler 
stdout_handler = logging.StreamHandler()
stdout_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stdout_handler.setFormatter(stdout_formatter)
logger.addHandler(stdout_handler)

# Set log level
logger.setLevel(logging.INFO)

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

from utils import savefig
import argparse
from pathlib import Path

BIGSTORAGE_DIR = Path("/mnt/bigstorage")

# returns the synsets that make it through the filter, in order of heatmap row
def get_mat_dim(ontology_name: str, params: str, step: str, filter: int):
    device = torch.device("cpu")
    # tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    tokenizer = AutoTokenizer.from_pretrained(
        f"EleutherAI/pythia-{params}-deduped",
        revision=f"{step}",
        cache_dir=BIGSTORAGE_DIR / f"raymond/huggingface_cache/pythia-{params}-deduped/{step}"
    )

    g = torch.load(BIGSTORAGE_DIR / f'raymond/pythia/{params}-unembeddings/{step}').to(device) # 'FILE_PATH' in store_matrices.py

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


    cats, G, sorted_keys = hrc.get_categories_ontology(ontology_name, filter)
    return cats

# saves row terms to text file in `owl_row_terms` directory
def save_row_terms(ontology_name: str, params: str, step: str, filter: int, multi: bool):
    # ontology_name = ontology_dir.split("/")[-1][:-4]

    obo_ontology_heatmaps.save_ontology_hypernym(params, step, ontology_name, multi)
    cats = get_mat_dim(ontology_name, params, step, filter)  #50
    if len(cats) > 30:      #number of rows(number of synsets that make it past the filter), if greater than 30 synsets, then save the row terms
        with open(f"owl_row_terms/{ontology_name}_row_terms.txt", "w") as f:
            for term in list(cats.keys()):
                f.write(term + "\n")


# based off the row terms and heatmaps, returns the depth, scores, terms, and term classes
def get_data(adj: torch.Tensor, cos: torch.Tensor, hier: torch.Tensor, row_terms_txt_dir: str, ontology: ontology_class.Onto, score: str):
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
        hier[i][i] = 0
    
    # term scores calculation
    depth = []
    scores = []
    terms = []
    term_classes = []
    # freqs = []
    for i in range(size[0]):
        if score == "sep":
            diff = cos[i] - adj[i]
            scores.append(np.linalg.norm(diff))
        elif score == "hier":
            scores.append(np.linalg.norm(hier[i]))

        depth.append(ontology.get_synset(row_terms[i]).get_depth())
        
        terms.append(row_terms[i])
        term_classes.append(ontology.get_synset(row_terms[i]).get_ontology_class())
        # freqs.append(term_freq[row_terms[i]])

    return depth, scores, terms, term_classes

# saves a depth scatterplot, you need to have already saved the row terms
def save_depth_scatterplot(ontology_name: str, params: str, step: str, filter: int, multi: bool, score: str):
    ontology_dir = BIGSTORAGE_DIR / f"raymond/owl/{ontology_name}.owl"

    term_txt_dir = "owl_row_terms/" + ontology_name + "_row_terms.txt"

    obo_ontology_heatmaps.save_ontology_hypernym(params, step, ontology_name, multi)
    mats = obo_ontology_heatmaps.get_mats(params, step, multi, filter, ontology_name)
    adj = mats[0]
    cos = mats[1]
    hier = mats[2]

    depth, scores, terms, term_classes = get_data(adj, cos, hier, term_txt_dir, ontology_class.Onto(ontology_dir), score)
    df = pd.DataFrame({'Depth': depth, 'Score': scores, 'Term': terms, "Term Class": term_classes})

    logger.info(df)

    vis_idea_1 = {
        "x": "Depth",
        "y": "Score",
        "color": "Term Class",
        "tooltip": ["Term", "Depth", "Score", "Term Class"]
    }
    # TODO define other vis ideas
    vis_ideas = [vis_idea_1]

    for vis_idea in vis_ideas:

        chart = alt.Chart(df).mark_circle(size=60).encode(**vis_idea).interactive()
        # TODO uniquely name the file associated with the vis idea
        # TODO extend savefig from https://github.com/g-simmons/persona-research-internship/issues/230 function to handle altair charts
        # TODO call savefig with the chart and filename
        chart.save(f"figures/depth_scatterplots_3_html/{ontology_name}_depth_scatterplot.html")
        chart.save(f"figures/depth_scatterplots_3_png/{ontology_name}_depth_scatterplot.png")



if __name__ == "__main__":
    # # SETTINGS
    # params = "70M"          # chosen because it is the smallest model size, so it is the fastest to run 
    # step = "step143000"     # chosen because it is the latest step, and therefore will have the "best" term scores
    # filter = 15              # is the threshold for how many lemmas a synset must have to be part of the heatmap
    # multi = True           # whether or not to include multi-word lemmas.


    # # ontology_dir = "/mnt/bigstorage/raymond/owl/cl.owl"
    # ontology_name = "aism"

    # save_row_terms(ontology_name, params, step, filter, multi)   # only need to run this if you choose new settings, otherwise they are already saved into `owl_row_terms` directory
    # save_depth_scatterplot(ontology_name, params, step, filter, multi, "sep")


    parser = argparse.ArgumentParser(description='Generate depth scatterplots for ontology terms')
    parser.add_argument('--params', type=str, default='70M',
                        help='Model size parameter (default: 70M)')
    parser.add_argument('--step', type=str, default='step143000',
                        help='Training step (default: step143000)') 
    parser.add_argument('--filter', type=int, default=15,
                        help='Minimum number of lemmas threshold (default: 15)')
    parser.add_argument('--multi', type=bool, default=True,
                        help='Whether to include multi-word lemmas (default: True)')
    parser.add_argument('--ontology', type=str, default='aism',
                        help='Name of ontology (default: aism)')
    parser.add_argument('--score', type=str, default='sep',
                        help='Score type to plot (default: sep)')

    args = parser.parse_args()

    logger.info(args)
    save_row_terms(args.ontology, args.params, args.step, args.filter, args.multi)
    save_depth_scatterplot(args.ontology, args.params, args.step, args.filter, args.multi, args.score)