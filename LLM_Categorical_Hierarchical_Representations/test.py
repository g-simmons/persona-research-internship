#!/usr/bin/env python3

import os
import torch
import numpy as np
import json
from transformers import AutoTokenizer
import networkx as nx
import matplotlib.pyplot as plt
import hierarchical as hrc
import inflect
import warnings
warnings.filterwarnings('ignore')

import ontology_class
from obo_ontology_heatmaps import *

from ontology_scores import get_scores

import altair as alt
import pandas as pd
from joblib import Parallel, delayed

parameter_models = ["2.8B"]
steps = [f"step{i}" for i in range(1000, 145000, 2000)]
  
for parameter_model in parameter_models: 
    results = Parallel(n_jobs=-1)(delayed(get_scores)(parameter_model, step, True) for step in steps)

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

    


for term_txt in os.listdir("owl_row_terms"):
    save_ontology_hypernym("70M", "step143000", f"owl/{term_txt[:-14]}.owl", True)
    mats = get_mats("70M", "step143000", True, 30)
    adj = mats[0]
    cos = mats[1]

    depth, scores, terms, term_classes = get_data(adj, cos, f"owl_row_terms/{term_txt[:-14]}_row_terms.txt", ontology_class.Onto(f"owl/{term_txt[:-14]}.owl"))
    df = pd.DataFrame({'Depth': depth, 'Score': scores, 'Term': terms, "Term Class": term_classes})

    print(df)

    chart = alt.Chart(df).mark_circle(size=60).encode(
        x='Depth',
        y='Score',
        color = 'Term Class',
        tooltip=['Term', 'Depth', 'Score', 'Term Class']
    ).interactive()

    chart.save(f"depth_scatterplots_2_html/{term_txt[:-14]}_depth_scatterplot.html")
    chart.save(f"depth_scatterplots_2_png/{term_txt[:-14]}_depth_scatterplot.png")

