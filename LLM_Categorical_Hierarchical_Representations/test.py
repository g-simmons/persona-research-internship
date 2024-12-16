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

from ontology_scores import *

import altair as alt
import pandas as pd
from joblib import Parallel, delayed

# reproducability
torch.manual_seed(0)
np.random.seed(0)
import random
random.seed(0)

DO_IN_PARALLEL = True
parameter_models = ["12B"]
steps = [f"step{i}" for i in range(1000, 145000, 2000)]


save_wordnet_hypernym(parameter_models[0], steps[0], True)

def save_heatmaps(parameter_model, step, multi):
    mats = get_mats(parameter_model, step, multi)
    for mat in mats:
        print(mat.shape)

    torch.save(mats[0], f"/mnt/bigstorage/raymond/heatmaps/{parameter_model}/{parameter_model}-{step}-1.pt")
    torch.save(mats[1], f"/mnt/bigstorage/raymond/heatmaps/{parameter_model}/{parameter_model}-{step}-2.pt")
    torch.save(mats[2], f"/mnt/bigstorage/raymond/heatmaps/{parameter_model}/{parameter_model}-{step}-3.pt")




if (DO_IN_PARALLEL):
    for parameter_model in parameter_models: 
        results = Parallel(n_jobs=-8)(delayed(save_heatmaps)(parameter_model, step, True) for step in steps)
else:
    for parameter_model in parameter_models:
        results = []
        for step in steps:
            results.append(get_mats(parameter_model, step, True))

