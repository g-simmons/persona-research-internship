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

import argparse

def save_heatmaps(parameter_model, step, multi):
    mats = get_mats(parameter_model, step, multi)
    for mat in mats:
        print(mat.shape)

    torch.save(mats[0], f"/mnt/bigstorage/raymond/heatmaps/{parameter_model}/{parameter_model}-{step}-1.pt")
    torch.save(mats[1], f"/mnt/bigstorage/raymond/heatmaps/{parameter_model}/{parameter_model}-{step}-2.pt")
    torch.save(mats[2], f"/mnt/bigstorage/raymond/heatmaps/{parameter_model}/{parameter_model}-{step}-3.pt")


def save_linear_rep(parameter_model, step, multi):
    stuff = get_linear_rep(parameter_model, step, True)
    print(stuff)
    print(len(stuff))
    stuff_tens = torch.tensor(stuff)
    torch.save(stuff_tens, f"/mnt/bigstorage/raymond/heatmaps/{parameter_model}/{parameter_model}-{step}-4.pt")


if __name__ == "__main__":
    steps = [f"step{i}" for i in range(1000, 145000, 2000)]

    parser = argparse.ArgumentParser(description='Generate and save heatmaps or linear representations')
    parser.add_argument('--parallel', action='store_true', default=True,
                        help='Whether to run in parallel (default: True)')
    parser.add_argument('--model', type=str, default='12B',
                        help='Model size parameter (default: 12B)')
    parser.add_argument('--multi', action='store_true', default=True,
                        help='Whether to include multi-word terms (default: True)')
    parser.add_argument('--type', type=str, choices=['heatmaps', 'linear'], default='heatmaps',
                        help='Type of data to save (default: heatmaps)')
    parser.add_argument('--jobs', type=int, default=10,
                        help='Number of parallel jobs (default: 10)')

    args = parser.parse_args()

    save_wordnet_hypernym(args.model, steps[0], args.multi)
    print(steps)
    
    if args.parallel:
        if args.type == 'heatmaps':
            Parallel(n_jobs=args.jobs)(delayed(save_heatmaps)(args.model, step, args.multi) for step in steps)
        elif args.type == 'linear':
            Parallel(n_jobs=args.jobs)(delayed(save_linear_rep)(args.model, step, args.multi) for step in steps)
    else:
        if args.type == 'heatmaps':
            for step in steps:
                save_heatmaps(args.model, step, args.multi)
        elif args.type == 'linear':
            for step in steps:
                save_linear_rep(args.model, step, args.multi)
