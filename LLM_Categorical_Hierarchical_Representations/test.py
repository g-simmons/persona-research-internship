#!/usr/bin/env python3


import json
import ast


term_freq = {}
with open('term_frequencies/wordnet-frequencies.json', 'r') as f:
    term_freq = json.load(f)

wordnet_freq = term_freq["wordnet.txt"]

# print(wordnet_freq["dog"])



import torch
from ontology_scores import *


steps = [f"step{i}" for i in range(1000, 145000, 2000)]
for step in steps:
    adj = torch.load(f"/mnt/bigstorage/raymond/heatmaps/160M/160M-{step}-1.pt")
    cos = torch.load(f"/mnt/bigstorage/raymond/heatmaps/160M/160M-{step}-2.pt")
    diff = torch.load(f"/mnt/bigstorage/raymond/heatmaps/160M/160M-{step}-3.pt")

    print(causal_sep_score(adj, cos))
    # print(hierarchy_score(diff))

# step = "step143000"

# adj = torch.load(f"/mnt/bigstorage/raymond/heatmaps/1.4B/1.4B-{step}-1.pt")
# cos = torch.load(f"/mnt/bigstorage/raymond/heatmaps/1.4B/1.4B-{step}-2.pt")
# diff = torch.load(f"/mnt/bigstorage/raymond/heatmaps/1.4B/1.4B-{step}-3.pt")


