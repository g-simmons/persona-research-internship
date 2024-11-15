#!/usr/bin/env python3


import torch
import numpy as np
import json
from transformers import AutoTokenizer
import networkx as nx
import matplotlib.pyplot as plt
import hierarchical as hrc
import warnings
warnings.filterwarnings('ignore')

import ontology_class


# run this AFTER running get_ontology_hypernym.py
# params and step needs to match the params and step in get_ontology_hypernym.py
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
    return len(cats)


dropoff_points = {}

temp = 100000   # arbitrary number
counter = 0
while temp > 1:
    temp = get_mat_dim("1.4B", "step99000", counter)
    counter += 1
    print("DIM:" + str(temp))
    dropoff_points[counter] = temp



# plot
x = list(dropoff_points.keys())
y = list(dropoff_points.values())

plt.plot(x, y)

plt.title('Synset Dropoff Rate')
plt.xlabel('Filter')
plt.ylabel('Heatmap Dimension')

plt.savefig('dropoffTEST.png')