#!/usr/bin/env python3

# import nltk
# nltk.download('wordnet')

import json
import networkx as nx
# from nltk.corpus import wordnet as wn
from transformers import AutoTokenizer

import inflect

import huggingface_hub
huggingface_hub.login(token="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")

import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import hierarchical as hrc
import warnings
warnings.filterwarnings('ignore')

import ontology_class

def save_ontology_hypernym(params: str, step: str, ontology_dir: str):
    # model_name = "gemma-2b"
    # tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    model_name = "pythia"
    
    tokenizer = AutoTokenizer.from_pretrained(
        f"EleutherAI/pythia-{params}-deduped",
        revision=f"{step}",
        cache_dir=f"/mnt/bigstorage/raymond/huggingface_cache/pythia-{params}-deduped/{step}",
    )

    vocab = tokenizer.get_vocab()
    vocab_set = set(vocab.keys())
    vocab_set = set(map(lambda x: x.replace(" ", "_"), vocab_set))

    p = inflect.engine()

    def get_all_hyponym_lemmas(synset):
        hyponyms = synset.hyponyms()
        lemmas = set()
        for hyponym in hyponyms:
            lemmas.update(lemma.name() for lemma in hyponym.lemmas())
            lemmas.update(get_all_hyponym_lemmas(hyponym))  # Recursively get lemmas from hyponyms,
        
        return lemmas


    # all_noun_synsets = list(wn.all_synsets(pos=wn.NOUN))
    test = ontology_class.Onto(ontology_dir)
    all_noun_synsets = test.all_synsets()
    print(len(all_noun_synsets))
    noun_lemmas = {}
    for s in all_noun_synsets:
        lemmas = get_all_hyponym_lemmas(s)
        # add and remove space bc of how gemma vocab works
        if model_name == "gemma-2b":
            lemmas = vocab_set.intersection({"▁" + l for l in lemmas})
            # lemmas = vocab_set.intersection({l for l in lemmas})
            noun_lemmas[s.name()] = {l[1:] for l in lemmas}
        elif model_name == "pythia":
            lemmas = vocab_set.intersection({l for l in lemmas})
            noun_lemmas[s.name()] = {l for l in lemmas}
            
    # print(len(noun_lemmas))
    # for k, v in noun_lemmas.items():
    #     print(k, v)
    large_nouns = {k: v for k, v in noun_lemmas.items() if len(v) > 1}


    print(len(all_noun_synsets))
    print(len(large_nouns))


    # Construct the hypernym inclusion graph among large categories
    G_noun = nx.DiGraph()

    nodes = list(large_nouns.keys())
    for key in nodes:
        # print("key:" + key)
        # for path in wn.synset(key).hypernym_paths():
        for path in test.get_synset(key).hypernym_paths():
            # ancestors included in the cleaned set
            ancestors = [s.name() for s in path if s.name() in nodes]

            if len(ancestors) > 1:
                G_noun.add_edge(ancestors[-2],key) # first entry is itself
            else:
                print(f"no ancestors for {key}")



    # print(large_nouns)

    # if a node has only one child, and that child has only one parent, merge the two nodes
    def merge_nodes(G, lemma_dict):
        topological_sorted_nodes = list(reversed(list(nx.topological_sort(G))))

        for node in topological_sorted_nodes:
            children = list(G.successors(node))
            if len(children) == 1:
                child = children[0]
                parent_lemmas_not_in_child = lemma_dict[node] - lemma_dict[child]
                if len(list(G.predecessors(child))) == 1 or len(parent_lemmas_not_in_child) <6:
                    grandchildren = list(G.successors(child))
                    
                    if len(parent_lemmas_not_in_child) > 1:
                        if len(grandchildren) > 0:
                            lemma_dict[node + '.other'] = parent_lemmas_not_in_child
                            G.add_edge(node, node + '.other')

                    # del synset_lemmas[child]
                    for grandchild in grandchildren:
                        G.add_edge(node, grandchild)
                    G.remove_node(child)
                    print(f"merged {node} and {child}")

    # print(G_noun.nodes())
    merge_nodes(G_noun, large_nouns)
    large_nouns = {k: v for k, v in large_nouns.items() if k in G_noun.nodes()}


    G_noun = nx.DiGraph(G_noun.subgraph(nodes))


    # make a gemma specific version
    def _noun_to_gemma_vocab_elements(word):
        word = word.lower()
        plural = p.plural(word)

        add_cap_and_plural = [word, word.capitalize(), plural, plural.capitalize()]
        # add_space = ["▁" + w for w in add_cap_and_plural]
        add_space = [w for w in add_cap_and_plural]
        return vocab_set.intersection(add_space)

    ## save the data
    with open(f'data/ontologies/noun_synsets_ontology_pythia.json', 'w') as f:
        prev_pythia_words = []
        corr_synsets = []
        for synset, lemmas in large_nouns.items():
            print(G_noun.nodes())
            pythia_words = []
            for w in lemmas:
                pythia_words.extend(_noun_to_gemma_vocab_elements(w))

            if pythia_words in prev_pythia_words:
                print(f"repeated: {synset}")
            prev_pythia_words.append(pythia_words)
            corr_synsets.append(synset)

            f.write(json.dumps({synset: pythia_words}) + "\n")
            
    nx.write_adjlist(G_noun, f"data/ontologies/noun_synsets_ontology_hypernym_graph.adjlist")


save_ontology_hypernym("70M", "step13000", "owl/aism.owl")

