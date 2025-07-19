#!/usr/bin/env python3

import json
import numpy as np
import torch
import networkx as nx
import tempfile
import ontology_class
import hierarchical as hrc
from transformers import AutoTokenizer
import inflect
import warnings
import pathlib
#import huggingface_hub
# Jaxtyping imports
from jaxtyping import Float, Int

warnings.filterwarnings('ignore')

# Global variable for BIGSTORAGE_DIR
BIGSTORAGE_DIR = pathlib.Path("/mnt/bigstorage")

# Find FIGURES_DIR and DATA_DIR relative to the script directory
SCRIPT_DIR = pathlib.Path(__file__).parent
FIGURES_DIR = SCRIPT_DIR.parent / "figures"
DATA_DIR = SCRIPT_DIR.parent / "data"

# Function from get_ontology_hypernym.py
def save_ontology_hypernym(params: str, step: str, ontology_dir: str):
    model_name = "pythia"
    
    tokenizer = AutoTokenizer.from_pretrained(
        f"EleutherAI/pythia-{params}-deduped",
        revision=f"{step}",
        cache_dir=BIGSTORAGE_DIR / "raymond" / "huggingface_cache" / f"pythia-{params}-deduped" / step,
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

    test = ontology_class.Onto(ontology_dir)
    all_noun_synsets = test.all_synsets()
    noun_lemmas = {}
    for s in all_noun_synsets:
        lemmas = get_all_hyponym_lemmas(s)
        if model_name == "gemma-2b":
            lemmas = vocab_set.intersection({"▁" + l for l in lemmas})
            noun_lemmas[s.name()] = {l[1:] for l in lemmas}
        elif model_name == "pythia":
            lemmas = vocab_set.intersection({l for l in lemmas})
            noun_lemmas[s.name()] = {l for l in lemmas}
            
    large_nouns = {k: v for k, v in noun_lemmas.items() if len(v) > 1}

    G_noun = nx.DiGraph()

    nodes = list(large_nouns.keys())
    for key in nodes:
        for path in test.get_synset(key).hypernym_paths():
            ancestors = [s.name() for s in path if s.name() in nodes]
            if len(ancestors) > 1:
                G_noun.add_edge(ancestors[-2], key)
            else:
                print(f"no ancestors for {key}")

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
    with open(FIGURES_DIR / 'ontologies' / 'noun_synsets_ontology_pythia.json', 'w') as f:
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
            
    nx.write_adjlist(G_noun, DATA_DIR / 'ontologies' / 'noun_synsets_ontology_hypernym_graph.adjlist')
# Function from 3_Noun_Heatmap
def generate_noun_heatmaps(params: str, step: str):
    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        f"EleutherAI/pythia-{params}-deduped",
        revision=f"{step}",
        cache_dir=BIGSTORAGE_DIR / "raymond" / "huggingface_cache" / f"pythia-{params}-deduped" / step
    )

    g: Float[torch.Tensor, "vocab_size embedding_dim"] = torch.load(BIGSTORAGE_DIR / "raymond" / f"{params}-unembeddings" / step).to(device)

    vocab_dict = tokenizer.get_vocab()
    new_vocab_dict = {key.replace(" ", "_"): value for key, value in vocab_dict.items()}
    vocab_list = [None] * (max(new_vocab_dict.values()) + 1)
    for word, index in new_vocab_dict.items():
        vocab_list[index] = word

    cats, G, sorted_keys = hrc.get_categories_ontology('noun')
    error_count, total_count = 0, 0
    messed_up, dirs = {}, {}
    for k, v in cats.items():
        total_count += 1
        try:
            dirs[k] = hrc.estimate_cat_dir(v, g, new_vocab_dict, False)
        except Exception as e:
            error_count += 1
            print(e)
            messed_up[k] = v

    tc_G = nx.algorithms.dag.transitive_closure(G)
    adj_mat = nx.adjacency_matrix(tc_G, nodelist=sorted_keys).todense()
    adj_mat = adj_mat + adj_mat.T

    lda_dirs: Float[torch.Tensor, "synset_size embedding_dim"] = torch.stack([v['lda'] for k, v in dirs.items()])
    lda_dirs = lda_dirs / lda_dirs.norm(dim=1).unsqueeze(1)

    child_parent = {}
    for node in sorted_keys:
        if len(list(G.predecessors(node))) > 0:
            parent = list(G.predecessors(node))[0]
            if parent not in messed_up:
                if [*dirs[node]['lda']] == [*dirs[parent]['lda']]:
                    print(f"equal: {node}, {parent}")
                child_parent.update({node: dirs[node]['lda'] - dirs[parent]['lda']})
        else:
            print("reject: " + node)

    lda_diff: Float[torch.Tensor, "synset_size embedding_dim"] = torch.stack([lda_dirs[0]] + [v for v in child_parent.values()])
    lda_diff = lda_diff / lda_diff.norm(dim=1).unsqueeze(1)

    hi: Float[np.ndarray, "synset_size synset_size"] = (lda_diff @ lda_diff.T).cpu().numpy()
    mats = [adj_mat, (lda_dirs @ lda_dirs.T).cpu().numpy(), hi]
    return mats

def run_obofoundry_scores(params: str, step: str, ontology_dir: str):
    save_ontology_hypernym(params, step, ontology_dir)
    return generate_noun_heatmaps(params, step)

mats = run_obofoundry_scores('70M', 'step13000', 'https://raw.githubusercontent.com/insect-morphology/aism/master/aism.owl')
print(mats)
