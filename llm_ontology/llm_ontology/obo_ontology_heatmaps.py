#!/usr/bin/env python3

# import nltk
# nltk.download('wordnet')

import json
import networkx as nx
import argparse

# from nltk.corpus import wordnet as wn
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
import pathlib
import logging
import inflect

import huggingface_hub

import torch
import numpy as np
import networkx as nx
from typing import List, Tuple
import matplotlib.pyplot as plt
import hierarchical as hrc
import warnings
from jaxtyping import Float

warnings.filterwarnings("ignore")

import ontology_class

# Set up logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Create file handler
    file_handler = logging.FileHandler("obo_ontology_heatmaps.log")
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Create stdout handler
    stdout_handler = logging.StreamHandler()
    stdout_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    stdout_handler.setFormatter(stdout_formatter)
    logger.addHandler(stdout_handler)

    logger.setLevel(logging.INFO)

# Global paths
BIGSTORAGE_DIR = pathlib.Path("/mnt/bigstorage")
SCRIPT_DIR = pathlib.Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
FIGURES_DIR = SCRIPT_DIR.parent / "figures"


def merge_nodes(G: nx.DiGraph, lemma_dict: dict):
    # if a node has only one child, and that child has only one parent, merge the two nodes
    topological_sorted_nodes = list(reversed(list(nx.topological_sort(G))))

    for node in topological_sorted_nodes:
        children = list(G.successors(node))
        if len(children) == 1:
            child = children[0]
            parent_lemmas_not_in_child = lemma_dict[node] - lemma_dict[child]
            if (
                len(list(G.predecessors(child))) == 1
                or len(parent_lemmas_not_in_child) < 6
            ):
                grandchildren = list(G.successors(child))

                # if len(parent_lemmas_not_in_child) > 1:
                #     if len(grandchildren) > 0:
                #         lemma_dict[node + '.other'] = parent_lemmas_not_in_child
                #         G.add_edge(node, node + '.other')
                #         logger.info(node)

                # del synset_lemmas[child]
                for grandchild in grandchildren:
                    G.add_edge(node, grandchild)
                G.remove_node(child)
                logger.info(f"merged {node} and {child}")


def _noun_to_gemma_vocab_elements(
    word, multi: bool, vocab_set: set[str], p: inflect.engine
) -> list[str]:
    word = word.lower()
    plural = p.plural(word)
    if multi:
        word_list = word.split("_")
        plural_list = plural.split("_")
        word_cap_list = word.capitalize().split("_")
        plural_cap_list = plural.capitalize().split("_")

        add_cap_and_plural = [word_list, plural_list, word_cap_list, plural_cap_list]
        corr_words = [word, plural, word.capitalize(), plural.capitalize()]

        included = []
        for i in range(len(add_cap_and_plural)):
            if set(add_cap_and_plural[i]).issubset(vocab_set):
                included.append(corr_words[i])

        return included
    else:
        add_cap_and_plural = [word, word.capitalize(), plural, plural.capitalize()]
        # add_space = ["▁" + w for w in add_cap_and_plural]
        add_space = [w for w in add_cap_and_plural]
        return vocab_set.intersection(add_space)


def save_ontology_hypernym(params: str, step: str, ontology_name: str, multi: bool):

    ontology_dir = BIGSTORAGE_DIR / "raymond" / "owl" / f"{ontology_name}.owl"
    # model_name = "gemma-2b"
    # tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    model_name = "pythia"

    tokenizer = AutoTokenizer.from_pretrained(
        f"EleutherAI/pythia-{params}-deduped",
        revision=f"{step}",
        cache_dir=BIGSTORAGE_DIR
        / "raymond"
        / "huggingface_cache"
        / f"pythia-{params}-deduped"
        / step,
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
            lemmas.update(
                get_all_hyponym_lemmas(hyponym)
            )  # Recursively get lemmas from hyponyms,

        return lemmas

    # all_noun_synsets = list(wn.all_synsets(pos=wn.NOUN))
    test = ontology_class.Onto(ontology_dir)
    all_noun_synsets = test.all_synsets()
    logger.info(f"Number of noun synsets: {len(all_noun_synsets)}")
    noun_lemmas = {}
    for s in all_noun_synsets:
        lemmas = get_all_hyponym_lemmas(s)
        if model_name == "gemma-2b":  # add and remove space bc of how gemma vocab works
            lemmas = vocab_set.intersection({"▁" + l for l in lemmas})
            noun_lemmas[s.name()] = {l[1:] for l in lemmas}
        elif model_name == "pythia":
            if multi:
                lemmas = list(lemmas)
                lemmas_split = [set(l.split("_")) for l in lemmas]
                lemmas_included = []
                for i in range(len(lemmas)):
                    if lemmas_split[i].issubset(vocab_set):
                        lemmas_included.append(lemmas[i])

                noun_lemmas[s.name()] = set(lemmas_included)
            else:
                lemmas = vocab_set.intersection(lemmas)
                noun_lemmas[s.name()] = lemmas

    large_nouns = {k: v for k, v in noun_lemmas.items() if len(v) > 1}

    logger.info(f"Total noun lemmas: {len(noun_lemmas)}")
    logger.info(f"Large nouns: {len(large_nouns)}")

    G_noun = nx.DiGraph()
    nodes = list(large_nouns.keys())

    for key in nodes:
        for path in test.get_synset(key).hypernym_paths():
            ancestors = [s.name() for s in path if s.name() in nodes]

            if len(ancestors) > 1:
                G_noun.add_edge(ancestors[-2], key)  # first entry is itself
            else:
                logger.info(f"no ancestors for {key}")

    merge_nodes(G_noun, large_nouns)
    # logger.info("SECOND MERGE")
    # merge_nodes(G_noun, large_nouns)

    large_nouns = {k: v for k, v in large_nouns.items() if k in G_noun.nodes()}

    G_noun = nx.DiGraph(G_noun.subgraph(nodes))

    ## save the data
    ontology_data_path = (
        DATA_DIR / "ontologies" / f"noun_synsets_ontology_pythia_{ontology_name}.json"
    )
    with open(ontology_data_path, "w") as f:
        prev_pythia_words = []
        corr_synsets = []
        for synset, lemmas in large_nouns.items():
            pythia_words = []
            for w in lemmas:
                pythia_words.extend(
                    _noun_to_gemma_vocab_elements(w, multi, vocab_set, p)
                )
                if synset == "organism_substance":
                    logger.info(w)

            if pythia_words in prev_pythia_words:
                logger.info(f"repeated: {synset}")  # for repeating synset lemmas
            prev_pythia_words.append(pythia_words)
            corr_synsets.append(synset)

            f.write(json.dumps({synset: pythia_words}) + "\n")

    graph_path = (
        DATA_DIR
        / "ontologies"
        / f"noun_synsets_ontology_hypernym_graph_{ontology_name}.adjlist"
    )
    nx.write_adjlist(G_noun, graph_path)


def update_vocab_dict(tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast) -> dict:
    vocab_dict = tokenizer.get_vocab()  # type: ignore
    new_vocab_dict = {}
    for key, value in vocab_dict.items():
        new_key = key.replace(" ", "_")
        new_vocab_dict[new_key] = value
    vocab_dict = new_vocab_dict
    return vocab_dict


def get_mats(
    params: str, step: str, multi: bool, filter: int, ontology_name: str
) -> List[np.ndarray]:
    """
    Args
    multi: bool
        Use multi-word embeddings
    filter: int
        Remove synsets that don't have this many terms
    """
    device = torch.device("cpu")
    # tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    tokenizer = AutoTokenizer.from_pretrained(
        f"EleutherAI/pythia-{params}-deduped",
        revision=f"{step}",
        cache_dir=BIGSTORAGE_DIR
        / "raymond"
        / "huggingface_cache"
        / f"pythia-{params}-deduped"
        / step,
    )

    g: Float[torch.Tensor, "vocab_size embedding_dim"] = torch.load(
        BIGSTORAGE_DIR / "raymond" / "pythia" / f"{params}-unembeddings" / step
    ).to(
        device
    )  # 'FILE_PATH' in store_matrices.py

    vocab_dict = update_vocab_dict(tokenizer)
    cats, G, sorted_keys = hrc.get_categories_ontology(
        ontology_name=ontology_name, filter=filter
    )

    error_count = 0
    total_count = 0
    messed_up = {}
    dirs = {}
    for k, v in cats.items():
        total_count += 1
        try:
            dirs[k] = hrc.estimate_cat_dir(v, g, vocab_dict, multi)
        except Exception as e:
            error_count += 1
            logger.info(str(e))
            messed_up[k] = v
    logger.info(f"Error count: {error_count}")
    logger.info(f"Total count: {total_count}")
    logger.info(f"Messed up: {messed_up}")

    tc_G = nx.algorithms.dag.transitive_closure(G)
    adj_mat = nx.adjacency_matrix(tc_G, nodelist=sorted_keys).todense()
    adj_mat = adj_mat + adj_mat.T

    lda_dirs = torch.stack([v["lda"] for k, v in dirs.items()])
    lda_dirs = lda_dirs / lda_dirs.norm(dim=1).unsqueeze(1)

    for k, v in dirs.items():
        logger.info(k)

    child_parent = {}

    logger.info(f"Sorted keys: {sorted_keys}")

    for node in list(messed_up.keys()):
        sorted_keys.remove(node)
        logger.info(node)

    class_counter = 0
    for node in sorted_keys:
        if len(list(G.predecessors(node))) > 0:
            parent = list(G.predecessors(node))[0]  # direct parent
            if parent not in list(messed_up.keys()):
                if [*dirs[node]["lda"]] == [*dirs[parent]["lda"]]:
                    logger.info(f"equal: {node}, {parent}")
                child_parent.update({node: dirs[node]["lda"] - dirs[parent]["lda"]})
        else:
            class_counter += 1
            logger.info(f"reject: {node}")  # throws out 3

    # lda_diff = torch.stack([lda_dirs[0]] + [v for k, v in child_parent.items()])    #adds back 1: 100 - 3 + 1 = 98
    lda_diff = torch.stack(
        [lda_dirs[i] for i in range(class_counter)]
        + [v for k, v in child_parent.items()]
    )

    logger.info(f"norms: {lda_diff.norm(dim = 1)}")
    for thing in lda_diff:
        logger.info(str(thing))

    lda_diff = lda_diff / lda_diff.norm(dim=1).unsqueeze(1)

    # multiplying by transpose to get cosine similarity
    # num_concepts x embedding_dim * embedding_dim x num_concepts = num_concepts x num_concepts
    mats = [
        adj_mat,
        (lda_dirs @ lda_dirs.T).cpu().numpy(),
        (lda_diff @ lda_diff.T).cpu().numpy(),
    ]

    return mats


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate ontology heatmaps',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--params', 
        type=str, 
        default='70M',
        help='Model parameter size'
    )
    
    parser.add_argument(
        '--step-int', 
        type=int, 
        default=143000,
        help='Training step number'
    )
    
    parser.add_argument(
        '--step', 
        type=str,
        help='Training step string (e.g., step143000). If provided, overrides --step-int'
    )
    
    parser.add_argument(
        '--ontology-name', 
        type=str, 
        default='cl',
        help='Name of ontology file without .owl extension'
    )
    
    parser.add_argument(
        '--multiword', 
        action='store_true', 
        default=True,
        help='Whether to use multiword processing'
    )
    
    parser.add_argument(
        '--filter', 
        type=int, 
        default=15,
        help='Filter out synsets with less than this many terms'
    )
    
    parser.add_argument(
        '--action',
        type=str,
        choices=['save_hypernym', 'get_mats', 'both'],
        default='both',
        help='Action to perform'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for saving matrices'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Determine step string
    if args.step:
        step = args.step
    else:
        step = f"step{args.step_int}"
    
    logger.info(f"Processing {args.params} at {step} with ontology {args.ontology_name}")
    
    if args.action in ['save_hypernym', 'both']:
        save_ontology_hypernym(
            params=args.params, 
            step=step, 
            ontology_name=args.ontology_name, 
            multi=args.multiword
        )
    
    if args.action in ['get_mats', 'both']:
        mats = get_mats(
            params=args.params,
            step=step,
            multi=args.multiword,
            filter=args.filter,
            ontology_name=args.ontology_name
        )
        
        logger.info(f"Matrix 0 shape: {mats[0].shape}")
        logger.info(f"Matrix 1 shape: {mats[1].shape}")
        logger.info(f"Matrix 2 shape: {mats[2].shape}")
        
        # Save matrices if output directory specified
        if args.output_dir:
            output_path = pathlib.Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            torch.save(mats[0], output_path / f"{args.ontology_name}_{step}_adj.pt")
            torch.save(mats[1], output_path / f"{args.ontology_name}_{step}_lda_dirs.pt")
            torch.save(mats[2], output_path / f"{args.ontology_name}_{step}_lda_diff.pt")
            
            logger.info(f"Matrices saved to {output_path}")


if __name__ == "__main__":
    main()
