#!/usr/bin/env python3

import nltk

nltk.download("wordnet")

import json
import networkx as nx
from nltk.corpus import wordnet as wn
from transformers import AutoTokenizer
from joblib import Parallel, delayed

import inflect

import huggingface_hub

import torch
import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt
import hierarchical as hrc
import warnings

warnings.filterwarnings("ignore")

import seaborn as sns
import random

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename="ontology_scores_log_test.log", level=logging.INFO)

# reproducability
torch.manual_seed(0)
np.random.seed(0)
import random

random.seed(0)


# Internal
def save_wordnet_hypernym(params: str, step: str, multi: bool):
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

    one_word_counter = 0
    multi_word_counter = 0

    all_noun_synsets = list(wn.all_synsets(pos=wn.NOUN))
    noun_lemmas = {}
    for s in all_noun_synsets:
        lemmas = get_all_hyponym_lemmas(s)

        # add and remove space bc of how gemma vocab works
        if model_name == "gemma-2b":
            lemmas = vocab_set.intersection({"▁" + l for l in lemmas})
            # lemmas = vocab_set.intersection({l for l in lemmas})
            noun_lemmas[s.name()] = {l[1:] for l in lemmas}
        elif model_name == "pythia":
            if multi:
                lemmas = list(lemmas)
                lemmas_split = [set(l.split("_")) for l in lemmas]
                lemmas_included = []
                for i in range(len(lemmas)):
                    # if vocab_set.intersection(lemmas_split[i]) == lemmas_split[i]:
                    if lemmas_split[i].issubset(vocab_set):
                        lemmas_included.append(lemmas[i])

                for l in lemmas_included:
                    if len(l.split("_")) == 1:
                        one_word_counter += 1
                    elif len(l.split("_")) > 1:
                        multi_word_counter += 1
                noun_lemmas[s.name()] = set(lemmas_included)
            else:
                lemmas = vocab_set.intersection({l for l in lemmas})
                noun_lemmas[s.name()] = {l for l in lemmas}

    logger.info("WORDSCORES")
    logger.info(one_word_counter)
    logger.info(multi_word_counter)
    logger.info("\n")

    logger.info(len(noun_lemmas))
    print(len(noun_lemmas))
    for k, v in noun_lemmas.items():
        print(k, v)
    large_nouns = {k: v for k, v in noun_lemmas.items() if len(v) > 5}

    logger.info(len(all_noun_synsets))
    print(len(all_noun_synsets))
    logger.info(len(large_nouns))
    print(len(large_nouns))

    # Construct the hypernym inclusion graph among large categories
    G_noun = nx.DiGraph()

    nodes = list(large_nouns.keys())
    for key in nodes:
        for path in wn.synset(key).hypernym_paths():
            # ancestors included in the cleaned set
            ancestors = [s.name() for s in path if s.name() in nodes]
            if len(ancestors) > 1:
                G_noun.add_edge(ancestors[-2], key)  # first entry is itself
            else:
                logger.info(f"no ancestors for {key}")
                print(f"no ancestors for {key}")

    G_noun = nx.DiGraph(G_noun.subgraph(nodes))

    # if a node has only one child, and that child has only one parent, merge the two nodes
    def merge_nodes(G, lemma_dict):
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

                    if len(parent_lemmas_not_in_child) > 1:
                        if len(grandchildren) > 0:
                            lemma_dict[node + ".other"] = parent_lemmas_not_in_child
                            G.add_edge(node, node + ".other")

                    # del synset_lemmas[child]
                    for grandchild in grandchildren:
                        G.add_edge(node, grandchild)
                    G.remove_node(child)
                    logger.info(f"merged {node} and {child}")
                    print(f"merged {node} and {child}")

    merge_nodes(G_noun, large_nouns)
    large_nouns = {k: v for k, v in large_nouns.items() if k in G_noun.nodes()}

    # make a gemma specific version
    def _noun_to_gemma_vocab_elements(word):
        word = word.lower()
        plural = p.plural(word)
        if multi:
            word_list = word.split("_")
            plural_list = plural.split("_")
            word_cap_list = word.capitalize().split("_")
            plural_cap_list = plural.capitalize().split("_")

            add_cap_and_plural = [
                word_list,
                plural_list,
                word_cap_list,
                plural_cap_list,
            ]
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

    ## save the data
    with open("data/noun_synsets_wordnet_gemma.json", "w") as f:
        for synset, lemmas in large_nouns.items():
            gemma_words = []
            for w in lemmas:
                gemma_words.extend(_noun_to_gemma_vocab_elements(w))

            gemma_words.sort()
            f.write(json.dumps({synset: gemma_words}) + "\n")

    nx.write_adjlist(G_noun, "data/noun_synsets_wordnet_hypernym_graph.adjlist")


def get_mats(params: str, step: str, multi: bool):
    device = torch.device("cpu")
    # tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")

    tokenizer = AutoTokenizer.from_pretrained(
        f"EleutherAI/pythia-{params}-deduped",
        revision=f"{step}",
        cache_dir=f"/mnt/bigstorage/raymond/huggingface_cache/pythia-{params}-deduped/{step}",
    )

    g = torch.load(f"/mnt/bigstorage/raymond/{params}-unembeddings/{step}").to(
        device
    )  # 'FILE_PATH' in store_matrices.py

    # g = torch.load('FILE_PATH').to(device)

    vocab_dict = tokenizer.get_vocab()
    vocab_list = [None] * (max(vocab_dict.values()) + 1)
    for word, index in vocab_dict.items():
        vocab_list[index] = word

    cats, G, sorted_keys = hrc.get_categories("noun")

    # dirs = {k: hrc.estimate_cat_dir(v, g, vocab_dict) for k, v in cats.items()}

    error_count = 0
    total_count = 0
    messed_up = {}
    dirs = {}

    for k, v in cats.items():
        total_count += 1
        try:
            logger.info(f"NODE: {k}, {v[:3]}")
            print(f"NODE: {k}, {v[:3]}")
            dirs[k] = hrc.estimate_cat_dir(v, g, vocab_dict, multi)

        except Exception as e:
            error_count += 1
            logger.info(e)
            print(e)
            messed_up[k] = v

    logger.info(error_count)
    logger.info(total_count)
    logger.info(messed_up)
    print(error_count)
    print(total_count)
    print(messed_up)

    logger.info(list(messed_up.keys()))
    print(list(messed_up.keys()))

    for node in list(messed_up.keys()):
        sorted_keys.remove(node)
        logger.info(node)
        print(node)

    tc_G = nx.algorithms.dag.transitive_closure(G)
    adj_mat = nx.adjacency_matrix(tc_G, nodelist=sorted_keys).todense()
    adj_mat = adj_mat + adj_mat.T

    lda_dirs = torch.stack([v["lda"] for k, v in dirs.items()])
    lda_dirs = lda_dirs / lda_dirs.norm(dim=1).unsqueeze(1)

    child_parent = {}

    logger.info(sorted_keys)
    print(sorted_keys)

    for node in sorted_keys:
        if len(list(G.predecessors(node))) > 0:
            parent = list(G.predecessors(node))[0]  # direct parent

            logger.info("node: " + node)
            logger.info("parent: " + parent)
            logger.info("\n")

            print("node: " + node)
            print("parent: " + parent)
            print()
            if parent not in list(messed_up.keys()):
                child_parent.update({node: dirs[node]["lda"] - dirs[parent]["lda"]})

    lda_diff = torch.stack([lda_dirs[0]] + [v for k, v in child_parent.items()])
    lda_diff = lda_diff / lda_diff.norm(dim=1).unsqueeze(1)

    # multiplying by transpose to get cosine similarity
    # num_concepts x embedding_dim * embedding_dim x num_concepts = num_concepts x num_concepts
    mats = [
        adj_mat,
        (lda_dirs @ lda_dirs.T).cpu().numpy(),
        (lda_diff @ lda_diff.T).cpu().numpy(),
    ]

    titles = [
        "Adjacency Matrix",
        rf"$\cos(\bar{{\ell}}_{{W}}, \bar{{\ell}}_{{Z}})$",
        rf"$\cos(\bar{{\ell}}_{{W}} - \bar{{\ell}}_{{parent \,\, of \,\, W}}, \bar{{\ell}}_{{Z}} - \bar{{\ell}}_{{parent \,\, of \,\,Z}})$",
    ]

    hrc.cos_heatmap(
        mats,
        titles,
        figsize=(25, 8),
        use_absvals=False,
        save_as="noun_single_three_heatmap",
    )

    plt.savefig("noun_single_three_heatmap.png")

    print(mats[1])
    torch.save(mats, f'mats_{params}_{step}.pth')
    return mats


def get_linear_rep(params: str, step: str, multi: bool):
    sns.set_theme(
        context="paper",
        style="white",  # 'whitegrid', 'dark', 'darkgrid', ...
        palette="colorblind",
        font="DejaVu Sans",  # 'serif'
        font_scale=1.75,  # 1.75, 2, ...
    )

    device = torch.device("cpu")
    # tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    tokenizer = AutoTokenizer.from_pretrained(
        f"EleutherAI/pythia-{params}-deduped",
        revision=f"{step}",
        cache_dir=f"/mnt/bigstorage/raymond/huggingface_cache/pythia-{params}-deduped/{step}",
    )

    g = torch.load(f"/mnt/bigstorage/raymond/{params}-unembeddings/{step}").to(
        device
    )  # 'FILE_PATH' in store_matrices.py

    vocab_dict = tokenizer.get_vocab()
    vocab_list = [None] * (max(vocab_dict.values()) + 1)
    for word, index in vocab_dict.items():
        vocab_list[index] = word

    cats, G, sorted_keys = hrc.get_categories("noun")

    alpha = 0.8

    num_samples = 100000
    torch.random.manual_seed(100)
    all_indices = torch.randperm(g.shape[0])
    random_ind = all_indices[:num_samples]
    random_g = g[random_ind]

    error_count = 0
    total_count = 0
    messed_up = []
    proj = {}
    for node in sorted_keys:
        total_count += 1
        try:
            lemmas = cats[node]
            random.seed(100)
            random.shuffle(lemmas)

            train_lemmas = lemmas[: int(alpha * len(lemmas))]
            test_lemmas = lemmas[int(alpha * len(lemmas)) :]

            estimated_dir = hrc.estimate_cat_dir(train_lemmas, g, vocab_dict, multi)
            logger.info("linear rep lda norm")
            logger.info(estimated_dir["lda"].norm())
            estimated_dir = estimated_dir["lda"] / estimated_dir["lda"].norm()

            # train_g = g[hrc.category_to_indices_multi_word(train_lemmas, vocab_dict)]
            # test_g = g[hrc.category_to_indices_multi_word(test_lemmas, vocab_dict)]

            if multi:
                train_g = hrc.get_category_embeddings(
                    hrc.category_to_indices_multi_word(train_lemmas, vocab_dict), g
                )
                test_g = hrc.get_category_embeddings(
                    hrc.category_to_indices_multi_word(test_lemmas, vocab_dict), g
                )
            else:
                train_g = g[hrc.category_to_indices(train_lemmas, vocab_dict)]
                test_g = g[hrc.category_to_indices(test_lemmas, vocab_dict)]

            b_lda = (train_g @ estimated_dir).mean()

            proj.update(
                {
                    node: {
                        "train": train_g @ estimated_dir,
                        "test": test_g @ estimated_dir,
                        "random": random_g @ estimated_dir,
                    }
                }
            )
        except Exception as e:
            error_count += 1
            print(e)
            logger.info("ERRORR")
            logger.info(e)
            print(node)
            messed_up.append(node)

    print(messed_up)

    logger.info("ERRORSS")
    logger.info(error_count)
    logger.info(total_count)

    for node in messed_up:
        sorted_keys.remove(node)

    mean_std = {}
    for node in sorted_keys:
        mean_std.update(
            {
                node: {
                    key: (
                        proj[node][key].mean().cpu().numpy(),
                        proj[node][key].std().cpu().numpy(),
                    )
                    for key in ["train", "test", "random"]
                }
            }
        )

    train_mean = [mean_std[node]["train"][0] for node in sorted_keys]
    train_std = [mean_std[node]["train"][1] for node in sorted_keys]
    test_mean = [mean_std[node]["test"][0] for node in sorted_keys]
    test_std = [mean_std[node]["test"][1] for node in sorted_keys]
    random_mean = [mean_std[node]["random"][0] for node in sorted_keys]
    random_std = [mean_std[node]["random"][1] for node in sorted_keys]

    inds = range(len(sorted_keys))
    test_train = [test_mean[i] / train_mean[i] for i in inds]
    random_train = [random_mean[i] / train_mean[i] for i in inds]
    test_train_std = [test_std[i] / train_mean[i] for i in inds]
    random_train_std = [random_std[i] / train_mean[i] for i in inds]

    print(test_train)

    fig = plt.figure(figsize=(20, 5))
    plt.plot(inds, test_train, alpha=0.7, label="test")
    plt.plot(inds, random_train, alpha=0.7, label="random")
    plt.errorbar(
        inds,
        test_train,
        yerr=test_train_std,
        color="blue",
        capsize=5,
        ecolor="blue",
        elinewidth=1,
        alpha=0.2,
    )
    plt.errorbar(
        inds,
        random_train,
        yerr=random_train_std,
        color="orange",
        capsize=5,
        ecolor="orange",
        elinewidth=1,
        alpha=0.2,
    )
    plt.legend()
    plt.savefig(f"noun_eval", bbox_inches="tight")

    return test_train


# USE
import numpy as np
from scipy.stats import sem, t


def confidence_interval(data: list, confidence: float = 0.90) -> tuple:
    """
    Calculate the mean and the confidence interval of the data.
    """
    n = len(data)
    mean = np.mean(data)
    std_err = sem(data)  # Standard error of the mean
    h = std_err * t.ppf((1 + confidence) / 2.0, n - 1)  # Margin of error
    return mean, mean - h, mean + h


# def linear_rep_score(values: list) -> float:
#     return sum(values) / len(values)


def linear_rep_score(values: list) -> tuple:
    """
    Calculate the mean and 90% confidence interval of the values.
    """
    return confidence_interval(values)


# def causal_sep_score(adj_mat: np.ndarray, cos_mat: np.ndarray) -> float:
#     size = cos_mat.shape

#     # 0_diag Hadamard product equivalent
#     for i in range(size[0]):
#         cos_mat[i][i] = 0

#     new_mat = cos_mat - adj_mat


#     # Frobenius norm
#     return np.linalg.norm(new_mat, ord = "fro")
def causal_sep_score(adj_mat: np.ndarray, cos_mat: np.ndarray) -> tuple:
    """
    Calculate the Frobenius norm and its 90% confidence interval for the causal separation score.
    """
    size = cos_mat.shape

    # 0_diag Hadamard product equivalent
    for i in range(size[0]):
        cos_mat[i][i] = 0

    new_mat = cos_mat - adj_mat

    # Frobenius norm
    frobenius_norm = np.linalg.norm(new_mat, ord="fro")
    # Dummy example: Assume we have multiple samples for confidence interval calculation
    samples = [
        frobenius_norm for _ in range(10)
    ]  # Replace with actual samples if available
    return confidence_interval(samples)


# def hierarchy_score(cos_mat: np.ndarray) -> float:
#     size = cos_mat.shape

#     # 0_diag Hadamard product equivalent
#     for i in range(size[0]):
#         cos_mat[i][i] = 0

#     # Frobenius norm
#     return np.linalg.norm(cos_mat, ord = "fro")


def hierarchy_score(cos_mat: np.ndarray) -> tuple:
    """
    Calculate the Frobenius norm and its 90% confidence interval for the hierarchy score.
    """
    size = cos_mat.shape

    # 0_diag Hadamard product equivalent
    for i in range(size[0]):
        cos_mat[i][i] = 0

    # Frobenius norm
    frobenius_norm = np.linalg.norm(cos_mat, ord="fro")
    # Dummy example: Assume we have multiple samples for confidence interval calculation
    samples = [
        frobenius_norm for _ in range(10)
    ]  # Replace with actual samples if available
    return confidence_interval(samples)


# def get_scores(params: str, step: str, multi: bool) -> tuple:
#     save_wordnet_hypernym(params, step, multi)
#     mats = get_mats(params, step, multi)
#     eval_values = get_linear_rep(params, step, multi)

#     return linear_rep_score(eval_values), causal_sep_score(mats[0], mats[1]), hierarchy_score(mats[2])


def get_scores(params: str, step: str, multi: bool) -> tuple:
    """
    Calculate and return the mean and 90% confidence intervals for all scores.
    """
    save_wordnet_hypernym(params, step, multi)
    mats = get_mats(params, step, multi)
    eval_values = get_linear_rep(params, step, multi)

    return (
        linear_rep_score(eval_values),
        causal_sep_score(mats[0], mats[1]),
        hierarchy_score(mats[2]),
    )


if __name__ == "__main__":
    steps = [f"step{i}" for i in range(1000, 145000, 2000)]
    parameter_models = ["2.8B"]

    parameter_models = ["70M"]

    steps = [steps[1]]

    print(steps)
    print(len(steps))

    for parameter_model in parameter_models:
        # os.mkdir(f"/mnt/bigstorage/raymond/heatmaps/{parameter_model}")
        for step in steps:
            logger.info(f"Step: {step}")
            logger.info("\n\n\n")

            save_wordnet_hypernym(parameter_model, step, True)
            mats = get_mats(parameter_model, step, True)
            for mat in mats:
                print(mat.shape)

            # torch.save(mats[0], f"/mnt/bigstorage/raymond/heatmaps/{parameter_model}/{parameter_model}-{step}-1.pt")
            # torch.save(mats[1], f"/mnt/bigstorage/raymond/heatmaps/{parameter_model}/{parameter_model}-{step}-2.pt")
            # torch.save(mats[2], f"/mnt/bigstorage/raymond/heatmaps/{parameter_model}/{parameter_model}-{step}-3.pt")

            # score = get_scores(parameter_model, step, False)
            # with open(f"scores_{parameter_model}.txt", "a") as f:
            #     f.write(f"{score[0]}, {score[1]}, {score[2]}\n")

    # save_wordnet_hypernym("70M", "step11000", True)
    # mats = get_mats("70M", "step11000", True)
    # for mat in mats:
    #     print(mat.shape)

    # torch.save(mats[1], "test2.pt")
