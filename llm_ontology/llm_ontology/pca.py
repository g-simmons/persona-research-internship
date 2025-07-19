#!/usr/bin/env python3

import time
import torch
import numpy as np
import networkx as nx
from transformers import AutoTokenizer
from hf_olmo import OLMoForCausalLM, OLMoTokenizerFast
from sklearn.decomposition import PCA
import logging
# Jaxtyping imports
from jaxtyping import Float, Int


import hierarchical as hrc
import ontology_scores


# reproducability
torch.manual_seed(0)
np.random.seed(0)
import random
random.seed(0)


logger = logging.getLogger(__name__)
logging.basicConfig(filename=f"pca_scores.log", level=logging.INFO)

def load_raw_unembedding_matrix_and_tokenizer(
    params: str,
    step: str,
    model_name: str
) -> tuple[Float[torch.Tensor, "vocab_size embedding_dim"], object]:

    device = torch.device("cpu")
    # tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")

    if model_name == "pythia":
        tokenizer = AutoTokenizer.from_pretrained(
            f"EleutherAI/pythia-{params}-deduped",
            revision=f"{step}",
            cache_dir=f"/mnt/bigstorage/raymond/huggingface_cache/pythia-{params}-deduped/{step}",
        )

        g: Float[torch.Tensor, "vocab_size embedding_dim"] = torch.load(f"/mnt/bigstorage/raymond/pythia/{params}-unembeddings/{step}").to(device)

    elif model_name == "olmo":
        tokenizer = OLMoTokenizerFast.from_pretrained("allenai/OLMo-7B", revision=step)
        g: Float[torch.Tensor, "vocab_size embedding_dim"] = torch.load(f"/mnt/bigstorage/raymond/olmo/7B-unembeddings/{step}")
    
    return g, tokenizer

def pca_unembedding_matrix(
    g: Float[torch.Tensor, "vocab_size embedding_dim"],
    new_dim: int,
    output_dir: str = None
) -> Float[torch.Tensor, "vocab_size new_dim"]:
    pca = PCA(n_components=new_dim)
    g_pca = pca.fit_transform(g)
    g_pca = torch.tensor(g_pca)
    if output_dir is not None:
        with open(output_dir, "a") as f:
            f.write("[")
            for i in range(len(pca.explained_variance_ratio_)):
                f.write(str(pca.explained_variance_ratio_[i]) + ", ")
            f.write("], ")
    return g_pca


def get_mats_lol(params: str, step: str, multi: bool, model_name: str, new_dim: int, output_dir: str, apply_pca: bool = True):

    g, tokenizer = load_raw_unembedding_matrix_and_tokenizer(params, step, model_name)

    if apply_pca:
        g = pca_unembedding_matrix(g, new_dim, output_dir=output_dir)

    vocab_dict = tokenizer.get_vocab()

    cats, G, sorted_keys = hrc.get_categories(model_name)

    error_count = 0
    total_count = 0
    messed_up = {}
    dirs = {}

    for k, v in cats.items():
        total_count += 1
        try:
            print(f"NODE: {k}, {v[:3]}")
            # dirs[k] = hrc.estimate_cat_dir_pca(v, g, vocab_dict, multi, new_dim=new_dim)
            dirs[k] = hrc.estimate_cat_dir(v, g, vocab_dict, multi)

        except Exception as e:
            error_count += 1
            print(e)
            messed_up[k] = v

            logger.info(f"NODEERROR: {k}, {v}\n")

    print(error_count)
    print(total_count)
    # print(messed_up)

    # print(list(messed_up.keys()))

    
    for node in list(messed_up.keys()):
        sorted_keys.remove(node)
        print(node)

    logger.info(str(list(messed_up.keys())))
    logger.info(str(sorted_keys))

    tc_G = nx.algorithms.dag.transitive_closure(G)
    adj_mat = nx.adjacency_matrix(tc_G, nodelist=sorted_keys).todense()
    adj_mat = adj_mat + adj_mat.T

    lda_dirs: Float[torch.Tensor, "synset_size new_dim"] = torch.stack([v["lda"] for k, v in dirs.items()])
    lda_dirs = lda_dirs / lda_dirs.norm(dim=1).unsqueeze(1)

    child_parent = {}

    print(sorted_keys)

    for node in sorted_keys:
        if len(list(G.predecessors(node))) > 0:
            parent = list(G.predecessors(node))[0]  # direct parent

            print("node: " + node)
            print("parent: " + parent)
            print()
            if parent not in list(messed_up.keys()):
                child_parent.update({node: dirs[node]["lda"] - dirs[parent]["lda"]})
            else:
                logger.info(f"ERROR1: {node}, parent: {parent}")
                logger.info(str(cats[node]))
                logger.info(str())
        else:
            logger.info(f"ERROR2: {node}")

    lda_diff: Float[torch.Tensor, "synset_size new_dim"] = torch.stack([lda_dirs[0]] + [v for k, v in child_parent.items()])
    lda_diff = lda_diff / lda_diff.norm(dim=1).unsqueeze(1)

    # multiplying by transpose to get cosine similarity
    # num_concepts x embedding_dim * embedding_dim x num_concepts = num_concepts x num_concepts
    mats = [
        adj_mat,
        (lda_dirs @ lda_dirs.T).cpu().numpy(),
        (lda_diff @ lda_diff.T).cpu().numpy(),
    ]

    logger.info(str(mats[0].shape))
    logger.info(str(mats[1].shape))
    logger.info(str(mats[2].shape))

    return mats



if __name__ == "__main__":
    steps = [f"step{i}" for i in range(1000, 145000, 2000)]
    param_model = "1.4B"
    

    g2 = torch.load(f"/mnt/bigstorage/raymond/pythia/1.4B-unembeddings/step1000")
    g3 = torch.load(f"/mnt/bigstorage/raymond/pythia/70M-unembeddings/step1000")

    print(g2.shape)
    print(g3.shape)

    dims = sorted(list(set(list(map(int, list(np.floor(np.geomspace(2, 2048, num=30))))))))
    print(dims[23:])


    for dim in dims[23:]:
        ontology_scores.save_wordnet_hypernym(params = param_model, step = steps[0], multi=True, model_name="pythia")
        output_dir = f"../data/pca_scores/1.4B/dim{dim}.txt"
        for step in steps:
            logger.info(f"{"\n\n"}STEP: {step}")
            mats = get_mats_lol(params = param_model, step = step, multi=True, model_name="pythia", new_dim=dim, output_dir=output_dir)
            adj = mats[0]
            cos = mats[1]
            hier = mats[2]


            for i in range(adj.shape[0]):
                cos[i][i] = 0
                hier[i][i] = 0
            sep_score = np.linalg.norm(cos - adj, ord="fro")
            hier_score = np.linalg.norm(hier, ord="fro")
        

            with open(output_dir, "a") as f:
                f.write(f"{sep_score}, {hier_score}\n")






