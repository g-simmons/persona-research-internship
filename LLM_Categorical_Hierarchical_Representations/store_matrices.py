#!/usr/bin/env python3

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPTNeoXForCausalLM
from tqdm import tqdm
import os

# ### load model ###
# device = torch.device("cuda:0")
# tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
# model = AutoModelForCausalLM.from_pretrained("google/gemma-2b",
#                                              torch_dtype=torch.float32,
#                                              device_map="auto")

# gamma = model.get_output_embeddings().weight.detach()
# W, d = gamma.shape
# gamma_bar = torch.mean(gamma, dim = 0)
# centered_gamma = gamma - gamma_bar

# ### compute Cov(gamma) and tranform gamma to g ###
# Cov_gamma = centered_gamma.T @ centered_gamma / W
# eigenvalues, eigenvectors = torch.linalg.eigh(Cov_gamma)
# inv_sqrt_Cov_gamma = eigenvectors @ torch.diag(1/torch.sqrt(eigenvalues)) @ eigenvectors.T
# sqrt_Cov_gamma = eigenvectors @ torch.diag(torch.sqrt(eigenvalues)) @ eigenvectors.T
# g = centered_gamma @ inv_sqrt_Cov_gamma


# ## Use this PATH to load g in the notebooks=
# torch.save(g, f"FILE_PATH")

def generate_unembedding_matrix(parameter_model: str, step: str, output_dir: str):
    model = GPTNeoXForCausalLM.from_pretrained(
    f"EleutherAI/pythia-{parameter_model}-deduped",
    revision=f"{step}",
    cache_dir=f"/mnt/bigstorage/raymond/huggingface_cache/pythia-{parameter_model}-deduped/{step}",
    )
    tokenizer = AutoTokenizer.from_pretrained(
    f"EleutherAI/pythia-{parameter_model}-deduped",
    revision=f"{step}",
    cache_dir=f"/mnt/bigstorage/raymond/huggingface_cache/pythia-{parameter_model}-deduped/{step}",
    )


    ### load unembdding vectors ###
    gamma = model.get_output_embeddings().weight.detach()
    W, d = gamma.shape
    gamma_bar = torch.mean(gamma, dim = 0)
    centered_gamma = gamma - gamma_bar

    ### compute Cov(gamma) and tranform gamma to g ###
    Cov_gamma = centered_gamma.T @ centered_gamma / W
    eigenvalues, eigenvectors = torch.linalg.eigh(Cov_gamma)
    inv_sqrt_Cov_gamma = eigenvectors @ torch.diag(1/torch.sqrt(eigenvalues)) @ eigenvectors.T
    sqrt_Cov_gamma = eigenvectors @ torch.diag(torch.sqrt(eigenvalues)) @ eigenvectors.T
    g = centered_gamma @ inv_sqrt_Cov_gamma


    ## Use this PATH to load g in the notebooks=
    torch.save(g, f"{output_dir}/{step}")


steps = ["step3000", "step7000", "step9000"]
parameter_models = ["6.9B"]

steps = [f"step{i}" for i in range(1000, 145000, 2000)]
print(steps)


for parameter_model in parameter_models:
    folder = f"/mnt/bigstorage/raymond/{parameter_model}-unembeddings"
    os.mkdir(folder)
    for step in steps:
        generate_unembedding_matrix(parameter_model, step, folder)

