#!/usr/bin/env python3

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPTNeoXForCausalLM
from tqdm import tqdm
import os

from hf_olmo import OLMoForCausalLM, OLMoTokenizerFast
  # pip install ai2-olmo


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
    # model = GPTNeoXForCausalLM.from_pretrained(
    # f"EleutherAI/pythia-{parameter_model}-deduped",
    # revision=f"{step}",
    # cache_dir=f"/mnt/bigstorage/raymond/huggingface_cache/pythia-{parameter_model}-deduped/{step}",
    # )
    # tokenizer = AutoTokenizer.from_pretrained(
    # f"EleutherAI/pythia-{parameter_model}-deduped",
    # revision=f"{step}",
    # cache_dir=f"/mnt/bigstorage/raymond/huggingface_cache/pythia-{parameter_model}-deduped/{step}",
    # )

    
    model = OLMoForCausalLM.from_pretrained(f"allenai/OLMo-{parameter_model}", revision=step)

    tokenizer = OLMoTokenizerFast.from_pretrained(f"allenai/OLMo-{parameter_model}", revision=step)

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


# parameter_models = ["7B"]

# steps = [f"step{i}-tokens{int((i/1000)*4)}B" for i in range(3000, 145000, 1000)]
# print(steps)



# for parameter_model in parameter_models:
#     folder = f"/mnt/bigstorage/raymond/olmo/{parameter_model}-unembeddings"
#     # os.mkdir(folder)
#     for step in steps:
#         generate_unembedding_matrix(parameter_model, step, folder)



# model = OLMoForCausalLM.from_pretrained("allenai/OLMo-7B", revision="step1000-tokens4B")

# tokenizer = OLMoTokenizerFast.from_pretrained("allenai/OLMo-7B", revision="step1000-tokens4B")

tokenizer_olmo = OLMoTokenizerFast.from_pretrained("allenai/OLMo-7B", revision="step1000-tokens4B")
vocab_olmo = tokenizer_olmo.get_vocab()
vocab_set_olmo = set(vocab_olmo.keys())



tokenizer_pythia = AutoTokenizer.from_pretrained(
    f"EleutherAI/pythia-70M-deduped",
    revision=f"step1000",
    cache_dir=f"/mnt/bigstorage/raymond/huggingface_cache/pythia-70M-deduped/step1000",
)
vocab_pythia = tokenizer_pythia.get_vocab()
vocab_set_pythia = set(vocab_pythia.keys())


print(vocab_set_olmo.difference(vocab_set_pythia))
print(vocab_set_pythia.difference(vocab_set_olmo))