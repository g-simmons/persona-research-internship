#!/usr/bin/env python3

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import GPTNeoXForCausalLM, PretrainedConfig
from tqdm import tqdm
import os
import pathlib
import logging
import argparse
from hf_olmo import OLMoForCausalLM, OLMoTokenizerFast
from huggingface_hub import hf_hub_download
import json
import psutil
import gc
from pathlib import Path
from safetensors.torch import load_file
import inspect
import torch.nn.functional as F
from jaxtyping import Float, Int, Bool  # or use Scalar for non-tensors


# pip install ai2-olmo

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers
file_handler = logging.FileHandler("store_matrices.log")
stdout_handler = logging.StreamHandler()

# Create formatters and add it to handlers
log_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(log_format)
stdout_handler.setFormatter(log_format)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

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


def get_memory_usage() -> float:
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    memory_gb = process.memory_info().rss / 1024 / 1024 / 1024
    return memory_gb

def load_olmo_last_layer(model_name: str) -> tuple[dict, float]:
    """Load only the last layer weights and return them with memory usage."""
    initial_memory = get_memory_usage()

    # Get the model config first
    config = AutoConfig.from_pretrained(model_name)
    last_layer_idx = config.num_hidden_layers - 1

    # Define last layer weight keys
    last_layer_keys = [
        f"model.layers.{last_layer_idx}.self_attn.q_proj",
        f"model.layers.{last_layer_idx}.self_attn.k_proj",
        f"model.layers.{last_layer_idx}.self_attn.v_proj",
        f"model.layers.{last_layer_idx}.self_attn.o_proj",
        f"model.layers.{last_layer_idx}.mlp.gate_proj",
        f"model.layers.{last_layer_idx}.mlp.up_proj",
        f"model.layers.{last_layer_idx}.mlp.down_proj",
        f"model.layers.{last_layer_idx}.input_layernorm",
        f"model.layers.{last_layer_idx}.post_attention_layernorm",
    ]

    # Load index file
    index_file = hf_hub_download(model_name, "pytorch_model.bin.index.json")
    with open(index_file, "r") as f:
        index = json.load(f)
    
    last_layer_weights: dict = {}

    # Load only last layer weights
    for key in last_layer_keys:
        for weight_key, filename in index["weight_map"].items():
            if key in weight_key:
                checkpoint_file = hf_hub_download(model_name, filename)
                weights = torch.load(checkpoint_file, map_location="cpu")
                last_layer_weights[key] = weights[key]
                break

    final_memory = get_memory_usage()
    return last_layer_weights, final_memory - initial_memory


def generate_unembedding_matrix(
    parameter_model: str, step: str, output_dir: str, use_gpu: bool = False
):
    """
    Apply the causal inner product to the unembedding matrix and save it.
    The causal inner product is estimated as the product of the square root of the
    covariance matrix of the unembedding vectors and the centered unembedding vectors.
    """
    # TODO parameterize instead of comment

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

    model = OLMoForCausalLM.from_pretrained(
        f"allenai/OLMo-{parameter_model}", revision=step
    )

    # load unembdding vectors
    gamma = model.get_output_embeddings().weight.detach() # type: ignore

    if use_gpu:
        gamma = gamma.to("cuda")
    
    W, d = gamma.shape

    gamma_bar: Float[torch.Tensor, "embedding_dim"] = torch.mean(gamma, dim=0)
    centered_gamma: Float[torch.Tensor, "vocab_size embedding_dim"] = gamma - gamma_bar

    # compute Cov(gamma) and tranform gamma to g
    Cov_gamma = centered_gamma.T @ centered_gamma / W

    if use_gpu:
        eigenvalues, eigenvectors = torch.linalg.eigh(
            Cov_gamma
        )  # for hermitian matrices
    else:
        eigenvalues, eigenvectors = np.linalg.eigh(Cov_gamma)
    
    if not use_gpu:
        eigenvalues = torch.from_numpy(eigenvalues)
        eigenvectors = torch.from_numpy(eigenvectors)

    if not isinstance(eigenvalues, torch.Tensor):
        eigenvalues = torch.from_numpy(eigenvalues)

    inv_sqrt_Cov_gamma = (
        eigenvectors @ torch.diag(1 / torch.sqrt(eigenvalues)) @ eigenvectors.T
    )
    
    g: Float[torch.Tensor, "vocab_size embedding_dim"] = centered_gamma @ inv_sqrt_Cov_gamma

    torch.save(g, f"{output_dir}/{step}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate unembedding matrices for OLMo models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--parameter-models",
        type=str,
        nargs="+",
        default=["7B"],
        help="Parameter models to process (e.g., 7B, 1B)"
    )
    
    parser.add_argument(
        "--steps",
        type=str,
        nargs="+",
        help="Specific steps to process. If not specified, loads from olmo_7B_model_names.txt"
    )
    
    parser.add_argument(
        "--step-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        default=[1, 15],
        help="Range of steps to process (start, end) when loading from file"
    )
    
    parser.add_argument(
        "--single-step",
        action="store_true",
        help="Process only the second step from the file (index 1)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory. Defaults to /mnt/bigstorage/{user}/olmo/{model}-unembeddings"
    )
    
    parser.add_argument(
        "--user",
        type=str,
        default="raymond",
        help="User name for output directory path"
    )
    
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU for computations"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Data directory containing olmo_7B_model_names.txt. Defaults to ../data relative to script"
    )
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    SCRIPT_DIR = pathlib.Path(__file__).parent
    DATA_DIR = pathlib.Path(args.data_dir) if args.data_dir else SCRIPT_DIR / "../data"
    BIGSTORAGE_DIR = pathlib.Path("/mnt/bigstorage")
    
    # Load steps
    if args.steps:
        steps = args.steps
    else:
        try:
            with open(DATA_DIR / "olmo_7B_model_names.txt", "r") as f:
                steps = [line.strip() for line in f.readlines()]
            steps.sort(key=lambda x: int(x.split("-")[0].split("p")[1]))
            logger.info(f"Total number of steps from file: {len(steps)}")
            
            if args.single_step:
                steps = [steps[1]] if len(steps) > 1 else steps[:1]
            else:
                start_idx, end_idx = args.step_range
                steps = steps[start_idx:end_idx]
                
        except FileNotFoundError:
            logger.error(f"Could not find olmo_7B_model_names.txt in {DATA_DIR}")
            logger.error("Please specify --steps manually or provide --data-dir")
            return
    
    logger.info(f"Selected steps: {steps}")
    logger.info(f"Number of selected steps: {len(steps)}")
    
    for parameter_model in args.parameter_models:
        if args.output_dir:
            folder = pathlib.Path(args.output_dir)
        else:
            folder = BIGSTORAGE_DIR / args.user / "olmo" / f"{parameter_model}-unembeddings"
        
        folder.mkdir(parents=True, exist_ok=True)
        
        for step in steps:
            logger.info(f"Processing model {parameter_model} at step {step}")
            generate_unembedding_matrix(parameter_model, step, str(folder), use_gpu=args.use_gpu)


if __name__ == "__main__":
    main()