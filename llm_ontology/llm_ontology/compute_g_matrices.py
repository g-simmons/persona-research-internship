import torch
import random
import requests
from joblib import Parallel, delayed
import numpy as np
from transformers import GPTNeoXForCausalLM, AutoConfig
import os
import pathlib
import logging
from hf_olmo import OLMoForCausalLM, OLMoTokenizerFast
from huggingface_hub import HfApi, get_safetensors_metadata
from huggingface_hub.utils._safetensors import TensorInfo
from joblib_progress import joblib_progress
from safetensors import safe_open

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("compute_g_matrices.log")
stdout_handler = logging.StreamHandler()

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
stdout_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

model_name = "allenai/OLMo-1.4B"

BIGSTORAGE_DIR = pathlib.Path("/mnt/bigstorage")


def get_current_user() -> str:
    return os.environ["USER"]


def save_gamma_matrix(model_name: str, revision: str, user: str, fast: bool = True) -> None:
    """

    The park paper uses model.get_output_embeddings().weight.detach()

    The definition of get_output_embeddings is:

        def get_output_embeddings(self):
            if self.config.weight_tying:
                return self.model.transformer.wte
            else:
                return self.model.transformer.ff_out

    To avoid downloading the whole model (fast mode), we can use the config to check for weight_tying, then get the appropriate embedding matrix from the safetensors metadata.
    """
    if fast:
        config = AutoConfig.from_pretrained(model_name, revision=revision)

        if config.weight_tying:
            matrix_name = "model.transformer.wte.weight"
        else:
            matrix_name = "model.transformer.ff_out.weight"

        metadata = get_safetensors_metadata(model_name, revision=revision)

        file = metadata.weight_map[matrix_name]
        tensor_info: TensorInfo = metadata.files_metadata[file].tensors[matrix_name]
        # tensor_info is something like TensorInfo(dtype='F32', shape=[50304, 2048], data_offsets=(4294967296, 4707057664), parameter_count=103022592)

        # Let's download only these bytes using the data_offsets
        url = f"https://huggingface.co/{model_name}/resolve/{revision}/model.safetensors"
        headers = {
            "Range": f"bytes={tensor_info.data_offsets[0]}-{tensor_info.data_offsets[1] - 1}"
        }


        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.content
        gamma = torch.frombuffer(data, dtype=torch.float32).reshape(tensor_info.shape)
    
    else:
        if "OLMo" in model_name:
            model = OLMoForCausalLM.from_pretrained(
                model_name,
                revision=revision,
                cache_dir=BIGSTORAGE_DIR / user / "huggingface_cache",
            )
        elif "pythia" in model_name:
            model = GPTNeoXForCausalLM.from_pretrained(
                model_name,
                revision=revision,
                cache_dir=BIGSTORAGE_DIR / user / "huggingface_cache",
            )
        else:
            raise ValueError(f"Model {model_name} not supported")

        gamma = model.get_output_embeddings().weight.detach()  # type: ignore

    torch.save(gamma, f"{model_name.split('/')[-1]}-{revision}.pt")


def generate_unembedding_matrix(
    model_name: str, step: str, output_dir: str, use_gpu: bool = False
) -> None:
    """
    Apply the causal inner product to the unembedding matrix and save it.
    The causal inner product is estimated as the product of the square root of the
    covariance matrix of the unembedding vectors and the centered unembedding vectors.
    """

    logger.info(f"Loading model {model_name} at step {step}")
    model = OLMoForCausalLM.from_pretrained(model_name, revision=step)

    # load unembdding vectors
    gamma: Float[Tensor, "vocab_size embedding_dim"] = model.get_output_embeddings().weight.detach()  # type: ignore

    if use_gpu:
        logger.info("Moving gamma to CUDA")
        gamma = gamma.to("cuda")

    W, d = gamma.shape
    logger.info(f"gamma shape: {gamma.shape}")

    gamma_bar = torch.mean(gamma, dim=0)
    centered_gamma = gamma - gamma_bar

    # compute Cov(gamma) and tranform gamma to g
    Cov_gamma = centered_gamma.T @ centered_gamma / W

    if use_gpu:
        logger.info("Computing eigenvalues/eigenvectors with torch.linalg.eigh")
        eigenvalues, eigenvectors = torch.linalg.eigh(
            Cov_gamma
        )  # for hermitian matrices
    else:
        logger.info("Computing eigenvalues/eigenvectors with np.linalg.eigh")
        eigenvalues, eigenvectors = np.linalg.eigh(Cov_gamma)

    if not use_gpu:
        eigenvalues = torch.from_numpy(eigenvalues)
        eigenvectors = torch.from_numpy(eigenvectors)

    if not isinstance(eigenvalues, torch.Tensor):
        eigenvalues = torch.from_numpy(eigenvalues)

    inv_sqrt_Cov_gamma = (
        eigenvectors @ torch.diag(1 / torch.sqrt(eigenvalues)) @ eigenvectors.T
    )

    g = centered_gamma @ inv_sqrt_Cov_gamma

    output_path = f"{output_dir}/{step}"
    logger.info(f"Saving g matrix to {output_path}")
    torch.save(g, output_path)


def get_output_path(model_name: str, step: str, user: str) -> str:
    return str(BIGSTORAGE_DIR / user / "g_matrices" / f"{model_name}-unembeddings")


def get_olmo_model_revisions(model_name: str) -> list[str]:
    api = HfApi()
    refs = api.list_repo_refs(model_name)
    revisions = []
    for branch in refs.branches:
        revision = branch.name
        revisions.append(revision)
    return revisions


def get_olmo_model_names() -> list[str]:
    olmo_model_names = ["allenai/OLMo-1B", "allenai/OLMo-7B"]
    return olmo_model_names


def main():
    # save_gamma_matrix("allenai/OLMo-1B", "main", "gabe", fast=True)

    slow = torch.load("OLMo-1B-main-slow.pt")
    fast = torch.load("OLMo-1B-main.pt")
    from ptpython import embed
    embed(globals(), locals(), vi_mode=True)

    # parallel: bool = False
    # debug: bool = True
    # user: str = get_current_user()
    # model_names: list[str] = get_olmo_model_names()
    # name_revisions = {}

    # for model_name in model_names:
    #     revisions: list[str] = get_olmo_model_revisions(model_name)

    #     if debug:
    #         revisions = random.sample(revisions, 3)

    #     name_revisions[model_name] = revisions

    # if parallel:
    #     with joblib_progress(
    #         "Calculating gamma matrices...",
    #         total=sum(len(revisions) for revisions in name_revisions.values()),
    #     ):
    #         Parallel(n_jobs=4)(
    #             delayed(save_gamma_matrix)(model_name, revision, user)
    #             for model_name in model_names
    #             for revision in name_revisions[model_name]
    #         )
    # else:
    #     for model_name in model_names:
    #         for revision in name_revisions[model_name]:
    #             gamma = save_gamma_matrix(model_name, revision, user)
    #             logger.info(f"{model_name} {revision}")


if __name__ == "__main__":
    main()
