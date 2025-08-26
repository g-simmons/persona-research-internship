#!/usr/bin/env python3

import json
import logging
import pathlib
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import altair as alt
import argparse
from pathlib import Path
# Jaxtyping imports
from jaxtyping import Float, Int
from typeguard import typechecked

#from utils import savefig, figname_from_fig_metadata
from utils import savefig, figname_from_fig_metadata
#from .ontology_scores import causal_sep_score_simple as causal_sep_score

# Set up logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Global paths
BIGSTORAGE_DIR = pathlib.Path("/mnt/bigstorage")

def save_scatterplot(
    adj: Float[torch.Tensor, "synset_size embedding_dim"] | Float[np.ndarray, "synset_size embedding_dim"],
    cos: Float[torch.Tensor, "synset_size embedding_dim"] | Float[np.ndarray, "synset_size embedding_dim"],
    row_terms_path: pathlib.Path,
    term_freq_path: pathlib.Path,
    model_name: str = "",
    param_model: str = "",
    index: str = ""
) -> None:
    """Create and save a scatterplot comparing term frequencies to causal separability scores.
    
    Args:
        adj: Adjacency matrix tensor
        cos: Cosine similarity matrix tensor 
        row_terms_path: Path to file containing row terms
        term_freq_path: Path to JSON file containing term frequencies
    """
    # Load terms and frequencies
    with open(row_terms_path, "r") as f:
        row_terms = [line.strip() for line in f.readlines()]
    
    with open(term_freq_path, 'r') as f:
        outer_dict = json.load(f)
        print(outer_dict.keys())
        term_freq = outer_dict[list(outer_dict.keys())[0]]
        #print(term_freq)  # Access the nested dictionary directly

    # Calculate causal separability score
    #score = causal_sep_score(adj, cos)
    
    # Map frequencies to scores
    scores = {}  # freq: score
    for i, term in enumerate(row_terms):
        if term in term_freq:  # Check if term exists in frequency data
            diff = cos[i] - adj[i]
            #print(term_freq)
            #print(row_terms)
            scores[term_freq[term]] = float(np.linalg.norm(diff))

    # Create scatterplot
    freqs = list(scores.keys())
    term_scores = list(scores.values())
    
    # Set up the plot style
    sns.set_style("whitegrid")
    
    # Create figure and axis using object-oriented interface
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create scatterplot with seaborn
    sns.scatterplot(x=freqs, y=term_scores, alpha=0.6, ax=ax)
    ax.set_xscale('log')
    ax.set_xlabel("Pretraining Term Frequency")
    ax.set_ylabel("Term Causal Separability Score")

    if index:
        # Save plot
        metadata = {
            "type": "frequency_score_scatterplot",
            "title": "Term Frequency vs Causal Separability Score - {model_name} - {index}"
        }
        #figure_name = figname_from_fig_metadata(metadata)
        figure_name = f"frequency_score_scatterplot_{model_name}_{param_model}_{index}.png"
    else:
        # Save plot
        metadata = {
            "type": "frequency_score_scatterplot",
            "title": "Term Frequency vs Causal Separability Score - {model_name}"
        }
        #figure_name = figname_from_fig_metadata(metadata)
        figure_name = f"frequency_score_scatterplot_{model_name}_{param_model}.png"
    
    script_dir = pathlib.Path(__file__).parent
    figures_dir = script_dir.parent / "figures"
    
    savefig(
        fig=fig,
        figures_dir=figures_dir,
        figure_name=figure_name,
        formats=["png"],
        overwrite=True
    )
    plt.close(fig)

def create_interactive_scatterplot(
    adj: Float[np.ndarray, "synset_size embedding_dim"],
    cos: Float[np.ndarray, "synset_size embedding_dim"],
    row_terms_path: pathlib.Path,
    term_freq_path: pathlib.Path,
    model_name: str = "",
    param_model: str = "",
    index: str = ""
) -> alt.Chart:
    """Create an interactive Altair scatterplot comparing term frequencies to causal separability scores.

    Args:
        adj: Adjacency matrix numpy array
        cos: Cosine similarity matrix numpy array
        row_terms_path: Path to file containing row terms
        term_freq_path: Path to JSON file containing term frequencies
        model_name: Name of the model
        param_model: Parameterization of the model
        index: Optional index for the plot title/filename

    Returns:
        alt.Chart: An Altair chart object
    """
    # Load terms and frequencies
    with open(row_terms_path, "r") as f:
        row_terms = [line.strip() for line in f.readlines()]

    with open(term_freq_path, 'r') as f:
        print("/n/n")
        print(term_freq_path)
        outer_dict = json.load(f)
        #need to get the inside json (is nested and need to get the first one)
        term_freq = outer_dict[list(outer_dict.keys())[0]]

    # Calculate causal separability score
    scores: dict[float, float] = {}  # freq: score
    frequencies = []
    term_scores = []
    terms = []
    for i, term in enumerate(row_terms):
        if term in term_freq:
            diff = cos[i] - adj[i]
            score = float(np.linalg.norm(diff))
            frequency = float(term_freq[term])  # Ensure frequency is a float
            frequencies.append(frequency)
            term_scores.append(score)
            terms.append(term)

    print("Sample of processed data:")
    for i in range(min(5, len(frequencies))):
        print(f"Term: {terms[i]}, Frequency: {frequencies[i]}, Score: {term_scores[i]}")

    # Create a Pandas DataFrame for Altair
    data = pd.DataFrame({
        "frequency": frequencies,
        "score": term_scores,
        "term": terms
    })
    
    print("\nDataFrame head:")
    print(data.head())
    print("\nDataFrame info:")
    print(data.info())
    print("\nFrequency range:", data['frequency'].min(), "to", data['frequency'].max())

    # Create the Altair scatterplot
    chart = alt.Chart(data).mark_circle(opacity=0.7).encode(
        x=alt.X('frequency', 
                scale=alt.Scale(type="log", 
                              domain=[0.1, data['frequency'].max()],
                              nice=False),
                title="Pretraining Term Frequency"),
        y=alt.Y('score', 
                scale=alt.Scale(domain=[data['score'].min(), data['score'].max()]),
                title="Term Causal Separability Score"),
        tooltip=['term', 'frequency', 'score']
    ).properties(
        title=f"Term Frequency vs Causal Separability Score - {model_name} - {param_model}" + (f" - {index}" if index else ""),
        width=600,
        height=400
    ).interactive()

    return chart

def generate_all_scatterplots(indices: list, model: str, ontologies: list):
    """Generate scatterplots for all models and indices, organizing them into a structured folder hierarchy.
    
    The function creates scatterplots for both Pythia and Olmo models, with different parameterizations
    and indices, saving them in a structured folder hierarchy.
    """
    param_model = "160M"
    param_model_olmo = "7B"
    step = "step143000"
    step_olmo = "step150000-tokens664B"
    model_name = "pythia"
    adj_p = torch.load(str(BIGSTORAGE_DIR / f"raymond/heatmaps-{model_name}/{param_model}/{param_model}-{step}-1.pt"), weights_only=False)
    cos_p = torch.load(str(BIGSTORAGE_DIR / f"raymond/heatmaps-{model_name}/{param_model}/{param_model}-{step}-2.pt"), weights_only=False)
    model_name = "olmo"
    adj_o = torch.load(str(BIGSTORAGE_DIR / f"raymond/heatmaps-{model_name}/{param_model_olmo}/{step_olmo}-1.pt"), weights_only=False)
    cos_o = torch.load(str(BIGSTORAGE_DIR / f"raymond/heatmaps-{model_name}/{param_model_olmo}/{step_olmo}-2.pt"), weights_only=False)

    #script_dir = pathlib.Path(__file__).parent.parent / "data/"
    script_dir = Path("/home/logan/persona-research-internship/llm_ontology/data/")
    row_terms_path = script_dir / "owl_row_terms/wordnet_row_terms.txt"
    #term_freq_path = script_dir / "term_frequencies/wordnet.txt-frequencies.json"

    #save_scatterplot(adj_p, cos_p, row_terms_path, term_freq_path, model_name="pythia", param_model=param_model)
    #indices = ["v4_olmoe-0125-1b-7b-instruct_llama", "v4_olmo-2-1124-13b-instruct_llama", "v4_olmo-2-0325-32b-instruct_llama", "v4_dolmasample_olmo"]
    #indices = ["v4_olmo-2-0325-32b-instruct_llama"]
    for index in indices:
        for ontology in ontologies:
            term_freq_path = script_dir / "term_frequencies/multi-word-frequencies_v4_dolmasample_olmo" / index / ontology
            print(f"\nProcessing {ontology} for {index}")
            
            # Create directory structure
            figures_dir = pathlib.Path(__file__).parent.parent / "figures"
            output_dir = figures_dir / model / index / ontology.replace("-frequencies.json", "")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create interactive scatterplot
            scatterplot = create_interactive_scatterplot(
                adj=adj_o,
                cos=cos_o,
                row_terms_path=row_terms_path,
                term_freq_path=term_freq_path,
                model_name=model,
                param_model=param_model_olmo,
                index=index
            )
            
            # Save the chart as an HTML file
            filepath = output_dir / f"interactive_frequency_score_scatterplot_{index}.html"
            scatterplot.save(filepath)
            print(f"Interactive scatterplot saved to: {filepath}")

def generate_all_static_scatterplots(indices: list, model: str, ontologies: list):
    """Generate static scatterplots for all models and indices, organizing them into a structured folder hierarchy.
    
    Args:
        indices: List of model indices to process
        model: Model name (e.g., "olmo")
        ontologies: List of ontology filenames to process
    """
    param_model = "160M"
    param_model_olmo = "7B"
    step = "step143000"
    step_olmo = "step150000-tokens664B"
    model_name = "pythia"
    adj_p = torch.load(str(BIGSTORAGE_DIR / f"raymond/heatmaps-{model_name}/{param_model}/{param_model}-{step}-1.pt"), weights_only=False)
    cos_p = torch.load(str(BIGSTORAGE_DIR / f"raymond/heatmaps-{model_name}/{param_model}/{param_model}-{step}-2.pt"), weights_only=False)
    model_name = "olmo"
    adj_o = torch.load(str(BIGSTORAGE_DIR / f"raymond/heatmaps-{model_name}/{param_model_olmo}/{step_olmo}-1.pt"), weights_only=False)
    cos_o = torch.load(str(BIGSTORAGE_DIR / f"raymond/heatmaps-{model_name}/{param_model_olmo}/{step_olmo}-2.pt"), weights_only=False)

    script_dir = Path("/home/logan/persona-research-internship/llm_ontology/data/")
    row_terms_path = script_dir / "owl_row_terms/wordnet_row_terms.txt"

    # Generate static scatterplots
    for index in indices:
        for ontology in ontologies:
            term_freq_path = script_dir / "term_frequencies/multi-word-frequencies_v4_dolmasample_olmo" / index / ontology
            print(f"\nProcessing {ontology} for {index}")
            
            # Create directory structure
            figures_dir = pathlib.Path(__file__).parent.parent / "figures"
            output_dir = figures_dir / model / index / ontology.replace("-frequencies.json", "")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save static scatterplot
            save_scatterplot(
                adj=adj_o,
                cos=cos_o,
                row_terms_path=row_terms_path,
                term_freq_path=term_freq_path,
                model_name=model,
                param_model=param_model_olmo,
                index=index
            )
            print(f"Static scatterplot saved in: {output_dir}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate frequency score scatterplots for language models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["pythia", "olmo", "both"],
        default="olmo",
        help="Model type to generate plots for"
    )
    
    parser.add_argument(
        "--param-model-pythia",
        type=str,
        default="160M",
        help="Parameter model for Pythia"
    )
    
    parser.add_argument(
        "--param-model-olmo",
        type=str,
        default="7B",
        help="Parameter model for OLMo"
    )
    
    parser.add_argument(
        "--step-pythia",
        type=str,
        default="step143000",
        help="Step for Pythia model"
    )
    
    parser.add_argument(
        "--step-olmo",
        type=str,
        default="step150000-tokens664B",
        help="Step for OLMo model"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Data directory. Defaults to ../data relative to script or /home/logan/persona-research-internship/llm_ontology/data/"
    )
    
    parser.add_argument(
        "--row-terms-path",
        type=str,
        help="Path to row terms file. Defaults to {data_dir}/owl_row_terms/wordnet_row_terms.txt"
    )
    
    parser.add_argument(
        "--term-freq-path",
        type=str,
        help="Path to term frequency file. Defaults to {data_dir}/term_frequencies/wordnet.txt-frequencies.json"
    )
    
    parser.add_argument(
        "--indices",
        type=str,
        nargs="+",
        default=["v4_olmo-2-0325-32b-instruct_llama"],
        help="Indices to process"
    )
    
    parser.add_argument(
        "--ontologies",
        type=str,
        nargs="+",
        help="Specific ontologies to process. If not specified, discovers all available ontologies"
    )
    
    parser.add_argument(
        "--plot-type",
        type=str,
        choices=["static", "interactive", "both"],
        default="interactive",
        help="Type of plots to generate"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for figures. Defaults to ../figures relative to script"
    )
    
    parser.add_argument(
        "--user",
        type=str,
        default="raymond",
        help="User name for data paths"
    )
    
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    
    # Set up data directory
    if args.data_dir:
        script_dir = Path(args.data_dir)
    else:
        try:
            script_dir = pathlib.Path(__file__).parent.parent / "data"
            if not script_dir.exists():
                script_dir = Path("/home/logan/persona-research-internship/llm_ontology/data/")
        except:
            script_dir = Path("/home/logan/persona-research-internship/llm_ontology/data/")
    
    # Set up paths
    if args.row_terms_path:
        row_terms_path = Path(args.row_terms_path)
    else:
        row_terms_path = script_dir / "owl_row_terms/wordnet_row_terms.txt"
    
    if args.term_freq_path:
        term_freq_path = Path(args.term_freq_path)
    else:
        term_freq_path = script_dir / "term_frequencies/wordnet.txt-frequencies.json"
    
    # Load model data
    if args.model_type in ["pythia", "both"]:
        logger.info(f"Loading Pythia model data: {args.param_model_pythia}, {args.step_pythia}")
        adj_p = torch.load(str(BIGSTORAGE_DIR / f"{args.user}/heatmaps-pythia/{args.param_model_pythia}/{args.param_model_pythia}-{args.step_pythia}-1.pt"), weights_only=False)
        cos_p = torch.load(str(BIGSTORAGE_DIR / f"{args.user}/heatmaps-pythia/{args.param_model_pythia}/{args.param_model_pythia}-{args.step_pythia}-2.pt"), weights_only=False)
    
    if args.model_type in ["olmo", "both"]:
        logger.info(f"Loading OLMo model data: {args.param_model_olmo}, {args.step_olmo}")
        adj_o = torch.load(str(BIGSTORAGE_DIR / f"{args.user}/heatmaps-olmo/{args.param_model_olmo}/{args.step_olmo}-1.pt"), weights_only=False)
        cos_o = torch.load(str(BIGSTORAGE_DIR / f"{args.user}/heatmaps-olmo/{args.param_model_olmo}/{args.step_olmo}-2.pt"), weights_only=False)
    
    # Determine ontologies
    if args.ontologies:
        ontologies = args.ontologies
    else:
        # Auto-discover ontologies from the first index
        base_path = script_dir / "term_frequencies/multi-word-frequencies_v4_dolmasample_olmo" / args.indices[0]
        if base_path.exists():
            ontologies = [f.name for f in base_path.glob("*multi.txt-frequencies.json")]
            logger.info(f"Discovered ontologies: {ontologies}")
        else:
            logger.warning(f"Auto-discovery path does not exist: {base_path}")
            ontologies = []
    
    # Generate plots based on model type
    if args.model_type == "pythia" and 'adj_p' in locals():
        if args.plot_type in ["static", "both"]:
            save_scatterplot(adj_p, cos_p, row_terms_path, term_freq_path, model_name="pythia", param_model=args.param_model_pythia)
    
    if args.model_type in ["olmo", "both"] and 'adj_o' in locals():
        if ontologies:
            if args.plot_type in ["interactive", "both"]:
                generate_all_scatterplots(indices=args.indices, model="olmo", ontologies=ontologies)
            if args.plot_type in ["static", "both"]:
                generate_all_static_scatterplots(indices=args.indices, model="olmo", ontologies=ontologies)
        else:
            # Fallback to single scatterplot
            if args.plot_type in ["static", "both"]:
                save_scatterplot(adj_o, cos_o, row_terms_path, term_freq_path, model_name="olmo", param_model=args.param_model_olmo)
            if args.plot_type in ["interactive", "both"]:
                scatterplot = create_interactive_scatterplot(
                    adj=adj_o,
                    cos=cos_o,
                    row_terms_path=row_terms_path,
                    term_freq_path=term_freq_path,
                    model_name="olmo",
                    param_model=args.param_model_olmo
                )
                
                figures_dir = Path(args.output_dir) if args.output_dir else pathlib.Path(__file__).parent.parent / "figures"
                figures_dir.mkdir(parents=True, exist_ok=True)
                filepath = figures_dir / f"interactive_frequency_score_scatterplot_olmo_{args.param_model_olmo}.html"
                scatterplot.save(filepath)
                logger.info(f"Interactive scatterplot saved to: {filepath}")
        # Save the chart as an HTML file
        figures_dir = pathlib.Path(__file__).parent.parent / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        #filepath = figures_dir / f"interactive_frequency_score_scatterplot_example_{indices}.html"
        filepath = figures_dir / f"interactive_frequency_score_scatterplot_example.html"
        scatterplot.save(filepath)
        print(f"Interactive scatterplot saved to: {filepath}")
    

if __name__ == "__main__":
    main()
