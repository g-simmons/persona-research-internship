import json
import networkx as nx

import torch
from sklearn.covariance import ledoit_wolf


import logging
logger = logging.getLogger(__name__)


def get_categories(noun_or_verb = 'noun'):

    cats = {}
    if noun_or_verb == 'noun':
        with open('data/noun_synsets_wordnet_gemma.json', 'r') as f:
            for line in f:
                cats.update(json.loads(line))
        G = nx.read_adjlist("data/noun_synsets_wordnet_hypernym_graph.adjlist", create_using=nx.DiGraph())
    elif noun_or_verb == 'verb':
        with open('data/verb_synsets_wordnet_gemma.json', 'r') as f:
            for line in f:
                cats.update(json.loads(line))
        G = nx.read_adjlist("data/verb_synsets_wordnet_hypernym_graph.adjlist", create_using=nx.DiGraph())
    
    cats = {k: list(set(v)) for k, v in cats.items() if len(set(v)) > 50}
    # cats = {k: list(set(v)) for k, v in cats.items()}
    G = nx.DiGraph(G.subgraph(cats.keys()))


    reversed_nodes = list(reversed(list(nx.topological_sort(G))))
    for node in reversed_nodes:
        children = list(G.successors(node))
        if len(children) == 1:
            child = children[0]
            parent_lemmas_not_in_child = set(cats[node]) - set(cats[child])
            if len(list(G.predecessors(child))) == 1 or len(parent_lemmas_not_in_child) <5:
                grandchildren = list(G.successors(child))
                for grandchild in grandchildren:
                    G.add_edge(node, grandchild)
                G.remove_node(child)

    G = nx.DiGraph(G.subgraph(cats.keys()))
    sorted_keys = list(nx.topological_sort(G))

    cats = {k: cats[k] for k in sorted_keys}

    return cats, G, sorted_keys


def get_categories_ontology(noun_or_verb = 'noun'):

    cats = {}
    if noun_or_verb == 'noun':
        with open('data/ontologies/noun_synsets_ontology_pythia.json', 'r') as f:
            for line in f:
                cats.update(json.loads(line))
        G = nx.read_adjlist("data/ontologies/noun_synsets_ontology_hypernym_graph.adjlist", create_using=nx.DiGraph())
    elif noun_or_verb == 'verb':
        with open('data/verb_synsets_wordnet_gemma.json', 'r') as f:
            for line in f:
                cats.update(json.loads(line))
        G = nx.read_adjlist("data/verb_synsets_wordnet_hypernym_graph.adjlist", create_using=nx.DiGraph())
    
    # cats = {k: list(set(v)) for k, v in cats.items() if len(set(v)) > 50}
    cats = {k: list(set(v)) for k, v in cats.items() if len(set(v)) > 1}
    # nodes = list(cats.keys())
    # G = nx.DiGraph(G.subgraph(nodes))
    G = nx.DiGraph(G.subgraph(cats.keys()))


    reversed_nodes = list(reversed(list(nx.topological_sort(G))))
    for node in reversed_nodes:
        children = list(G.successors(node))
        if len(children) == 1:
            child = children[0]
            parent_lemmas_not_in_child = set(cats[node]) - set(cats[child])
            if len(list(G.predecessors(child))) == 1 or len(parent_lemmas_not_in_child) <5:
                grandchildren = list(G.successors(child))
                for grandchild in grandchildren:
                    G.add_edge(node, grandchild)
                G.remove_node(child)

    

    sorted_keys = list(nx.topological_sort(G))

    print(sorted_keys)
    cats = {k: cats[k] for k in sorted_keys}

    return cats, G, sorted_keys


def category_to_indices(category, vocab_dict):
    # for w in category:
        # print(w)
    indices = []
    for w in category:
        # print(w)
        indices.append(vocab_dict[w])
    
    return indices
    # return [vocab_dict[w] for w in category]

def category_to_indices_multi_word(category, vocab_dict):
    indices = []
    for w in category:
        words = w.split("_")
        indices.append([vocab_dict[word] for word in words])
    return indices

def get_category_embeddings(indices, unembed):
    embeddings_list = []
    logger.info("differences_embeddings")
    for index_list in indices:
        temp_tensor = torch.zeros(len(unembed[index_list[0]]))
        for i in index_list:
            # torch.add(temp_tensor, unembed[i])
            temp_tensor = temp_tensor + unembed[i]
        temp_tensor = temp_tensor / len(index_list)     # average over words

        # logger.info([torch.norm(temp_tensor - embed) for embed in embeddings_list])

        embeddings_list.append(temp_tensor)
    return torch.stack(tuple(embeddings_list))


def get_words_sim_to_vec(query: torch.tensor, unembed, vocab_list, k=300):
    similar_indices = torch.topk(unembed @ query, k, largest=True).indices.cpu().numpy()
    return [vocab_list[idx] for idx in similar_indices]

def estimate_single_dir_from_embeddings(category_embeddings):
    category_mean = category_embeddings.mean(dim=0)

    logger.info("category_embeddings")
    logger.info(category_embeddings)

    embeddingnan = torch.isnan(category_embeddings)
    logger.info("nan in category_embeddings")
    logger.info(True in embeddingnan)

    cov = ledoit_wolf(category_embeddings.cpu().numpy())
    logger.info("cov1")
    logger.info(cov)

    cov = torch.tensor(cov[0], device = category_embeddings.device)
    logger.info("cov2")
    logger.info(cov)

    covnan = torch.isnan(cov)
    logger.info("nan in cov")
    logger.info(True in covnan)
    

    logger.info("cov det")
    logger.info(torch.det(cov))

    pseudo_inv = torch.linalg.pinv(cov)
    lda_dir = pseudo_inv @ category_mean

    logger.info("pseudo_inv")
    logger.info(pseudo_inv)
    logger.info("category_mean")
    logger.info(category_mean)
    logger.info("lda_dir")
    logger.info(lda_dir)
    logger.info("LDA dir norm")
    logger.info(torch.norm(lda_dir))

    lda_dir = lda_dir / torch.norm(lda_dir)
    lda_dir = (category_mean @ lda_dir) * lda_dir

    return lda_dir, category_mean

def estimate_cat_dir(category_lemmas, unembed, vocab_dict, multi: bool):
    if multi:
        category_embeddings = get_category_embeddings(category_to_indices_multi_word(category_lemmas, vocab_dict), unembed)
    else:
        category_embeddings = unembed[category_to_indices(category_lemmas, vocab_dict)]
    
    lda_dir, category_mean = estimate_single_dir_from_embeddings(category_embeddings)
    
    return {'lda': lda_dir, 'mean': category_mean}



import inflect
p = inflect.engine()

def noun_to_gemma_vocab_elements(word, vocab_set):
    word = word.lower()
    plural = p.plural(word)
    add_cap_and_plural = [word, word.capitalize(), plural, plural.capitalize()]
    # add_space = ["▁" + w for w in add_cap_and_plural]
    add_space = [w for w in add_cap_and_plural]
    return vocab_set.intersection(add_space)


def get_animal_category(data, categories, vocab_dict, g):
    vocab_set = set(vocab_dict.keys())

    animals = {}
    animals_ind = {}
    animals_g = {}
    animals_token = {}

    for category in categories:
        animals[category] = []
        animals_ind[category] = []
        animals_g[category] = []
        animals_token[category] = []

    for category in categories:
        lemmas = data[category]
        for w in lemmas:
            animals[category].extend(noun_to_gemma_vocab_elements(w, vocab_set))
        
        for word in animals[category]:
            animals_ind[category].append(vocab_dict[word])
            animals_token[category].append(word)
            animals_g[category] = g[animals_ind[category]]
    return animals_token, animals_ind, animals_g