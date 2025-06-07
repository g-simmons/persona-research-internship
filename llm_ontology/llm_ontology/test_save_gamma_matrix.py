from llm_ontology.compute_g_matrices import save_gamma_matrix

if __name__ == "__main__":
    save_gamma_matrix("allenai/OLMo-1.4B", "main", "gabe", fast=False)

    # save_gamma_matrix("allenai/OLMo-1.4B", "main", "gabe", fast=True)