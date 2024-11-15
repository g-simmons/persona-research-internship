#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

def get_numpy_arrs(txt_dir: str):
    with open(txt_dir, "r") as f:
        scores = f.readlines()
        scores = list(map(lambda x: x[:-1], scores))
        scores = list(map(lambda x: x.split(", "), scores))
        scores = list(map(lambda x: list(map(float, x)), scores))
        
        linear_scores = []
        causal_sep_score = []
        hierarchy_score = []
        for score in scores:
            linear_scores.append(score[0])
            causal_sep_score.append(score[1])
            hierarchy_score.append(score[2])
    
    return np.array(linear_scores), np.array(causal_sep_score), np.array(hierarchy_score)



if __name__ == "__main__":
    # Init
    steps = np.array([i for i in range(1000, 145000, 2000)])
    steps = np.array([i for i in range(1, 145, 2)])
    parameter_models = ["70M", "160M", "1.4B", "2.8B", "12B"]

    linear_scores_70M, causal_sep_score_70M, hierarchy_score_70M = get_numpy_arrs("scores_70M_old.txt")
    linear_scores_160M, causal_sep_score_160M, hierarchy_score_160M = get_numpy_arrs("scores_160M.txt")
    linear_scores_14B, causal_sep_score_14B, hierarchy_score_14B = get_numpy_arrs("scores_1.4B.txt")        #1.4B, not 14B
    linear_scores_28B, causal_sep_score_28B, hierarchy_score_28B = get_numpy_arrs("scores_2.8B_old.txt")
    linear_scores_12B, causal_sep_score_12B, hierarchy_score_12B = get_numpy_arrs("scores_12B.txt")

    # Linear rep
    plt.plot(steps, linear_scores_70M)
    plt.plot(steps, linear_scores_160M)
    plt.plot(steps, linear_scores_14B)
    plt.plot(steps, linear_scores_28B)
    plt.plot(steps, linear_scores_12B)

    plt.xlabel("Steps")
    plt.ylabel("linear-rep-score")
    plt.title(f"Linear Representation Scores")
    plt.legend(parameter_models, title="Model Size", loc="upper right")

    plt.savefig(f"plots/linear_rep.png")
    plt.clf()


    # Causal sep
    plt.plot(steps, causal_sep_score_70M)
    plt.plot(steps, causal_sep_score_160M)
    plt.plot(steps, causal_sep_score_14B)
    plt.plot(steps, causal_sep_score_28B)
    plt.plot(steps, causal_sep_score_12B)

    plt.xlabel("Steps")
    plt.ylabel("causal-sep-score")
    plt.title(f"Causal Separation Scores")
    plt.legend(parameter_models, title="Model Size", loc="upper right")

    plt.savefig(f"plots/causal_sep.png")
    plt.clf()


    # Hierarchical
    plt.plot(steps, hierarchy_score_70M)
    plt.plot(steps, hierarchy_score_160M)
    plt.plot(steps, hierarchy_score_14B)
    plt.plot(steps, hierarchy_score_28B)
    plt.plot(steps, hierarchy_score_12B)

    plt.xlabel("Steps")
    plt.ylabel("hierarchical-score")
    plt.title(f"Hierarchical Scores")
    plt.legend(parameter_models, title="Model Size", loc="upper right")



    from scipy.optimize import curve_fit
    def model_func(x, a, b, c, d):
        return a*np.log(b*x + c) + d

    popt, pcov = curve_fit(model_func, steps, hierarchy_score_160M, maxfev=10000)

    print(popt)
    print(pcov)
    plt.plot(steps, model_func(steps, *popt), 'r-', label='Fit')



    plt.savefig(f"plots/hierarchical.png")
    plt.clf()



    # plt.savefig(f"plots/test.png")
    # plt.clf()
