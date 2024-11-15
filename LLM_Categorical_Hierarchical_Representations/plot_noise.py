#!/usr/bin/env python3

from plots import get_numpy_arrs
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt


steps = np.array([i for i in range(1, 145, 2)])

linear_scores_70M, causal_sep_score_70M, hierarchy_score_70M = get_numpy_arrs("scores_70M_old.txt")
linear_scores_160M, causal_sep_score_160M, hierarchy_score_160M = get_numpy_arrs("scores_160M.txt")
linear_scores_14B, causal_sep_score_14B, hierarchy_score_14B = get_numpy_arrs("scores_1.4B.txt")        #1.4B, not 14B
linear_scores_28B, causal_sep_score_28B, hierarchy_score_28B = get_numpy_arrs("scores_2.8B_old.txt")
linear_scores_12B, causal_sep_score_12B, hierarchy_score_12B = get_numpy_arrs("scores_12B.txt")


def model_func(x, a, b, c, d):
    return a*np.log(b*x + c) + d
    # return a*np.exp(b*(x + c)) + d

def model_func_exp(x, a, b, c, d):
    return a*np.exp(b*(x + c)) + d

def get_params(x, y, func, hier = False):
    if hier:
        popt, pcov = curve_fit(func, x, y, p0 = [-0.1637, 77.31414703, -77.30983708, 15.92394351], maxfev=100000)
    else:
        popt, pcov = curve_fit(func, x, y, maxfev=100000)
    print(popt)
    return popt

# NOISE CALCULATIONS
def get_noise(x, y, params, func):
    total = 0
    for i in range(len(x)):
        total += (y[i] - func(x[i], *params)) ** 2

    return total


params_70M = get_params(steps, hierarchy_score_70M, model_func, True)
params_160M = get_params(steps, hierarchy_score_160M, model_func, True)
params_14B = get_params(steps, hierarchy_score_14B, model_func, True)
params_28B = get_params(steps, hierarchy_score_28B, model_func, True)
params_12B = get_params(steps, hierarchy_score_12B, model_func, True)

plt.plot(steps, model_func(steps, *params_70M))
plt.plot(steps, model_func(steps, *params_160M))
plt.plot(steps, model_func(steps, *params_14B))
plt.plot(steps, model_func(steps, *params_28B))
plt.plot(steps, model_func(steps, *params_12B))

plt.plot(steps, hierarchy_score_70M)
plt.plot(steps, hierarchy_score_160M)
plt.plot(steps, hierarchy_score_14B)
plt.plot(steps, hierarchy_score_28B)
plt.plot(steps, hierarchy_score_12B)

plt.savefig(f"plots/test.png")
plt.clf()




print(get_noise(steps, hierarchy_score_70M, params_70M, model_func))
print(get_noise(steps, hierarchy_score_160M, params_160M, model_func))
print(get_noise(steps, hierarchy_score_14B, params_14B, model_func))
print(get_noise(steps, hierarchy_score_28B, params_28B, model_func))
print(get_noise(steps, hierarchy_score_12B, params_12B, model_func))




params_70M = get_params(steps, causal_sep_score_70M, model_func)
params_160M = get_params(steps, causal_sep_score_160M, model_func)
params_14B = get_params(steps, causal_sep_score_14B, model_func)
params_28B = get_params(steps, causal_sep_score_28B, model_func)
params_12B = get_params(steps, causal_sep_score_12B, model_func)

plt.plot(steps, model_func(steps, *params_70M))
plt.plot(steps, model_func(steps, *params_160M))
plt.plot(steps, model_func(steps, *params_14B))
plt.plot(steps, model_func(steps, *params_28B))
plt.plot(steps, model_func(steps, *params_12B))

plt.plot(steps, causal_sep_score_70M)
plt.plot(steps, causal_sep_score_160M)
plt.plot(steps, causal_sep_score_14B)
plt.plot(steps, causal_sep_score_28B)
plt.plot(steps, causal_sep_score_12B)

plt.savefig(f"plots/test2.png")
plt.clf()


print(get_noise(steps, causal_sep_score_70M, params_70M, model_func))
print(get_noise(steps, causal_sep_score_160M, params_160M, model_func))
print(get_noise(steps, causal_sep_score_14B, params_14B, model_func))
print(get_noise(steps, causal_sep_score_28B, params_28B, model_func))
print(get_noise(steps, causal_sep_score_12B, params_12B, model_func))





