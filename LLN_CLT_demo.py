# Quantitative Economics with Python
import random
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import t, beta, lognorm, expon, gamma, uniform, cauchy
from scipy.stats import gaussian_kde, poisson,binom, norm,chi2
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from scipy.linalg import inv, sqrtm

# classical LLN
# iid variables Kolmogorov's strong law
# X_1, X_2, ... X_n iid ~ F
# mu = E[X] = integral[xF](dx)
# X_bar = 1/n * sum(X_i)
# then, Prob(X_bar->mu as n-> inf) = 1

# Law of Large Number Demo
n = 100
distributions = {
    "student's t with 10 df ": t(10),
    "beta(2,2)": beta(2, 2),
    "lognormal": lognorm(0.5),
    "gamma(5,1/2)": gamma(5, scale=2),
    "poisson(4)": poisson(4),
    "exponential with lambda =1": expon(1)
}
num_plots = 3
fig, axes = plt.subplots(num_plots, 1, figsize=(8, 8))
bbox = (0., 1.02, 1, .102)
legend_args = {'ncol': 2, 'bbox_to_anchor': bbox, "loc": 3, 'mode': 'expand'}

plt.subplots_adjust(hspace=0.5)

for ax in axes:
    # pick a distribution
    name = random.choice(list(distributions.keys()))
    distribution = distributions.pop(name)

    # random draw
    data = distribution.rvs(n)

    # sample mean
    sample_mean = np.empty(n)
    for i in range(n):
        sample_mean[i] = np.mean(data[:i + 1])

    # plot
    ax.plot(list(range(n)), data, 'o', color='grey', alpha=0.5)
    axlabel = '$\\bar X_n$ for $x_i \sim$' + name
    ax.plot(list(range(n)), sample_mean, 'g-', lw=3, alpha=0.6, label=axlabel)
    m = distribution.mean()
    ax.plot(list(range(n)), [m] * n, 'k--', lw=1.5, label='$\mu$')
    ax.vlines(list(range(n)), m, data, lw=0.2)
    ax.legend(**legend_args)

plt.show(block=False)