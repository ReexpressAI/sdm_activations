# This is from https://github.com/aangelopoulos/conformal-prediction/blob/main/notebooks/correctness_checks.ipynb
# and included here for convenience.

import numpy as np
# import matplotlib.pyplot as plt
from scipy.stats import beta, betabinom
from scipy.optimize import brentq
import itertools

for alpha in [0.1, 0.05]:
    epsilons = [0.1, 0.05, 0.01, 0.005, 0.001]
    for epsilon in epsilons:
        def _condition(n):
            l = np.floor((n+1)*alpha)
            a = n + 1 - l
            b = l
            if (beta.ppf(0.05, a, b) < 1-alpha-epsilon) or (beta.ppf(0.95, a, b) > 1-alpha+epsilon):
                return -1
            else:
                return 1
        try:
            n_needed = int(np.ceil(brentq(_condition, np.ceil(1/alpha), 100000000000)))
            print(f"Alpha: {alpha}, epsilon: {epsilon}: n needed: {n_needed}")
        except ValueError:
            # Try to understand why it failed
            lower_bound = np.ceil(1/alpha)
            if _condition(lower_bound) == 1:
                print(f"Alpha: {alpha}, epsilon: {epsilon}: All n values satisfy the condition")
            else:
                print(f"Alpha: {alpha}, epsilon: {epsilon}: No n value satisfies the condition")


# Alpha: 0.1, epsilon: 0.1: n needed: 22
# Alpha: 0.1, epsilon: 0.05: n needed: 102
# Alpha: 0.1, epsilon: 0.01: n needed: 2491
# Alpha: 0.1, epsilon: 0.005: n needed: 9812
# Alpha: 0.1, epsilon: 0.001: n needed: 244390
# Alpha: 0.05, epsilon: 0.1: All n values satisfy the condition
# Alpha: 0.05, epsilon: 0.05: n needed: 46
# Alpha: 0.05, epsilon: 0.01: n needed: 1325
# Alpha: 0.05, epsilon: 0.005: n needed: 5263
# Alpha: 0.05, epsilon: 0.001: n needed: 129104

# See the tutorial for further discussion and relation to the DKW bounds for the prediction-conditional estimates.
