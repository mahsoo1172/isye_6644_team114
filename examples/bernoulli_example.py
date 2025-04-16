import sys
import numpy as np
from pathlib import Path
# Steps to import properly across different environments.
parent_dir_path = Path(__file__).resolve().parent.parent  # Get the abs path of the 'isye_6644_team114' parent dir.
sys.path.append(str(parent_dir_path))  # add the parent path to the runtime path.
from ranking_selection_procedures import rhinott_two_stage_procedure, kiefer_wolfowitz_sequential_procedure

# Two-Stage Bernoulli Example - Rinott's Procedure
k = 4
delta = 0.05
P_star = 0.95
n0 = 10

best, sample_sizes, estimates = rhinott_two_stage_procedure(k, delta, P_star, n0)

print("\nSelected best option:", best)
print("Final sample sizes per option:", sample_sizes)
print("Final success rate estimates:", np.round(estimates, 3))

# Sequentail Bernoulli Example - Kiefer-Wolfowitz Procedure
k = 3
p_true = np.array([0.7, 0.8, 0.75])  # True success probabilities for 3 alternatives
n0 = 10  # Initial sample size
delta = 0.05  # Minimum detectable difference
max_samples = 100  # Max total samples to avoid infinite loop

best, sample_sizes, sample_means = kiefer_wolfowitz_sequential_procedure(k, delta, n0, max_samples, probs=None)

print("Best alternative:", best)
print("Final sample sizes:", sample_sizes)
print("Final success rate estimates:", sample_means)