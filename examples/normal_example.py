import sys
from pathlib import Path
# Steps to import properly across different environments.
parent_dir_path = Path(__file__).resolve().parent.parent  # Get the abs path of the 'isye_6644_team114' parent dir.
sys.path.append(str(parent_dir_path))  # add the parent path to the runtime path.
import ranking_selection_procedures.normal as normal  # Including parent path in the current runtime path allows
                                                      # import from the ranking_selection_procedures folder

# Constants:
seed = 42
num_samples_to_generate = 10000   # setting arbitrarily high

# Single-Stage Procedure Example
print('Example 1: Single Stage Selection')
distribution_samples_ex1, actual_distribution_def_ex1 = normal.generate_normal_distrib_samples(
                                                            num_distributions=3,
                                                            num_samples_to_generate=num_samples_to_generate,
                                                            randomized_mean_range=(0, 20),
                                                            randomized_variance_range=(0.1, 15),
                                                            random_seed=seed,
                                                            rounding=3,
                                                              )

# Example 1: Compare 3 systems, requiring a confidence level (P* of 99%, alpha = 0.01) and a minimum detectable
#            difference in the means of 0.1 (delta star, indifference zone/stdev)
best_system_ex1, required_n_ex1 = normal.single_stage(input_system_samples=distribution_samples_ex1,
                                                      alpha=0.01,
                                                      delta_star=0.1)
print('PROCEDURE OUTPUT:')
print('BEST SYSTEM:', best_system_ex1)
print("MINIMUM # OF SAMPLES REQUIRED, n=", required_n_ex1)

print('\nGROUND TRUTH')
print('Example 1: actual distribution means/variances:')
for i in range(len(actual_distribution_def_ex1)):
    print('sys', i, ':', actual_distribution_def_ex1[i])

print('---------------------------------------------------------------------------------------------------------------')
print('Example 2: Sequential Selection')

distribution_samples_ex2, actual_distribution_def_ex2 = normal.generate_normal_distrib_samples(
                                                            num_distributions=7,
                                                            num_samples_to_generate=num_samples_to_generate,
                                                            randomized_mean_range=(0, 30),
                                                            randomized_variance_range=(0.1, 15),
                                                            random_seed=seed,
                                                            rounding=3,
                                                              )
# Example 2: Compare 7 systems, requiring a confidence level (P* of 95%, alpha = 0.05) and a minimum detectable
#            difference in the means of 0.5 (indifference zone). The 0.5 delta would be in units of the mean(ex: $, hrs)
best_system_ex2, required_n_ex2 = normal.sequential(input_system_samples=distribution_samples_ex2,
                                                    alpha=0.05,
                                                    indifference_zone=0.5,
                                                    n0=10)
print('PROCEDURE OUTPUT:')
print('BEST SYSTEM:', best_system_ex2)
print("MINIMUM # OF SAMPLES REQUIRED, n=", required_n_ex2)

print('\nGROUND TRUTH')
print('Example 2: actual distribution means/variances:')
for i in range(len(actual_distribution_def_ex2)):
    print('sys', i, ':', actual_distribution_def_ex2[i])
print('---------------------------------------------------------------------------------------------------------------')
#
#print('randomly generated true distributions:', actual_distribution_def)
#
#
# print('best distribution')