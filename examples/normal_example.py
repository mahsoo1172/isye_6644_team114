import sys
from pathlib import Path
# Steps to import properly across different environments.
parent_dir_path = Path(__file__).resolve().parent.parent  # Get the abs path of the 'isye_6644_team114' parent dir.
sys.path.append(str(parent_dir_path))  # add the parent path to the runtime path.
import ranking_selection_procedures.normal as normal  # Including parent path in the current runtime path allows
                                                      # import from the ranking_selection_procedures folder

# make distributions

# Constants:
seed = 42
num_samples_to_generate = 1000

# # do single stage
print('Example 1: Single Stage Selection')


print('Example 2: Sequential Selection')
# print("Suppose we're comparing four different queueing systems and determining which one had the highest # of customers"
#       " processed during a day (24 hrs).")

distribution_samples, actual_distribution_def = normal.generate_normal_distrib_samples(
                                                            num_distributions=4,
                                                            num_samples_to_generate=num_samples_to_generate,
                                                            randomized_mean_range=(0, 20),
                                                            randomized_variance_range=(0.1, 30),
                                                            random_seed=seed,
                                                            rounding=3,
                                                              )

normal.single_stage(distribution_samples, 0.01, 0.1)

#
#print('randomly generated true distributions:', actual_distribution_def)
#normal.sequential(distribution_samples, alpha=0.05, indifference_zone=0.5, n0=10)
#
# print('best distribution')