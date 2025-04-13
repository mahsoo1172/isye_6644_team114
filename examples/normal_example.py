
import ranking_selection_procedures.normal as normal
# make distributions

# Constants:
seed = 42
num_samples_to_generate = 1000

# # do single stage
print('Example 1: Single Stage Selection')
print("Suppose we're comparing four different queueing systems and determining which one had the highest # of customers"
      " processed during a day (24 hrs).")

distribution_samples, actual_distribution_def = normal.generate_normal_distrib_samples(
                                                            num_distributions=4,
                                                            num_samples=num_samples_to_generate,
                                                            randomized_mean_range=(0, 20),
                                                            randomized_variance_range=(0.1, 30),
                                                            random_seed=seed,
                                                            rounding=3,
                                                              )

# normal.single_stage(distribution_samples, 0.05, 0.1)

#
print(actual_distribution_def)
# # print(actual_distribution_def[29])
# print('best distribution')
# # do sequential
normal.sequential(distribution_samples, 0.05, 1, 24)
#
# print('best distribution')