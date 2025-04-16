import time
import math
import numpy as np
import scipy.stats as stats
#%% Sequential Bernoulli - Kiefer Wolfowitz
def kiefer_wolfowitz_sequential_procedure(k, delta, n0, max_samples, seed=None, probs=None):
    """
    Kiefer-Wolfowitz Procedure to select the best alternative from k Bernoulli distributions.

    Parameters:
    - k: Number of alternatives (arms)
    - p_true: True success probabilities for each alternative (array)
    - n0: Initial sample size per alternative
    - delta: Minimum difference to detect between alternatives
    - alpha: Significance level (default 0.05 for 95% confidence)
    - max_samples: Maximum total samples to avoid infinite loops

    Returns:
    - best_index: Index of the best alternative
    - sample_sizes: Final sample sizes for each alternative
    - sample_means: Final sample means (estimated probabilities)
    """

    if seed:
        print('Setting random seed to {seed}...'.format(seed=seed))
        time.sleep(1)
        np.random.seed(seed)
    else:
        print('No random seed defined by the user')
        time.sleep(1)

    if probs is None:
        print('Defining a random array of probabilities...')
        time.sleep(1)
        # Randomize true success probabilities for simulation
        probs = np.random.uniform(0.5, 0.95, k)
        print(f"True success probabilities: {np.round(probs, 3)}")

    # Step 1: Initial Sampling (take n0 samples from each alternative)
    print('Generating initial sample...')
    time.sleep(1)
    samples = [list(np.random.binomial(1, p, n0)) for p in probs]  # n0 samples for each of k alternatives
    sample_means = np.array([np.mean(s) for s in samples])  # Calculate sample means for each alternative

    sample_sizes = np.full(k, n0)  # Initialize sample sizes (each alternative gets n0 samples)
    
    # Maximum samples to prevent infinite loop
    total_samples = k * n0

    # Step 2: Compare alternatives pairwise and identify the least favorable alternative
    print('Comparing alternate choices...')
    time.sleep(3)
    while total_samples < max_samples:
        best_index = np.argmax(sample_means)
        second_best_index = np.argsort(sample_means)[-2]

        # Check if the difference between the best and second-best is sufficiently large
        if sample_means[best_index] - sample_means[second_best_index] > delta:
            break  # We can stop if the difference is large enough

        # Otherwise, continue sampling the alternative with the smaller sample size
        # Find the alternative with the smaller sample size to sample more
        if sample_means[best_index] < sample_means[second_best_index]:
            to_sample = best_index
        else:
            to_sample = second_best_index

        # Take an additional sample for the alternative with the smaller sample size
        new_sample = np.random.binomial(1, probs[to_sample], size=1)
        samples[to_sample] += [new_sample[0]]  # Add the new sample
        sample_means[to_sample] = np.mean(samples[to_sample])  # Recalculate the sample mean

        # Update the sample size for that alternative
        sample_sizes[to_sample] += 1
        total_samples += 1  # Increment the total sample count

    # Step 3: Return the best alternative based on the sample means
    print('Optimal choice identified...')
    time.sleep(1)
    best_index = np.argmax(sample_means)
    
    print('Complete')
    return best_index, sample_sizes, sample_means

#%% Two-Stage Bernoulli - Rinott's Procedure
def rhinott_two_stage_procedure(k, delta, P_star, n0, seed=None, probs=None):
    """
    Two-stage best-option selection for Bernoulli (binary) distributions using 
    Rinott's rank and selection procedure.

    If no probs are provided by the user, the function will generate a
    random array of assumed probabilities for the number of "k" alternatives
    between 5% and 95%.

    If no seed is specified it will be randomly assigned. It is suggested to
    specify a random seed for repeatability.

    Parameters:
    - k: number of alternatives
    - delta: indifference-zone parameter
    - P_star: desired probability of correct selection
    - n0: initial sample size
    - seed: optional random seed
    - probs: true success probabilities for each option (optional)

    Returns:
    - best_index: selected option
    - sample_sizes: number of samples per option
    - final_estimates: estimated success probabilities
    """

    if seed:
        print('Setting random seed to {seed}...'.format(seed=seed))
        time.sleep(1)
        np.random.seed(seed)
    else:
        print('No random seed defined by the user')
        time.sleep(1)

    if probs is None:
        print('Defining a random array of probabilities...')
        time.sleep(1)
        # Randomize true success probabilities for simulation
        probs = np.random.uniform(0.05, 0.95, k)
        print(f"True success probabilities: {np.round(probs, 3)}")

    # Stage 1: Initial sampling
    print('Generating initial sample...')
    time.sleep(1)
    initial_data = np.random.binomial(1, probs[:, None], size=(k, n0))
    hat_p = initial_data.mean(axis=1)
    hat_var = hat_p * (1 - hat_p)

    # Critical z value for normal approx
    print('Calculating critical Z-Value...')
    time.sleep(1)
    z = stats.norm.ppf((1 + P_star) / 2)

    # Stage 2: Compute required sample sizes
    print('Calculating minimum sample size...')
    time.sleep(1)
    n_i = np.array([max(n0, math.ceil((2 * z**2 * v) / delta**2)) for v in hat_var])
    print(f"Required sample size: {n_i}")

    # Collect additional samples
    print('Sampling Additional Data...')
    final_data = []
    for i in range(k):
        print('Sampling from option: ', str(i))
        time.sleep(1)
        n_extra = n_i[i] - n0
        if n_extra > 0:
            extra = np.random.binomial(1, probs[i], n_extra)
            combined = np.concatenate([initial_data[i], extra])
        else:
            combined = initial_data[i]
        final_data.append(combined)
    time.sleep(1)

    # Final estimates
    print('Calculating optimal choice..')
    time.sleep(1)
    final_estimates = np.array([np.mean(x) for x in final_data])
    best_index = int(np.argmax(final_estimates))

    return best_index, n_i, final_estimates