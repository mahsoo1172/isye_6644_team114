import numpy as np
import pandas as pd
from pathlib import Path

def single_stage(input_system_samples, alpha, delta_star):
    """
    Single-stage selection procedure to select the normal distribution with the highest mean.
    Based on Bechofer's 1954 paper: "A Single-Sample Multiple Decision Procedure for Ranking
    Means of Normal Populations with known Variances"

    Assumptions:
    1. Variances are both known and the same across the different distributions that you're comparing.
    2. Due to the table lookup method, the inputted combination of # of input distributions, P*, and the delta_star
       must be in the lookup table.

    Parameters:
    input_system_samples (list): List of samples from each system you want to rank & select. Size is [k x M] where k is
                                 the # of systems, and M is the # of samples from each system.
    alpha (float): significance level
    delta_star (float): The indifference zone / standard deviation. Indifference zone represented in terms of
                        standard deviations. The indifference zone is the smallest difference between the means of the
                        systems that you deem important to detect.
    Returns:
    best_input_system_index (int): Index of the best system from the original 'input_system_samples' system list
    number_of_samples_necessary (int): Number of samples (final value of r) used in the procedure.
    """
    csv_filepath = Path(__file__).parent  # generalized way to get filepath of the lookup table csv. The csv will
                                          # be in the same directory as the procedure code.
    lookup_table = pd.read_csv(csv_filepath / 'normal_single_stage_table.csv')

    num_systems = len(input_system_samples)
    number_of_samples_necessary = -1
    try:
        number_of_samples_necessary = lookup_table.loc[(lookup_table['k'] == num_systems) &
                                      (lookup_table['P*'] == (1-alpha))][str(delta_star)].iloc[0]

    except Exception as e:
        print("Lookup failed. Number of systems 'k' and 'P*' must be valid entries in the lookup table.")
        print('Python Runtime Error: ', e)
        return

    print('Determining best system with single stage selection')
    best_input_system_index = -1
    best_sample_mean = 0
    for i in range(0, num_systems):
        sample_mean = np.mean(input_system_samples[i][0: number_of_samples_necessary])
        # print('sample_mean', sample_mean)
        if sample_mean > best_sample_mean:
            best_sample_mean = sample_mean
            best_input_system_index = i
    # print('best system:', best_input_system_index)
    # print('required # of input samples:', number_of_samples_necessary)
    return best_input_system_index, number_of_samples_necessary

def sequential(input_system_samples, alpha, indifference_zone, n0):
    """
    Sequential selection procedure to select the normal distribution with the highest mean.
    Based on Kim and Nelson's 2001 paper: "A Fully Sequential Procedure for Indifference-Zone
    Selection in Simulation"

    Assumptions:
    1. Set the constant 'c', to c=1. This is typically the best choice as stated in the Kim & Nelson paper.
    2. In this procedure, the variances are only estimated a single time, during the 'initialization' phase of the
       procedure.
    3. This code assumes that you have the same number of samples from each system.

    Parameters:
    input_system_samples (list): List of samples from each system you want to rank & select. Size is [k x M] where k is
                                 the # of systems, and M is the # of samples from each system.
    alpha (float): significance level
    indifference_zone (float): the smallest difference between the means of the systems that you deem important to
                               detect.
    n0 (int): first stage sample size - # of samples to use in 'initialization' phase to estimate the variance of
              each distribution.

    Returns:
    best_input_system_index (int): Index of the best system from the original 'input_system_samples' system list
    number_of_samples_necessary (int): Number of samples (final value of r) used in the procedure.
    """

    print('# # # # # # Sequential removal process currently running: # # # # # # ')
    # Initializing input constants
    max_num_input_samples = len(input_system_samples[0])  # total number of samples from each input system
    I = list(range(len(input_system_samples)))  # I is the indices of the original input systems

    # First Stage: Initialization Phase ################################################################################
    # Purpose: Calculating the 'sample variance of the difference btwn systems i and l' to use in the screening phase. #

    # Calculating 'eta' and h^2
    c = 1       # 'c' constant in the Kim & Nelson procedure. Assumed c = 1
    k = len(I)  # number of systems to test
    eta_intermediate_term = ((2*alpha) / (k-1))**(-2/(n0-1))
    eta = (1/2) * (eta_intermediate_term - 1)
    h2 = 2 * c * eta * (n0 - 1)

    # Calculating sample variance of the difference between systems i and l
    N_il_list = []
    S_il2_array = np.zeros((k, k))  # array to store sample variance terms, S_il2. The [i, i] diagonal terms
                                    # won't be used.

    # Iterate through each combination of the input systems, indices i & l.
    for i in range(0, k):
        for l in range(0, k):
            if l != i:   # When i != l, you are comparing two different systems. Calculate the S_il2 term:
                         # S_il2 = the sample variance of the difference between systems i and l.
                system_diff_sample_variance = 0  # initializing(S_il)^2

                for j in range(0, n0):  # iterate through all samples up to n0. n0 is the first-stage sample size
                                        # set by the user.
                    sample_diff = input_system_samples[i][j] - input_system_samples[l][j]  # X_ij - X_il

                    # Xbar_i(n0) - Xbar_l(n0)
                    mean_diff = np.mean(input_system_samples[i][0:n0]) - np.mean(input_system_samples[l][0:n0])
                    system_diff_term = (sample_diff - mean_diff)**2
                    system_diff_sample_variance += system_diff_term

                S_il2_term = 1/(n0-1) * system_diff_sample_variance
                S_il2_array[i][l] = S_il2_term  # store the sample variance of system diff to use in screening phase

                N_il = int(h2*system_diff_sample_variance / indifference_zone**2)
                N_il_list.append(N_il)

    Ni = max(N_il_list)
    # Initialization Special End Case:
    number_of_samples_necessary = 0
    if n0 > (Ni + 1):
        print('Early Initialization Stoppping Rule ( n0 > (Ni+1) )')
        best_input_system_index = -1
        best_sample_mean = 0
        for i in range(0, k):
            sample_mean = np.mean(input_system_samples[i])
            if sample_mean > best_sample_mean:
                best_input_system_index = i
        # print('best system:', best_input_system_index)
    # Second Stage: Screening Phase ####################################################################################
    # Purpose: Remove systems based on Sample Mean - W_il checks #######################################################
    else:
        r = n0  # r is the current # of samples used from each input system in the screening process.
        best_found = False
        # Continue the screening process until best_found == True, or r > max_num_input_samples.
        # the r > max_num_input_samples case is handled further below.
        while not best_found:
            k_iter = len(I)  # current # of systems remaining
            systems_to_remove = []  # check if systems need to be removed

            # Screening loop: Between each combination of system 'i' and 'l', calculate sample mean Xbar_i, Xbar_l, and
            # the W_il term.
            # Removal rule: Remove system 'i' if Xbar_i < (Xbar_l - W_il) for any system 'l'.
            # If there is more than 1 system remaining at the end of the loop, add in one additional sample r+=1 and
            # repeat the Xbar_i < (Xbar_l - W_il) check until 1 system remains.

            for i in range(0, k_iter): # iterate through all systems 'i' and 'l'
                sys_i = I[i]
                sample_mean_X_i = np.mean(input_system_samples[sys_i][0:r])  # calc Xbar_i up to 'r' samples
                for l in range(0, k_iter):
                    if l != i:
                        sys_l = I[l]
                        sample_mean_X_l = np.mean(input_system_samples[sys_l][0:r])  # calc Xbar_l  up to 'r' samples
                        # W_il calc
                        W_il = indifference_zone/(2*c*r) * ((h2*S_il2_array[sys_i][sys_l]/(indifference_zone**2)) - r)
                        W_il = max(0, W_il)

                        if sample_mean_X_i < (sample_mean_X_l - W_il):
                            if sys_i not in systems_to_remove:  # only add to removal list if sys_i not already found.
                                                                # Prevents duplicates from being added to removal list.
                                systems_to_remove.append(sys_i)
                                print('remove system:', sys_i)
                                # print('r', r)

            if len(systems_to_remove) > 0:  # If systems_to_remove list is not empty, then remove systems from I.
                for bad_sys in systems_to_remove:
                    I.remove(bad_sys)
                    # print('I:', I)

            if len(I) == 1:
                best_found = True
            r = r+1
            # Screening Special Case: r >= max_num_input_samples
            # If you run out of samples, determine best system by which one has the highest sample mean from the
            # list of remaining systems at this point in the procedure.
            if r == max_num_input_samples:
                print('Screening early stopping case: exceeded max # of samples available')
                best_input_system_index = -1
                best_sample_mean = 0
                for i in range(0, len(I)):
                    sys_index = I[i]
                    sample_mean = np.mean(input_system_samples[sys_index])
                    # print('sample_mean', sample_mean)
                    if sample_mean > best_sample_mean:
                        best_sample_mean = sample_mean
                        best_input_system_index = sys_index
                # print('best system:', best_input_system_index)
                # print('num samples needed:', r)
                return best_input_system_index, number_of_samples_necessary
        print('# # # # # # Sequential removal process complete. # # # # # # # # # # ')
        best_input_system_index =I[0]
        number_of_samples_necessary = r
        # print('best system:', best_input_system_index)
        # print('num samples needed:', r)

    return best_input_system_index, number_of_samples_necessary

def generate_normal_distrib_samples(num_distributions, num_samples_to_generate,
                                    randomized_mean_range=(0, 50),
                                    randomized_variance_range=(0.1, 100),
                                    random_seed=0,
                                    rounding=None,
                                    ):
    """
    Function to generate samples from multiple normal distributions. These samples are used test the single-stage
    and sequential R&S procedures. Normal distributions with a common variance can be set. Otherwise, both the
    mean and the variance of the distributions will be unifomrly randomly selected from the 'mean_range' and
    'variance_range' arguments.
    Assumptions:

    Parameters:
    num_distributions (int): Number of normal distributions to generate samples for. This represents the number of
                             systems that you'd want to rank and select.
    num_samples_to_generate (int): Number of random samples to generate from each distribution.
    randomized_mean_range (tuple (float, float)): lower and upper range for uniform random selection of the normal
                                                  distribution's mean
    randomized_variance_range (tuple (float, float)): lower and upper range for uniform random selection of the normal
                                                      distribution's variance
    random_seed (int): Random seed to set for debugging/experiment replication purposes.
    rounding (int): # of decimal places to truncate to. Used for cleaner print statements of the randomized
                   distribution means and variances.

    Returns:
    output_distribution_samples_list (list, int): List of size [num_distributions X num_samples_to_generate]
    output_distribution_definition_list (list, tuple(float, float) ): List of size [num_distributions X 1]
                                                                      Each tuple contains the mean and variance of the
                                                                      distribution. Used for validating R&S results.
    """
    # Setting random seed for replicating results
    np.random.seed(random_seed)
    # Setting Variables
    output_distribution_samples_list = []  # initializing output sample list
    output_distribution_definition_list = []
    # retrieving mean low/high lim from function input
    mean_low_limit = randomized_mean_range[0]
    mean_high_limit = randomized_mean_range[1]
    # retrieving variance low/high lim from function input
    variance_low_limit = randomized_variance_range[0]
    variance_high_limit = randomized_variance_range[1]

    for i in range(num_distributions):
        random_mean = np.random.uniform(mean_low_limit, mean_high_limit)
        random_variance = np.random.uniform(variance_low_limit, variance_high_limit)
        if rounding:
            random_mean = round(random_mean, rounding)
            random_variance = round(random_variance, rounding)
        random_normal_samples = np.random.normal(loc=random_mean,
                                                 scale=np.sqrt(random_variance),
                                                 size=num_samples_to_generate)
        actual_distribution_def = {
            "mean": random_mean,
            "variance": random_variance,
        }
        output_distribution_samples_list.append(random_normal_samples)
        output_distribution_definition_list.append(actual_distribution_def)

    return output_distribution_samples_list, output_distribution_definition_list