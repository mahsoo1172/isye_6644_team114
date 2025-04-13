import numpy as np
import pandas as pd

# TODO: implement single stage lookup table (or just direct calculation)
def single_stage(input_system_samples, alpha, indifference_zone):
    """
    Single-stage selection procedure to select the normal distribution with the highest mean.
    Based on Bechofer's 1954 paper: "A Single-Sample Multiple Decision Procedure for Ranking
    Means of Normal Populations with known Variances"

    Assumptions:
    1. Variances are both known and the same across the different distributions.

    Parameters:
    parameter1 (type): Description of the first parameter.
    parameter2 (type): Description of the second parameter.

    Returns:
    return_type: Description of the return value.
    """
    lookup_table = pd.read_csv('../ranking_selection_procedures/normal_single_stage_table.csv')
    num_systems = len(input_system_samples)
    required_n = -1
    try:
        required_n = lookup_table.loc[(lookup_table['k'] == num_systems) &
                                      (lookup_table['P*'] == (1-alpha))][str(indifference_zone)].iloc[0]
        print(required_n)
    except Exception as e:
        print("Lookup failed. Number of systems 'k' and 'P*' must be valid entries in the lookup table.")
        return

    print('Determining best system with single stage selection')
    best_input_system_index = -1
    best_sample_mean = 0
    for i in range(0, num_systems):
        sample_mean = np.mean(input_system_samples[i][0: required_n])
        print('sample_mean', sample_mean)
        if sample_mean > best_sample_mean:
            best_sample_mean = sample_mean
            best_input_system_index = i
    print('best system:', best_input_system_index)
    return best_input_system_index

def sequential(input_system_samples, alpha, indifference_zone, n0):
    """
    Sequential selection procedure to select the normal distribution with the highest mean.
    Based on Kim and Nelson's 2001 paper: "A Fully Sequential Procedure for Indifference-Zone
    Selection in Simulation"

    Assumptions:
    1. c=1, 'typically the best choice in Kin&Nelson procedure'
    2. in this version, the variances are only calculated one time, during the initialization phase.
    Parameters:
    parameter1 (type): Description of the first parameter.
    parameter2 (type): Description of the second parameter.

    Returns:
    return_type: Description of the return value.
    """
    #n0 = first stage sample size
    # Setup Stage
    max_num_input_samples = len(input_system_samples[0])
    #Initialization
    # Calculating 'eta' constant
    c = 1
    I = list(range(len(input_system_samples)))  # I is the indices of the original input systems
    print('I:' , I)
    k = len(I)
    eta_intermediate_term = ((2*alpha) / (k-1))**(-2/(n0-1))
    eta = (1/2) * (eta_intermediate_term - 1)
    h2 = 2 * c * eta * (n0 - 1)
    r = 0 # observation counter
    # Calculating sample variance of the difference between systems i and l
    system_diff_sample_variance = 0  # (S_ij)^2
    N_il_list = []
    S_il2_array = np.zeros((k, k))  # the [i, i] diagonal terms won't be used.
    for i in range(0, k):
        for l in range(0, k):
            if l != i:
                system_diff_sample_variance = 0
                for j in range(0, n0): # iterate through all samples up to n0
                    sample_diff = input_system_samples[i][j] - input_system_samples[l][j]
                    mean_diff = np.mean(input_system_samples[i][0:n0]) - np.mean(input_system_samples[l][0:n0])
                    system_diff_term = (sample_diff - mean_diff)**2
                    system_diff_sample_variance += system_diff_term

                S_il2_array[i][l] = system_diff_sample_variance  # store the sample variance of system diff to use in
                                                                 # screening phase

            N_il = int(h2*system_diff_sample_variance / indifference_zone**2)
            N_il_list.append(N_il)

    Ni = max(N_il_list)
    # print(Ni)
    # print(S_il2_array)
    # Screening: Check if the sample mean of system 'i' is >= to (sample mean of 'l' - W_il)
    # Special case 1:
    number_of_samples_necessary = 0
    if n0 > (Ni + 1):
        print('Early Initialization Stoppping Rule ( n0 > (Ni+1) )')
        best_input_system_index = -1
        best_sample_mean = 0
        for i in range(0, k):
            sample_mean = np.mean(input_system_samples[i])
            if sample_mean > best_sample_mean:
                best_input_system_index = i
        print('best system:', best_input_system_index)

    else:
        r = n0
        best_found = False
        while not best_found:
            # print("I:", I)
            k_iter = len(I)
            systems_to_remove = []
            for i in range(0, k_iter):
                sys_i = I[i]
                sample_mean_X_i = np.mean(input_system_samples[sys_i][0:r])
                for l in range(0, k_iter):
                    if l != i:
                        sys_l = I[l]
                        sample_mean_X_l = np.mean(input_system_samples[sys_l][0:r])
                        W_il = indifference_zone/(2*c*r) * ((h2*S_il2_array[sys_i][sys_l]/(indifference_zone**2)) - r)
                        W_il = max(0, W_il)
                        # print('meanX_i', sample_mean_X_i)
                        # print('meanX_l - W_il', (sample_mean_X_l - W_il))
                        if sample_mean_X_i < (sample_mean_X_l - W_il):
                            if sys_i not in systems_to_remove:  # only add to removal list
                                                                # if sys_i not already found.
                                                                # Could possibly be found in a previous W_il check
                                systems_to_remove.append(sys_i)
                                print('remove system:', sys_i)
                                print('r', r)
            if len(systems_to_remove) > 0:
                for bad_sys in systems_to_remove:
                    I.remove(bad_sys)
                    print('I:', I)

            if len(I) == 1:
                best_found = True
            r = r+1

            # Screening Special Case:
            if r == max_num_input_samples:
                print('Screening early stopping case: exceeded max # of samples available')
                best_input_system_index = -1
                best_sample_mean = 0
                for i in range(0, k):
                    sample_mean = np.mean(input_system_samples[i])
                    print('sample_mean', sample_mean)
                    if sample_mean > best_sample_mean:
                        best_sample_mean = sample_mean
                        best_input_system_index = i
                print('best system:', best_input_system_index)
                print('num samples needed:', r)
                return best_input_system_index, number_of_samples_necessary

        best_input_system_index =I[0]
        number_of_samples_necessary = r
        print('best system:', best_input_system_index)
        print('num samples needed:', r)
    return best_input_system_index, number_of_samples_necessary

def generate_normal_distrib_samples(num_distributions, num_samples,
                                         common_variance_flag=False, common_variance_val=None,
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
    parameter1 (type): Description of the first parameter.
    parameter2 (type): Description of the second parameter.

    Returns:
    return_type: Description of the return value.
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
                                                 size=num_samples)
        actual_distribution_def = {
            "mean": random_mean,
            "variance": random_variance,
        }
        output_distribution_samples_list.append(random_normal_samples)
        output_distribution_definition_list.append(actual_distribution_def)

    return output_distribution_samples_list, output_distribution_definition_list