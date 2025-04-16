def example_format():
    """
    Single-stage selection procedure to select the normal distribution with the highest mean.
    Based on Bechhofer's 1954 paper: "A Single-Sample Multiple Decision Procedure for Ranking
    Means of Normal Populations with known Variances"

    Assumptions:

    Parameters:
    parameter1 (type): Description of the first parameter.
    parameter2 (type): Description of the second parameter.

    Returns:
    return_type: Description of the return value.
    """
    return

import pickle
import numpy as np
from scipy.stats import multinomial


def single_stage(k =  3, pstar = 0.75, thetastar = 1.4, n_reps=1, verbose=False, seed=None):
    """
    Demonstrates single-stage procedure for multinomial ranking and selection of most probable option. Generates n, the minimum number of trials needed with the given k, pstar, and thetastar

    Parameters:
    - k: Number of options (Default: 3, currently can take k=2 to k=5 for most cases)
    - pstar: Desired minimum probability of correctly selecting the underlying most probable option. (Default: 0.75, can take 0.75, 0.9, and 0.95 for most cases)
    - thetastar: Smallest ratio between the first and second best options that are worth detecting. (Default: 1.4, can take between 1.2 to 3 in increments of 0.2 for most cases)
    - n_reps: Number of experiments to run. (Default: 1)
    - verbose: Whether to provide print statements giving additional information. (Default: False)
    - seed: Random seed for random generator to have reproducible results. (Default: None)

    Returns:
    - best: The index of the best option found
    - n: The minimum number of trials determined based on provided k, pstar, and thetastar
    - probs: Probabilities that were randomly generated and unknown to compare with best option found
    """

    if seed is None:
        rand_gen = np.random.default_rng()
    else:
        rand_gen = np.random.default_rng(seed=seed)
    
    rand_nums = rand_gen.random(size=k) #Can't use uniform because need to equal 1, would be same process
    probs = rand_nums / rand_nums.sum()

    n_dict = pickle.load(open("n_dict.pickle", "rb"))
    n = n_dict[(k, pstar, thetastar)]
    
    replications = rand_gen.multinomial(n, probs, size=n_reps)

    if n_reps > 1:
        replications = np.sum(replications, axis=0)
    
    best = np.flatnonzero(replications==np.max(replications)) #from https://stackoverflow.com/questions/17568612/how-to-make-numpy-argmax-return-all-occurrences-of-the-maximum

    if len(best) > 1:
        rand_gen.choice(best)

    if verbose:
        if len(best) > 1:
            print('There was a tie for best and a winner was randomly selected as tiebreaker')
        print("n for given k, P*, and Theta*: ", n) 
        print('Winners: ', replications)
        print('Probabilities: ', probs)
        print('The best option [k=0...k] is: ', best)

    return best, n, probs



def seq(k = 3, pstar = 0.75, thetastar = 1.4, curtail_only=False, verbose=False, seed=None):
    """
    Demonstrates curtailment and sequential procedure for multinomial ranking and selection of most probable option. Generates n_0, the minimum number of trials needed with the given k, pstar, and thetastar.
    Creates a random trial winner one at a time until it reachs n_0 or one of the stopping conditions.

    Parameters:
    - k: Number of options (Default: 3, currently can take k=2 to k=6 for most cases)
    - pstar: Desired minimum probability of correctly selecting the underlying most probable option. (Default: 0.75, can take 0.75, 0.9, and 0.95 for most cases)
    - thetastar: Smallest ratio between the first and second best options that are worth detecting. (Default: 1.4, can take between 1.2 to 3 in increments of 0.2 for most cases)
    - curtail_only: Whether to only conduct the curtailment procedure only (False) or as part of the sequential procedure (True) (Default: False)
    - verbose: Whether to provide print statements giving additional information. (Default: False)
    - seed: Random seed for random generator to have reproducible results. (Default: None)

    Returns:
    - best: The index of the best option found
    - n: The minimum number of trials determined based on provided k, pstar, and thetastar
    - probs: Probabilities that were randomly generated and unknown to compare with best option found
    """

    if seed is None:
        rand_gen = np.random.default_rng()
    else:
        rand_gen = np.random.default_rng(seed=seed)
    
    rand_nums = rand_gen.random(size=k)
    probs = rand_nums / rand_nums.sum()

    n0_dict = pickle.load(open("n0_dict.pickle", "rb"))
    n_0 = n0_dict[(k, pstar, thetastar)]
    
    best_arr = np.zeros(shape=(n_0,k))
    
    for r in range(1, n_0+1):
    
        trial_pick = rand_gen.choice(k, p=probs)
        best_arr[r-1,trial_pick] = 1
            
        run_tally = best_arr.sum(axis=0)
    
        arr_sort = run_tally.argsort()
        
        check_ties = np.flatnonzero(run_tally==np.max(run_tally)) 
        #print(check_ties)
        
        if len(check_ties) > 1:
            #print('THERE WAS A TIE!!!!!!!!')
            best_idx, second_place = rand_gen.choice(check_ties, 2)
    
        else:
            best_idx = arr_sort[-1]
            second_place = arr_sort[-2]
    
        recip = 1 / thetastar
        clear_win = 0
    
        for idx in arr_sort[:-1]: #Even in case of ties, the math will remain same since last idx will be best value count
            not_first = arr_sort[idx]
            val = recip**(run_tally[best_idx] - run_tally[not_first])
            clear_win += val
    
        # print(clear_win)
        # print((1-pstar)/pstar)
    
        if (run_tally[best_idx] - run_tally[second_place]) >= n_0 - r:
            print('2nd Place can only tie at best')
            print('Best Win Count: ', run_tally[best_idx], '2nd Best Win Count: ', run_tally[second_place], 'n0: ', n_0, 'Current trial: ', r)
            break
    
        if not curtail_only:
        
            if clear_win <= (1 - pstar) / pstar:
                print('Best is clear winner!')
                break
            
            elif r == n_0:
                print('All rows seen')
                break
    
    if verbose:
        print("n_0 for given k, P*, and Theta*: ", n_0)
        print('Stopped at observation: ', r)
        print('Winners: ', run_tally)
        print('Probabilities: ', probs)
        print('The best option [k=0...k]: ', best_idx)
        print('best array: ', '\n', best_arr)

    return best_idx, n_0, probs



if __name__ == '__main__':
    single_stage(k =  3, pstar = 0.75, thetastar = 2, n_reps=1, verbose=True, seed=2025)
    print()
    seq(k =  3, pstar = 0.75, thetastar = 2, verbose=True, seed=2025)
