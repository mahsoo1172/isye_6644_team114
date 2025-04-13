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

def single_stage(n = 12, k =  3, Pstar = 0.75, thetastar = 2, n_reps=1):
    """

    """
    rand_gen = np.random.default_rng()
    rand_nums = rand_gen.random(size=k)
    print('random numbers: ', rand_nums)
    probs = rand_nums / rand_nums.sum()
    print('probabilities: ', probs)
    
    replications = np.random.multinomial(n, probs, size=n_reps)
    print(replications)
    
    best = np.argmax(replications, axis=1)
    print(best)
    result = np.mean(best==np.argmax(probs))
    
    print(result)
    print(best)

def seq(k = 3, Pstar = 0.75, thetastar = 1.4, n_reps = 52, curtail_only=False):
    """

    """
    
    rand_gen = np.random.default_rng()
    rand_nums = rand_gen.random(size=k)
    print('random numbers: ', rand_nums)
    probs = rand_nums / rand_nums.sum()
    print('probabilities: ', probs)
    
    best_arr = np.zeros(shape=(n_reps,k))
    
    for r in range(1, n_reps+1):
        print('r: ', r)
    
        trial_pick = rand_gen.choice(k, p=probs)
        best_arr[r-1,trial_pick] = 1
            
        run_tally = best_arr.sum(axis=0)
        print(run_tally)
    
        arr_sort = run_tally.argsort()
        
        if run_tally[arr_sort[-1]] == run_tally[arr_sort[-2]]:
            print('THERE WAS A TIE!!!!!!!!')
            tiebreak = rand_gen.uniform()
            if tiebreak < 0.5:
                best_idx = arr_sort[-1]
                second_place = arr_sort[-2]
            if tiebreak >= 0.5:
                best_idx = arr_sort[-2]
                second_place = arr_sort[-1]

    
        else:
            best_idx = arr_sort[-1]
            second_place = arr_sort[-2]
    
        print(best_idx, second_place)
    
        recip = 1 / thetastar
        clear_win = 0
    
        for idx in arr_sort[:-1]:
            not_first = arr_sort[idx]
            val = recip**(run_tally[best_idx] - run_tally[not_first])
            clear_win += val
    
        # print(clear_win)
        # print((1-Pstar)/Pstar)
    
        if (run_tally[best_idx] - run_tally[second_place]) >= n_reps - r:
            print('2nd Place can only tie at best')
            print(run_tally[best_idx], run_tally[second_place], n_reps, r)
            break
    
        if not curtail_only:
            
            if clear_win <= (1 - Pstar) / Pstar:
                print('Best is clear winner!')
                break
            
            elif r == n_reps:
                print('All rows seen')
                break
    
    print(best_arr)
