import sys
from pathlib import Path
parent_dir_path = Path(__file__).resolve().parent.parent  # Get the abs path of the 'isye_6644_team114' parent dir.
sys.path.append(str(parent_dir_path))  # add the parent path to the runtime path.
import ranking_selection_procedures.multinomial as multinomial

#Single-stage
#An example of using the Single-stage procedure to find the most probably option from a complete set of trials is contained in  /isye_6644_team114/ranking_selection_procedures/multinomial_example.py 
#In the example, for reproducibility, a seed number is passed. The parameters are k=3, P*=0.75, and q*=2. 
#With these parameters, the function will reference a dictionary created with values from Goldsman [1] to determine n, the minimum number of trials. 
#A function argument, verbose, is set to True to include print statements as part of the results.
#The function will randomly generate k number of probabilities and then create a sample of trials based on these probabilities. 
#This is to produce unknown probabilities similar to the real world, but can be seen once the function completes. 
#The most probably value is determined by the winner of most trials, with randomization in place to break ties.  

#Curtailment/Sequential
#The function will randomly generate k number of probabilities and then create one sample trial at a time based on these probabilities. 
#This is to produce unknown probabilities similar to the real world, but can be seen once the function completes. 
#The most probable value is determined by the winner of most trials after n0 trials, 2nd place can at best tie 1st place (Y[1] – Y[2] ≥ n0 – m).  
#To use the procedure, call multinomial.seq with curtail_only=True for curtailment procedure or curtail_only=False for sequential. 

if __name__ == '__main__':
    multinomial.single_stage(k =  3, pstar = 0.75, thetastar = 2, n_reps=1, verbose=True, seed=2025)
    print()
    multinomial.seq(k =  3, pstar = 0.75, thetastar = 2, verbose=True, seed=2025)
