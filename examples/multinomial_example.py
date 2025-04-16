import sys
from pathlib import Path
# Steps to import properly across different environments.
parent_dir_path = Path(__file__).resolve().parent.parent  # Get the abs path of the 'isye_6644_team114' parent dir.
sys.path.append(str(parent_dir_path))  # add the parent path to the runtime path.
import ranking_selection_procedures.multinomial as multinomial


if __name__ == '__main__':
    multinomial.single_stage(k =  3, pstar = 0.75, thetastar = 2, n_reps=1, verbose=True, seed=2025)
    print()
    multinomial.seq(k =  3, pstar = 0.75, thetastar = 2, verbose=True, seed=2025)