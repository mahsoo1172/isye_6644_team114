
# ISYE 6644 Group 114 - Ranking and Selection Procedure Python Library

## Introduction:
This repository contains functions to perform ranking and selection (R&S) procedures on Normal, Bernoulli, and Multinomial procedures.
Examples of using each procedure are also provided. The library was created by Tabriz Bumpas, Chuck Cao, and Matthew Truong
for ISYE 6644 during the Spring 2025 semester.

The repository contains functions for the following procedures:
- Normal Distribution
  - Single-Stage (Bechhofer)
  - Sequential (Kim & Nelson)
- Bernoulli Distribution
  - Two-Stage (Rhinott)
  - Sequential (Kiefer & Wolfowitz)
- Multinomial Distribution 
  - Single-Stage (Bechhofer, Elmaghraby, & Morse)
  - Curtailment (Bechhofer & Kulkarni)
  - Sequential (Bechhofer & Goldsman)

## Python Requirements:
To use these R&S procedures, the following external libraries are required:
- scipy (recommended version = 1.14.1)
- numpy (recommended version = 2.1.1)
- pandas (recommended version = 2.2.3)

The procedures and examples were validated to run properly using the recommended library versions in **Python 3.11.5**.

## Structure
The library consists of two folders:
- **ranking_selection_procedures**: contains the .py procedure functions
- **examples**: contains .py scripts** that contain examples of using each procedure. 
### ranking_selection_procedures
Within the ranking_selection_procedures folder are the following files:
- bernoulli.py: Contains the kiefer_wolfowitz_sequential_procedure() and rhinott_two_stage_procedure() functions.
- multinomial.py: Contains the single_stage() and seq() functions. 
- normal.py: Contains the single_stage(), sequential(), and generate_normal_distrib_samples() functions.
- n0_dict.pickle: Used in multinomial procedure lookups.
- n_dict.pickle: Used in multinomial procedure lookups.
- normal_single_stage_table.csv: Used in single-stage normal procedure lookups.

These procedures can be imported into other .py scripts using the typical Python import syntax below. (Make sure to use the correct
directory path to the ranking_selection_procedures folder in your working environment.):
```
from ranking_selection_procedures.bernoulli import rhinott_two_stage_procedure
```

### examples
Within the examples folder are the following files:
- bernoulli_example.py
- multinomial_example.py
- normal_example.py

Each of these .py scripts contain examples along with descriptive print outputs going through each procedure.

You can run these scripts in your Python IDE, or through your terminal using the terminal commands below:
```
cd isye_6644_team114/examples/
python normal_example.py
```