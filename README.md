# KLT-ermm
Runs Detran/Serment using python incorperating various basis sets

The program in run by calling main.py
This will call the module parameter.py that initializes the global variables
The common functions are located in common.py
The common module will import the problem specific functions from files of the name "problem_####.py"

Currently 3 test problems are implemented including '10-pin', 'bwrcore', and 'c5g7'
