from serment import *
from serment.serment_cartesian_tools import *
import sys
import timeit
import os
import parameters
import common

#------------------------------------------------------------------------------#
# PROBLEM PARAMETERS
#------------------------------------------------------------------------------#
# These are globally defined variables that control the rest of the program
# Editing the parameters outside of the preprogramed choices may cause error
# The selectable choices are shown in the comments to the right of the variable
# To add a new test problem, create a problem_%%%%%%.py file for the problem

# Max Task ID for 10-pin is 84
# Max Task ID for both is 186

# Set the mode to select the correct task ID numbers
mode = 'run'

task = os.environ['SGE_TASK_ID']
task = int(task) - 1

test_problem, nGroups, use_angular_moments, case, moment_order, config_number = parameters.init(mode, task)

if __name__ == "__main__":
    start = timeit.default_timer()
    ManagerERME.initialize(sys.argv)
    if mode == 'klt':
        common.KLT(case)
    elif mode == 'gendb':
        common.generate_database(case)
    elif mode == 'plot':
        if use_angular_moments is True and test_problem != 'c5g7':
            if nGroups == 238:
                common.angular_comparison(True)
            common.angular_comparison()
        if nGroups == 238:
            common.process_rf_data(True)
        common.process_rf_data()
    elif mode == 'refrf':
        common.run_ref_rf()
    elif mode == 'run':
        common.run(case)
    #create_dirs()
    #print test_problem, nGroups
    #run_reference(task)
    #print num_keff, kdel
    #run_1D_ref()
    #if int(os.environ['SGE_TASK_ID']) != 8 :
    #    run(case, True, num_keff, kdel)
    #else : print 'pass'
    #process_kstudy_rf(case)
    #if case == 1 :
        #generate_database(case)
    #    run(case)
    #db_test()
    #process_kstudy_rf(case)
    #run(case, True, num_keff, kdel)
    #ManagerERME.finalize()

    stop = timeit.default_timer()
    print stop - start
