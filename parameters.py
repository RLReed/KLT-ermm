'''
This file initializes the global variables used in the project
'''

def init(mode, task):
    global test_problem, nGroups, use_angular_moments, case, moment_order, config_number
    test_problem = '10-pin'
    nGroups = 44
    use_angular_moments = False
    case = 0
    moment_order = 0
    config_number = 0

    if mode in ['gendb', 'run', 'klt'] :
        if task < 84 :
            test_problem = '10-pin'
            if task < 42 :
                nGroups = 44
            else :
                nGroups = 238
                task -= 42
            if task < 14 :
                use_angular_moments = False
                case = task
            else :
                task -= 14
                use_angular_moments = True
                moment_order = task / 7
                case = task % 7 + 7
            print test_problem, nGroups, use_angular_moments, moment_order, case
                
        elif task < 186 : 
            test_problem = 'bwrcore'
            task -= 84
            config_number = task / 34
            task = task % 34
            if task < 17 :
                nGroups = 44
            else :
                nGroups = 238
                task -= 17
            if task < 5 :
                use_angular_moments = False
                case = task
            else :
                task -= 5
                use_angular_moments = True
                moment_order = task / 3
                case = task % 3 + 2
            print test_problem, config_number, nGroups, use_angular_moments, moment_order, case
        else:
            test_problem = 'c5g7'
            task -= 186
            nGroups = 44
            if task < 9:
                use_angular_moments = False
                case = task
            else:
                task -= 9
                use_angular_moments = True
                moment_order = task / 7
                case = task % 7 + 2
            print test_problem, nGroups, use_angular_moments, moment_order, case
    elif mode in ['ref','refrf'] :
        if task < 2 :
            test_problem = '10-pin'
            if task < 1 :
                nGroups = 44
            else :
                nGroups = 238
            print test_problem, nGroups
        else :
            task -= 2
            test_problem = 'bwrcore'
            config_number = task / 2
            if task % 2 == 0 :
                nGroups = 44
            elif task % 2 == 1 :
                nGroups = 238
            print test_problem, config_number, nGroups
            
    elif mode in ['plot']:
        if task < 4 :
            test_problem = '10-pin'
            if task < 2 :
                if task == 0: use_angular_moments = False
                else: use_angular_moments = True
                nGroups = 44
            else :
                task -= 2
                if task == 0: use_angular_moments = False
                else: use_angular_moments = True
                nGroups = 238
        elif task < 16 :
            task -= 4
            test_problem = 'bwrcore'
            config_number = task / 4
            if task % 4 == 0 :
                use_angular_moments = False
                nGroups = 44
            elif task % 4 == 1 :
                use_angular_moments = True
                nGroups = 44
            elif task % 4 == 2 :
                use_angular_moments = False
                nGroups = 238
            elif task % 4 == 3 :
                use_angular_moments = True
                nGroups = 238
            print test_problem, config_number, nGroups
        else:
            task -= 16
            test_problem = 'c5g7'
            nGroups = 44
            if task == 0:
                use_angular_moments = False
            elif task == 1:
                use_angular_moments = True

    else :
        test_problem = 'c5g7'
        config_number = 0
        nGroups = 44
        case = 7
        use_angular_moments = False
        moment_order = 0

