from parameters import test_problem, nGroups, use_angular_moments, case, moment_order, config_number
import main
from main import dir_name
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
from matplotlib import rc
from matplotlib.font_manager import FontProperties
import common
rc('font',**{'family':'serif'})
from matplotlib import rcParams
rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18
rcParams['lines.linewidth'] = 1.85
rcParams['axes.labelsize'] = 22
rcParams.update({'figure.autolayout': True})

cases = ['dlp1_L', 'dlp', 'core', 'assays', 'pins']

cases = ['dlp1_L', 'dlp', 'assays', 'pins', 'small_core', 'small_assays', '1D']
cases = ['dlp1_L', 'dlp', 'assays', 'pins', 'small_core', 'small_assays', '1D', 'red_full_core', 'red_small_core']
nDLPcases = 2
nKLTcases = int(len(cases) - nDLPcases)


def get_material() :
    """ Return the three material, 238 group library generated from SCALE 5.1.
        Materials 0 and 2 are UO2 and MOX while 1 is moderator.
    """
    io = IO_HDF5('cross_section_libraries/'+str(test_problem)+'_'+str(nGroups)+'_library.h5')
    mat = io.read_material()
    mat.compute_sigma_a()
    mat.compute_diff_coef()
    mat.finalize()
    
    return mat
    
def get_pins(number, flag) :
  """ Return the pins for the C5G238 benchmark.

  @param number   The number of meshes per dimension
  @param flag     The mesh type (false for uniform mesh with 
                  cell-center material, true for staircase)
  """

  # Shared
  pitch = Point(1.26, 1.26)
  radii = [0.54]
  # Pin 0 - UO2 
  pin0 = PinCell.Create(pitch, [0,5], radii, 0)
  pin0.meshify(number, flag)
  # Pin 1 - 4.3% MOX
  pin1 = PinCell.Create(pitch, [1,5], radii, 0)
  pin1.meshify(number, flag)
  # Pin 2 - 7.0% MOX
  pin2 = PinCell.Create(pitch, [2,5], radii, 0)
  pin2.meshify(number, flag)
  # Pin 3 - 8.7% MOX
  pin3 = PinCell.Create(pitch, [3,5], radii, 0)
  pin3.meshify(number, flag)
  # Pin 4 - Guide Tube or Fission Chamber
  pin4 = PinCell.Create(pitch, [4,5], radii, 0)
  pin4.meshify(number, flag)
  # Pin 5 - Moderator
  pin5 = PinCell.Create(pitch, [5,5], radii, 0)
  pin5.meshify(number, flag)

  return pin0, pin1, pin2, pin3, pin4, pin5

def get_assemblies(number, flag) :
    """ Return the assemblies for the C5G238 benchmark.

    See get_pincells for parameter definition.
    """

    # Shared things
    G = 4 # guide tube
    pin0, pin1, pin2, pin3, pin4, pin5 = get_pins(number, flag)

    # Assembly 1 -- UO2
    assem1 = Assembly.Create(17)
    assem1.add_pincell(pin0)
    assem1.add_pincell(pin1)
    assem1.add_pincell(pin2)
    assem1.add_pincell(pin3)
    assem1.add_pincell(pin4)
    assem1.add_pincell(pin5)
    pin_map1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                0,0,0,0,0,G,0,0,G,0,0,G,0,0,0,0,0,
                0,0,0,G,0,0,0,0,0,0,0,0,0,G,0,0,0,
                0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                0,0,G,0,0,G,0,0,G,0,0,G,0,0,G,0,0,
                0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                0,0,G,0,0,G,0,0,G,0,0,G,0,0,G,0,0,
                0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                0,0,G,0,0,G,0,0,G,0,0,G,0,0,G,0,0,
                0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                0,0,0,G,0,0,0,0,0,0,0,0,0,G,0,0,0,
                0,0,0,0,0,G,0,0,G,0,0,G,0,0,0,0,0,
                0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    #pin_map1 = [0,0,0,  0,0,0,  0,0,0]
    assem1.finalize(pin_map1)

    # Assembly 2 - MOX
    assem2 = Assembly.Create(17)
    assem2.add_pincell(pin0)
    assem2.add_pincell(pin1)
    assem2.add_pincell(pin2)
    assem2.add_pincell(pin3)
    assem2.add_pincell(pin4)
    assem2.add_pincell(pin5)
    pin_map2 = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,
                1,2,2,2,2,G,2,2,G,2,2,G,2,2,2,2,1,
                1,2,2,G,2,3,3,3,3,3,3,3,2,G,2,2,1,
                1,2,2,2,3,3,3,3,3,3,3,3,3,2,2,2,1,
                1,2,G,3,3,G,3,3,G,3,3,G,3,3,G,2,1,
                1,2,2,3,3,3,3,3,3,3,3,3,3,3,2,2,1,
                1,2,2,3,3,3,3,3,3,3,3,3,3,3,2,2,1,
                1,2,G,3,3,G,3,3,G,3,3,G,3,3,G,2,1,
                1,2,2,3,3,3,3,3,3,3,3,3,3,3,2,2,1,
                1,2,2,3,3,3,3,3,3,3,3,3,3,3,2,2,1,
                1,2,G,3,3,G,3,3,G,3,3,G,3,3,G,2,1,
                1,2,2,2,3,3,3,3,3,3,3,3,3,2,2,2,1,
                1,2,2,G,2,3,3,3,3,3,3,3,2,G,2,2,1,
                1,2,2,2,2,G,2,2,G,2,2,G,2,2,2,2,1,
                1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,
                1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1];
    #pin_map2 = [1,1,1, 1,1,1, 1,1,1]
    assem2.finalize(pin_map2)

    # Assembly 3 - Moderator
    assem3 = Assembly.Create(17)
    assem3.add_pincell(pin0)
    assem3.add_pincell(pin1)
    assem3.add_pincell(pin2)
    assem3.add_pincell(pin3)
    assem3.add_pincell(pin4)
    assem3.add_pincell(pin5)
    pin_map3 = [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
                5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
                5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
                5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
                5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
                5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
                5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
                5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
                5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
                5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
                5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
                5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
                5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
                5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
                5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
                5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
                5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5];
    #pin_map3 = [5,5,5, 5,5,5,  5,5,5]
    assem3.finalize(pin_map3)

    return assem1, assem2, assem3
    
def get_smallassemblies(number, flag) :
    """ Return the small assemblies for the C5G238 benchmark.

    See get_pincells for parameter definition.
    """

    # Shared things
    G = 4 # guide tube
    pin0, pin1, pin2, pin3, pin4, pin5 = get_pins(number, flag)

    # Assembly 1 -- UO2
    assem1 = Assembly.Create(8)
    assem1.add_pincell(pin0)
    assem1.add_pincell(pin1)
    assem1.add_pincell(pin2)
    assem1.add_pincell(pin3)
    assem1.add_pincell(pin4)
    assem1.add_pincell(pin5)
    pin_map1 =  [0,0,0,0,0,0,0,0,
                 0,0,0,0,0,0,0,0,
                 0,0,G,0,0,G,0,0,
                 0,0,0,0,0,0,0,0,
                 0,0,0,0,0,0,0,0,
                 0,0,G,0,0,G,0,0,
                 0,0,0,0,0,0,0,0,
                 0,0,0,0,0,0,0,0];
    assem1.finalize(pin_map1)

    # Assembly 2 - MOX
    assem2 = Assembly.Create(8)
    assem2.add_pincell(pin0)
    assem2.add_pincell(pin1)
    assem2.add_pincell(pin2)
    assem2.add_pincell(pin3)
    assem2.add_pincell(pin4)
    assem2.add_pincell(pin5)
    pin_map2 =  [1,1,2,2,2,2,1,1,
                 1,2,2,2,2,2,2,1,
                 2,2,G,3,3,G,2,2,
                 2,2,3,3,3,3,2,2,
                 2,2,3,3,3,3,2,2,
                 2,2,G,3,3,G,2,2,
                 1,2,2,2,2,2,2,1,
                 1,1,2,2,2,2,1,1];
    assem2.finalize(pin_map2)

    # Assembly 3 - Moderator
    assem3 = Assembly.Create(8)
    assem3.add_pincell(pin0)
    assem3.add_pincell(pin1)
    assem3.add_pincell(pin2)
    assem3.add_pincell(pin3)
    assem3.add_pincell(pin4)
    assem3.add_pincell(pin5)
    pin_map3 = [5,5,5,5,5,5,5,5,
                5,5,5,5,5,5,5,5,
                5,5,5,5,5,5,5,5,
                5,5,5,5,5,5,5,5,
                5,5,5,5,5,5,5,5,
                5,5,5,5,5,5,5,5,
                5,5,5,5,5,5,5,5,
                5,5,5,5,5,5,5,5];
    assem3.finalize(pin_map3)

    return assem1, assem2, assem3
    
def get_junctions(number, flag) :
    """ Return the pin junctions for the C5G238 benchmark.

    See get_pincells for parameter definition.
    """

    # Shared things
    G = 4 # guide tube
    pin0, pin1, pin2, pin3, pin4, pin5 = get_pins(number, flag)

    # Junction 1 -- UO2 / MOX 4.3
    assem1 = Assembly.Create(2)
    assem1.add_pincell(pin0)
    assem1.add_pincell(pin1)
    assem1.add_pincell(pin2)
    assem1.add_pincell(pin3)
    assem1.add_pincell(pin4)
    assem1.add_pincell(pin5)
    pin_map1 =  [0,1,
                 1,0];
    assem1.finalize(pin_map1)

    # Junction 2 - UO2 / MOX 7
    assem2 = Assembly.Create(2)
    assem2.add_pincell(pin0)
    assem2.add_pincell(pin1)
    assem2.add_pincell(pin2)
    assem2.add_pincell(pin3)
    assem2.add_pincell(pin4)
    assem2.add_pincell(pin5)
    pin_map2 =  [0,2,
                 2,0];
    assem2.finalize(pin_map2)

    # Junction 3 - UO2 / MOX 8.3
    assem3 = Assembly.Create(2)
    assem3.add_pincell(pin0)
    assem3.add_pincell(pin1)
    assem3.add_pincell(pin2)
    assem3.add_pincell(pin3)
    assem3.add_pincell(pin4)
    assem3.add_pincell(pin5)
    pin_map3 = [0,3,
                3,0];
    assem3.finalize(pin_map3)
    
    # Junction 4 - MOX 4.3 / MOX 7
    assem4 = Assembly.Create(2)
    assem4.add_pincell(pin0)
    assem4.add_pincell(pin1)
    assem4.add_pincell(pin2)
    assem4.add_pincell(pin3)
    assem4.add_pincell(pin4)
    assem4.add_pincell(pin5)
    pin_map4 = [1,2,
                2,1];
    assem4.finalize(pin_map4)
    
    # Junction 5 - MOX 4.3 / MOX 8.3
    assem5 = Assembly.Create(2)
    assem5.add_pincell(pin0)
    assem5.add_pincell(pin1)
    assem5.add_pincell(pin2)
    assem5.add_pincell(pin3)
    assem5.add_pincell(pin4)
    assem5.add_pincell(pin5)
    pin_map5 = [1,3,
                3,1];
    assem5.finalize(pin_map5)
    
    # Junction 6 - MOX 7 / MOX 8.3
    assem6 = Assembly.Create(2)
    assem6.add_pincell(pin0)
    assem6.add_pincell(pin1)
    assem6.add_pincell(pin2)
    assem6.add_pincell(pin3)
    assem6.add_pincell(pin4)
    assem6.add_pincell(pin5)
    pin_map6 = [2,3,
                3,2];
    assem6.finalize(pin_map6)

    return assem1, assem2, assem3, assem4, assem5, assem6

def get_core(number, flag, small=False) :
    """ Return the core for the C5G7 benchmark.

    See get_pincells for parameter definition.
    """
    if small == True :
        assemblies = get_smallassemblies(number, flag)
    else :
        assemblies = get_assemblies(number, flag)
    core       = Core.Create(3)
    core.add_assembly(assemblies[0])
    core.add_assembly(assemblies[1])
    core.add_assembly(assemblies[2])
    core_map = [0,1,2,
                1,0,2,
                2,2,2]
    core.finalize(core_map)
    return core
    
def get_db(case, new_klt=False) :
	
    # Outer solver database
    outer_db = InputDB.Create("outer_solver_db")
    outer_db.put_dbl("linear_solver_atol",              1.0e-12);
    outer_db.put_dbl("linear_solver_rtol",              0.0);
    outer_db.put_str("linear_solver_type",              "petsc");
    outer_db.put_int("linear_solver_maxit",             1000);
    outer_db.put_int("linear_solver_gmres_restart",     30);
    outer_db.put_int("linear_solver_monitor_level",     0);
    
    # Outer preconditioner database
    pc_db = InputDB.Create("outer_solver_db")
    pc_db.put_dbl("linear_solver_atol",                 1.0e-12);
    pc_db.put_dbl("linear_solver_rtol",                 0.0);
    pc_db.put_str("linear_solver_type",                 "petsc");
    pc_db.put_int("linear_solver_maxit",                1000);
    pc_db.put_int("linear_solver_gmres_restart",        30);
    pc_db.put_int("linear_solver_monitor_level",        0);
    pc_db.put_str("pc_type",                            "petsc"); 
    pc_db.put_str("petsc_pc_type",                      "lu");
	
    basis_inp = InputDB.Create("basis_data")
    """ Provides the base nodal db (or for use in the reference)
    """
    db = InputDB.Create()
    db.put_int("number_groups",                  nGroups)
    db.put_int("dimension",                      2)
    db.put_str("equation",                       "dd")
    db.put_str("quad_type",                      "asqr-gc")
    db.put_int("quad_number_polar_octant",       8)
    db.put_int("quad_number_azimuth_octant",     8)
    # SOLVER
    db.put_str("outer_solver",                   "GMRES")
    db.put_int("inner_max_iters",                100)
    db.put_dbl("inner_tolerance",                1.0e-8)
    db.put_int("inner_print_level",              0)
    db.put_dbl("eigen_tolerance",                1.0e-8)
    db.put_int("eigen_max_iters",                100000)
    db.put_int("outer_print_level",              0)
    db.put_dbl("outer_tolerance",                1.0e-8)
    db.put_int("outer_max_iters",                1000000)    
    db.put_int("outer_krylov_group_cutoff",      0)
    db.put_str("outer_pc_type",                  "mgcmdsa")
    db.put_int('mgpc_coarse_mesh_level',         7)
    db.put_int('mgpc_condensation_option',       1)
    db.put_int('mgpc_cmdsa_use_smoothing',       1)
    db.put_int('mgpc_cmdsa_smoothing_iters',     3)
    db.put_dbl('mgpc_cmdsa_smoothing_relax',     0.7)
    db.put_spdb('outer_solver_db',               outer_db)
    db.put_spdb("outer_pc_db",                   pc_db)
    db.put_str("basis_s_type",                   "dlp")
    db.put_str("basis_p_type",                   "cheby")
    db.put_str("basis_a_type",                   "clp")
    db.put_str("bc_west",                        "fixed")
    db.put_str("bc_east",                        "fixed")
    db.put_str("bc_south",                       "fixed")
    db.put_str("bc_north",                       "fixed")      
    db.put_int("erme_expand_angular_flux",       1)
    if case >= nDLPcases :
        if new_klt is True :
            A = common.KLT(case)
        else:
            if use_angular_moments == False :
                A = np.loadtxt(dir_name+'/basis_functions/phi/functions_case'+str(case)+'.txt')
            elif moment_order == 0 :
                A = np.loadtxt(dir_name+'/basis_functions/partial/functions_case'+str(case)+'.txt')
            else :
                A = np.loadtxt(dir_name+'/basis_functions/angular/functions_case'+str(case)+'order'+str(moment_order)+'.txt')
        A = np.real(A)
        for o in range(nGroups) :
	        basis_inp.put_vec_dbl("vec"+str(o), A[o])
        db.put_spdb("basis_e_db", basis_inp)
    return db
    
def get_snapshots(case) :
    c = case - nDLPcases
    if Comm.rank() == 0 :
        print 'adding snapshots of phi'
    if c == 0 :
        Data = np.loadtxt(dir_name+'/refdata/assay0_flux')
        Data1 = np.loadtxt(dir_name+'/refdata/assay1_flux')
        Data = np.concatenate((Data, Data1), axis=1)
    elif c == 1 :
        Data = np.loadtxt(dir_name+'/refdata/fuel0_flux')
        Data1 = np.loadtxt(dir_name+'/refdata/fuel1_flux')
        Data = np.concatenate((Data, Data1), axis=1)
        Data1 = np.loadtxt(dir_name+'/refdata/fuel2_flux')
        Data = np.concatenate((Data, Data1), axis=1)
        Data1 = np.loadtxt(dir_name+'/refdata/fuel3_flux')
        Data = np.concatenate((Data, Data1), axis=1)
        Data1 = np.loadtxt(dir_name+'/refdata/junction0_flux')
        Data = np.concatenate((Data, Data1), axis=1)
        Data1 = np.loadtxt(dir_name+'/refdata/junction1_flux')
        Data = np.concatenate((Data, Data1), axis=1)
        Data1 = np.loadtxt(dir_name+'/refdata/junction2_flux')
        Data = np.concatenate((Data, Data1), axis=1)
        Data1 = np.loadtxt(dir_name+'/refdata/junction3_flux')
        Data = np.concatenate((Data, Data1), axis=1)
        Data1 = np.loadtxt(dir_name+'/refdata/junction4_flux')
        Data = np.concatenate((Data, Data1), axis=1)
        Data1 = np.loadtxt(dir_name+'/refdata/junction5_flux')
        Data = np.concatenate((Data, Data1), axis=1)
    elif c == 2 :
        Data = np.loadtxt(dir_name+'/refdata/smallcore_flux')
    elif c == 3 :
        Data = np.loadtxt(dir_name+'/refdata/smallassay0_flux')
        Data1 = np.loadtxt(dir_name+'/refdata/smallassay1_flux')
        Data = np.concatenate((Data, Data1), axis=1)
    elif c == 4 :
        Data = np.loadtxt(dir_name+'/refdata/1D_flux')
    elif c == 5 :
        Data = np.loadtxt(dir_name+'/refdata/core_flux')
        red_data = np.zeros((Data.shape[0], Data.shape[1] / 49))
        for i in range(len(red_data)):
            red_data[i] = np.mean(np.mean(Data[i].reshape(-1,7),axis=1).reshape(np.sqrt(Data.shape[1]),-1).T.reshape(-1,7),axis=1)
        Data = red_data
    elif c == 6 :
        Data = np.loadtxt(dir_name+'/refdata/smallcore_flux')
        red_data = np.zeros((Data.shape[0], Data.shape[1] / 49))
        for i in range(len(red_data)):
            red_data[i] = np.mean(np.mean(Data[i].reshape(-1,7),axis=1).reshape(np.sqrt(Data.shape[1]),-1).T.reshape(-1,7),axis=1)
        Data = red_data
        
    if Comm.rank() == 0 :
        print 'size = ', Data.shape
        
    if use_angular_moments == False :
        if Comm.rank() == 0 :
            print 'removing duplicate snapshots'
        Data = Data.T
        b = np.ascontiguousarray(np.around(Data, decimals=6)).view(np.dtype((np.void, Data.dtype.itemsize * Data.shape[1])))
        Data = np.unique(b).view(Data.dtype).reshape(-1, Data.shape[1]).T
        if Comm.rank() == 0 :
            print 'reduced size = ', Data.shape
        return Data
        
    for o in range(moment_order+1):
        if Comm.rank() == 0 :
            if o == 0 :
                print 'adding snapshots of leftward partial current'
            else :
                print 'adding snapshots of moment ', o
        if c == 0 :
            angdata = np.loadtxt(dir_name+'/refdata/assay0_moment'+str(o))
            angdata1 = np.loadtxt(dir_name+'/refdata/assay1_moment'+str(o))
            angdata = np.concatenate((angdata, angdata1), axis=1)
        elif c == 1 :
            angdata = np.loadtxt(dir_name+'/refdata/fuel0_moment'+str(o))
            angdata1 = np.loadtxt(dir_name+'/refdata/fuel1_moment'+str(o))
            angdata = np.concatenate((angdata, angdata1), axis=1)
            angdata1 = np.loadtxt(dir_name+'/refdata/fuel2_moment'+str(o))
            angdata = np.concatenate((angdata, angdata1), axis=1)
            angdata1 = np.loadtxt(dir_name+'/refdata/fuel3_moment'+str(o))
            angdata = np.concatenate((angdata, angdata1), axis=1)
            angdata1 = np.loadtxt(dir_name+'/refdata/junction0_moment'+str(o))
            angdata = np.concatenate((angdata, angdata1), axis=1)
            angdata1 = np.loadtxt(dir_name+'/refdata/junction1_moment'+str(o))
            angdata = np.concatenate((angdata, angdata1), axis=1)
            angdata1 = np.loadtxt(dir_name+'/refdata/junction2_moment'+str(o))
            angdata = np.concatenate((angdata, angdata1), axis=1)
            angdata1 = np.loadtxt(dir_name+'/refdata/junction3_moment'+str(o))
            angdata = np.concatenate((angdata, angdata1), axis=1)
            angdata1 = np.loadtxt(dir_name+'/refdata/junction4_moment'+str(o))
            angdata = np.concatenate((angdata, angdata1), axis=1)
            angdata1 = np.loadtxt(dir_name+'/refdata/junction5_moment'+str(o))
            angdata = np.concatenate((angdata, angdata1), axis=1)
        elif c == 2 :
            angdata = np.loadtxt(dir_name+'/refdata/smallcore_moment'+str(o))
        elif c == 3 :
            angdata = np.loadtxt(dir_name+'/refdata/smallassay0_moment'+str(o))
            angdata1 = np.loadtxt(dir_name+'/refdata/smallassay1_moment'+str(o))
            angdata = np.concatenate((angdata, angdata1), axis=1)
        elif c == 4 :
            angdata = np.loadtxt(dir_name+'/refdata/1D_moment'+str(o))
        elif c == 5 :
            angdata = np.loadtxt(dir_name+'/refdata/core_moment'+str(o))
            red_data = np.zeros((angdata.shape[0], angdata.shape[1] / 49))
            for i in range(len(red_data)):
                red_data[i] = np.mean(np.mean(angdata[i].reshape(-1,7),axis=1).reshape(np.sqrt(angdata.shape[1]),-1).T.reshape(-1,7),axis=1)
            angdata = red_data
        elif c == 6 :
            angdata = np.loadtxt(dir_name+'/refdata/smallcore_moment'+str(o))
            red_data = np.zeros((angdata.shape[0], angdata.shape[1] / 49))
            for i in range(len(red_data)):
                red_data[i] = np.mean(np.mean(angdata[i].reshape(-1,7),axis=1).reshape(np.sqrt(angdata.shape[1]),-1).T.reshape(-1,7),axis=1)
            angdata = red_data
            
            
        Data = np.concatenate((Data, angdata), axis=1)
        if Comm.rank() == 0 :
            print 'size = ', Data.shape
    if Comm.rank() == 0 :
        print 'removing duplicate snapshots'
    Data = Data.T
    b = np.ascontiguousarray(np.around(Data, decimals=6)).view(np.dtype((np.void, Data.dtype.itemsize * Data.shape[1])))
    Data = np.unique(b).view(Data.dtype).reshape(-1, Data.shape[1]).T
    if Comm.rank() == 0 :
        print 'reduced size = ', Data.shape
    return Data
        
def get_nodes(inp, energy_order, rfdb, dbgen) :
    mat = get_material()
    # meshes
    assems = get_assemblies(7, True)
    mesh = []
    for m in range(len(assems)) :
        mesh.append(assems[m].mesh())
        nm = mesh[m].number_cells()
        mesh[m].add_mesh_map("NODAL", vec_int(nm, 0))
    # orders
    so = 2
    ao = 2
    po = 2
    so = [[so],[so],[so],[so]]
    po = [po,po,po,po]
    ao = [ao,ao,ao,ao]
    eo = [energy_order, energy_order, energy_order, energy_order]
    # nodes
    w  = [mesh[0].total_width_x(), mesh[0].total_width_y(), 1.0]
    nodes = []
    if rfdb == True :
        if Comm.rank() == 0 :
            print 'Using Cartesian Nodes'
        for i in range(len(mesh)) :
            tmp = CartesianNode.Create(2, 'node'+str(i), so, po, ao, eo, w)
            tmp.set_number_pins(17*17)
            nodes.append(tmp)
    else :
        if Comm.rank() == 0 :
            print 'Using Detran Nodes'
        for i in range(len(mesh)) :
            nodes.append(CartesianNodeDetran.Create(2, 'node'+str(i), so, po, ao, eo, w, inp, mat, mesh[i]))
    if dbgen == True:
        nodal_map = np.array([[0,1,2]],'i')
    else :
        nodal_map = np.array([[0,1,2], 
                              [1,0,2], 
                              [2,2,2]],'i')
    # global boundary conditions
    V = Node.VACUUM
    R = Node.REFLECT
    bc = [R,V,R,V]
    nodes[2].set_fuel(False)
    nodes = build_model(nodes, nodal_map, bc)
    return nodes
    
def run_1D_ref() :
    """ Run the reference eigenvalue problem using Detran.  Also, produce
        the reference modes for producing an orthogonal basis.
    """
    db = InputDB.Create()
    db.put_int("number_groups",                  nGroups)
    db.put_int("dimension",                      1)
    db.put_str("equation",                       "sc")
    db.put_str("quad_type",                      "dgl")
    db.put_int("quad_number_polar_octant",       8)
    db.put_str("inner_solver",                   "SI")
    db.put_int("inner_print_level",              0)
    db.put_int("inner_max_iters",                10)
    db.put_dbl("inner_tolerance",                1.0e-12)
    db.put_dbl("eigen_tolerance",                1.0e-12)
    db.put_int("eigen_max_iters",                1000)
    db.put_str("outer_solver",                   "GMRES")
    db.put_int("outer_max_iters",                1000)
    db.put_int("outer_print_level",              0)
    db.put_dbl("outer_tolerance",                1.0e-12)
    db.put_int("outer_krylov_group_cutoff",      0)
    db.put_str("basis_p_type",                   "jacobi")
    db.put_str("bc_west",      "reflect")
    db.put_str("bc_east",      "vacuum")
    db.put_int("store_angular_flux",             1)
    
    mat  = get_material()

    #--------------------------------------------------------------------------#
    # MODELS
    #--------------------------------------------------------------------------#
    
    cm_pin = [0.09, 1.17, 1.26]
    fm_pin = [3, 22, 3]
    n = 51
    cm = [0.0]
    fm = []
    cm_map = []
    for i in range(0, n) :
        for j in range(0, len(cm_pin)) :
            cm.append(cm_pin[j] + 1.26*float(i))
            fm.append(fm_pin[j])
        cm_map += [i,i,i]
    pin_map = [0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,0,
               1,1,2,2,3,3,3,3,4,3,3,3,3,2,2,1,1,
               5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5]
    mts = []
    mts.append([5, 0, 5])
    mts.append([5, 1, 5])
    mts.append([5, 2, 5])
    mts.append([5, 3, 5])
    mts.append([5, 3, 5])
    mts.append([5, 5, 5])
    
    mt = []
    for i in range(0, len(pin_map)) :
        mt += mts[pin_map[i]]
    mesh = Mesh1D.Create(fm, cm, mt)
    mesh.add_coarse_mesh_map("PINS", cm_map)           
    
    solver = Eigen1D(db, mat, mesh)
    solver.solve()
    
    ng = solver.material().number_groups()
    nc = solver.mesh().number_cells()
    no = solver.quadrature().number_octants()/2
    na = solver.quadrature().number_angles_octant()
    phig = np.zeros((ng,nc))
    for g in range(0, ng) :
        for i in range(0, nc) :
            phig[g][i] = solver.state().phi(g)[i]
    np.savetxt(dir_name+'/refdata/1D_flux', phig)

    nOrder = 4
    angmom = np.zeros((nOrder,ng,nc))
    P = OrthogonalBasisParameters()
    P.order = nOrder
    P.size = na
    P.x.resize(P.size, 0.0)
    P.qw.resize(P.size, 0.0)
    P.lower_bound = 0.0
    P.upper_bound = 1.0
    P.orthonormal = False
    for g in range(ng) :
        for i in range(nc) :
            for o in range(no) :
                psi = np.zeros((na))
                for a in range(na) :
                    psi[a] = solver.state().psi(g,o,a)[i]
                for a in range(na) :
                    P.x[a] = solver.quadrature().mu(o,a)
                    P.qw[a] = solver.quadrature().weight(a)
                ang = OrthogonalBasis.Create("jacobi", P)
                angular = np.zeros(nOrder+1)
                y = vec_dbl(psi.tolist())
                z = vec_dbl(angular.tolist())
                ang.transform(y, z)
                for order in range(nOrder) :  
                    angmom[order][g][i] += z[order]
    for order in range(nOrder) :
        np.savetxt(dir_name+'/refdata/1D_moment'+str(order), angmom[order])
    

def run_reference(number) :
    """ Run the reference eigenvalue problem using Detran.  Also, produce
        the reference modes for producing an orthogonal basis.
    """
    db   = get_db(0)
    db.put_int('adjoint',      0)
    db.put_str("bc_west",      "reflect")
    db.put_str("bc_east",      "vacuum")
    db.put_str("bc_south",     "reflect")
    db.put_str("bc_north",     "vacuum") 
    db.put_int("store_angular_flux",             1)
    
    mat  = get_material()

    #--------------------------------------------------------------------------#
    # MODELS
    #--------------------------------------------------------------------------#
    phi = []
    numcells = []
    angularmoment = []
    if nGroups == 238 :
        thermal_cutoff_group = 203
    elif nGroups == 44 :
        thermal_cutoff_group = 27
    #for n in range(16) :
    for n in range(number, number + 1):
        x = []
        if n > 1 :
            db.put_str("bc_east",      "reflect")
            db.put_str("bc_north",     "reflect")
        if n == 0 :
            core = get_core(7, True)
            mesh = core.mesh()
        elif n == 1 :
            core = get_core(7, True, True)
            mesh = core.mesh()
        elif n < 4 :
            assems = get_assemblies(7, True)
            mesh = assems[n-2].mesh()
        elif n < 6 :
            assems = get_smallassemblies(7, True)
            mesh = assems[n-4].mesh()
        elif n < 12 :
            assems = get_junctions(7, True)
            mesh = assems[n-6].mesh()
        else :
            pins = get_pins(7, True)
            mesh = pins[n-12].mesh()
        solver = Eigen2D(db, mat, mesh)
        solver.solve()
        
        if n == 0 :
            keff = solver.state().eigenvalue()
            rates = ReactionRates(mat, mesh, solver.state())
            pin_power = rates.region_power("PINS")
        ng = solver.material().number_groups()
        nc = solver.mesh().number_cells()
        no = solver.quadrature().number_octants()/2
        na = solver.quadrature().number_angles_octant()
        phig = np.zeros((ng,nc))
        x.append(mesh.dx(0) *0.5)
        for i in range(1,nc) :
            x.append(x[i-1]+mesh.dx(i-1))
        numcells.append(x)
        for g in range(0, ng) :
            for i in range(0, nc) :
                phig[g][i] = solver.state().phi(g)[i]
        if n == 0 :
            np.savetxt(dir_name+'/refdata/core_flux', phig)
        elif n == 1 :
            np.savetxt(dir_name+'/refdata/smallcore_flux', phig)
        elif n < 4 :
            np.savetxt(dir_name+'/refdata/assay'+str(n-2)+'_flux', phig)
        elif n < 6 :
            np.savetxt(dir_name+'/refdata/smallassay'+str(n-4)+'_flux', phig)
        elif n < 12 :
            np.savetxt(dir_name+'/refdata/junction'+str(n-6)+'_flux', phig)
        else :
            np.savetxt(dir_name+'/refdata/fuel'+str(n-12)+'_flux', phig)
        phi.append(phig)

        nOrder = 4
        angmom = np.zeros((nOrder,ng,nc))
        P = OrthogonalBasisParameters()
        P.order = nOrder
        P.size = na
        P.x.resize(P.size, 0.0)
        P.qw.resize(P.size, 0.0)
        P.lower_bound = 0.0
        P.upper_bound = 1.0
        P.orthonormal = False
        x = np.linspace(P.lower_bound, P.upper_bound, P.size)
        for g in range(ng) :
            for i in range(nc) :
                for o in range(no) :
                    psi = np.zeros((na))
                    for a in range(na) :
                        psi[a] = solver.state().psi(g,o,a)[i]
                    for a in range(na) :
                        P.x[a] = solver.quadrature().mu(o,a)
                        P.qw[a] = solver.quadrature().weight(a)
                    ang = OrthogonalBasis.Create("jacobi", P)
                    angular = np.zeros(nOrder+1)
                    y = vec_dbl(psi.tolist())
                    z = vec_dbl(angular.tolist())
                    ang.transform(y, z)
                    for order in range(nOrder) :  
                        angmom[order][g][i] += z[order]
        for order in range(nOrder) :
            if n == 0 :
                np.savetxt(dir_name+'/refdata/core_moment'+str(order), angmom[order])
            elif n == 1 :
                np.savetxt(dir_name+'/refdata/smallcore_moment'+str(order), angmom[order])
            elif n < 4 :
                np.savetxt(dir_name+'/refdata/assay'+str(n-2)+'_moment'+str(order), angmom[order])
            elif n < 6 :
                np.savetxt(dir_name+'/refdata/smallassay'+str(n-4)+'_moment'+str(order), angmom[order])
            elif n < 12 :
                np.savetxt(dir_name+'/refdata/junction'+str(n-6)+'_moment'+str(order), angmom[order])
            else :
                np.savetxt(dir_name+'/refdata/fuel'+str(n-12)+'_moment'+str(order), angmom[order])
        angularmoment.append(angmom)
    
    #--------------------------------------------------------------------------#
    # PLOTS AND DATA
    #--------------------------------------------------------------------------#
     
    data = {}
    data['pin_power'] = pin_power
    data['phi_full'] = np.mean(phi[0],axis=1)
    data['keff'] = keff
    pickle.dump(data, open(dir_name+'/refdata/reference_data.p', 'wb'))

    io = IO_HDF5('cross_section_libraries/'+str(test_problem)+'_'+str(nGroups)+'_library.h5')
    db = io.read_input()
    EG = db.get_vec_dbl('neutron_energy_bounds')
    energyrange = np.zeros((len(EG)-1))
    deltaE = np.zeros((len(EG)-1))
    for i in range(len(energyrange)):
        energyrange[i] = (EG[i]+EG[i+1])/2
        deltaE[i] = abs(EG[i+1] - EG[i])
         
    Eave = []
    Save = []
    for order in range(nOrder) :
        E = np.mean(angularmoment[0][order], axis=0)
        S = np.mean(angularmoment[0][order], axis=1)
        Eave.append(E)
        Save.append(S)

    A = []
    # Data for Phi plots 
    for a in range(len(phi)): 
        y, A0 = common.barchart(EG, np.mean(phi[a], axis=1))
        A.append(A0)
    C = []
    for order in range(nOrder) :
        y, B00 = common.barchart(EG, Save[order])
        C.append(B00)

    groups = range(0, nGroups)
    fontP = FontProperties()
    fontP.set_size('small')
    plt.figure(0, figsize=(8, 6.5))
    plt.semilogy(groups, np.mean(phi[0], axis=1),      'k-', \
                 groups, np.mean(phi[1], axis=1),      'k--', \
                 groups, np.mean(phi[2], axis=1),      'b-', \
                 groups, np.mean(phi[3], axis=1),      'g-', \
                 groups, np.mean(phi[4], axis=1),      'b--', \
                 groups, np.mean(phi[5], axis=1),      'g--')
    plt.xlabel('$g$')
    plt.ylabel('$\phi_g$')
    leg = ['core', 'small_core', 'assay - UO$_2$', 'assay - MOX', 'smallassay - UO$_2$',
            'smallassay - MOX']
    plt.legend(leg, loc = 0, prop=fontP)
    plt.grid(True)
    plt.savefig(dir_name+'/reference_figures/'+str(nGroups)+'group_spectra.pdf')

    # Spectra Plot
    plt.figure(1, figsize=(8, 6.5))
    plt.loglog(y, A[0],     'k-', \
               y, A[1],     'k--', \
               y, A[2],     'b-', \
               y, A[3],     'g-', \
               y, A[4],     'b--', \
               y, A[5],     'g--')
    plt.xlabel('energy in $eV$')
    plt.ylabel('$\phi_g$')
    plt.legend(leg, loc = 0, prop=fontP)
    plt.grid(True)
    plt.savefig(dir_name+'/reference_figures/'+str(nGroups)+'group_spectra_energy.pdf')   
    
    # Spatially Averaged Moment Plots
    # Moment 0 Plot
    plt.figure(2, figsize=(8, 6.5))
    plt.loglog(y, C[0],     'k-', \
               y, C[1],     'r-', \
               y, C[2],     'g-', \
               y, C[3],     'b-'  )
    plt.xlabel('energy in $eV$')
    plt.ylabel('${moment}_g$')
    leg = ['$0^{th}$ moment', '$1^{st}$ moment', '$2^{nd}$ moment', '$3^{rd}$ moment']
    plt.legend(leg, loc = 0, prop=fontP)
    plt.grid(True)
    plt.savefig(dir_name+'/reference_figures/'+str(nGroups)+'monent_spectra_energy.pdf')
    
    # Energy Averaged Plots
    x = numcells[0] 
    X = np.reshape(x, (int(np.sqrt(len(x))),-1), order='F')
    phifast = np.mean(phi[0][:thermal_cutoff_group][0:len(x)],axis=0)
    
    plt.figure(3, figsize=(8, 6.5))
    im = plt.pcolormesh(X,X,np.reshape(phifast, (int(np.sqrt(len(phifast))), -1), order='F'), cmap='afmhot', shading='gouraud')
    plt.imshow(np.reshape(phifast, (int(np.sqrt(len(phifast))), -1), order='F'), cmap='afmhot')
    plt.colorbar(im, set_label='Energy Averaged Flux')
    plt.xticks(X[0])
    plt.yticks(X[0])
    plt.xlabel('position in cm', labelsize=20)
    plt.ylabel('position in cm', labelsize=20)
    plt.grid(True)
    plt.savefig(dir_name+'/reference_figures/fastspectra_core_energyaveraged.pdf')
    
    phifast = np.mean(phi[0][thermal_cutoff_group:][0:len(x)],axis=0)
    
    plt.figure(4, figsize=(8, 6.5))
    plt.imshow(np.reshape(phifast, (int(np.sqrt(len(phifast))), -1), order='F'), cmap='afmhot')
    plt.colorbar(im, set_label='Energy Averaged Flux')
    plt.xticks(X[0])
    plt.yticks(X[0])
    plt.xlabel('position in cm', labelsize=20)
    plt.ylabel('position in cm', labelsize=20)
    plt.grid(True)
    plt.savefig(dir_name+'/reference_figures/thermalspectra_core_energyaveraged.pdf')
    
    # Moment 0 Plot
    
    leg = ['$0^{th}$ moment', '$1^{st}$ moment', '$2^{nd}$ moment']
    plt.figure(6, figsize=(8, 6.5))
    plt.plot(x, Eave[0],     'k-', \
             x, Eave[1],     'r-', \
             x, Eave[2],     'g-'  )
    plt.xlabel('position in cm')
    plt.ylabel('${moment0}_g$')
    plt.legend(leg, loc = 0, prop=fontP)
    plt.grid(True)
    plt.savefig(dir_name+'/reference_figures/'+str(nGroups)+'monent_spectra_full.pdf')
    
    plt.figure(0, figsize=(8, 6.5))
    phifast = np.mean(phi[0],axis=0)
    im = plt.pcolormesh(X,X,np.reshape(phifast, (int(np.sqrt(len(phifast))), -1), order='F'), cmap='afmhot', shading='gouraud')
    plt.axis([0,mesh.total_width_x(),0,mesh.total_width_y()])
    cbar = plt.colorbar(im,format='%.1e')
    cbar.set_label('Scalar Flux', size=22)
    plt.xlabel('x position in cm')
    plt.ylabel('y position in cm')
    plt.savefig('fastspectra_core_energyaveraged.pdf')
    
def process_rf_data(tog44=False) :
    data = {}
    
    fontP = FontProperties()
    fontP.set_size('medium')
    
    # Load reference rf data
    rfdata =  pickle.load(open(dir_name+'/refdata/reference_rf_data.p', "rb"))
    ref_power = rfdata['ass_p']
    pmask = ref_power>0
    ref_pin_power = rfdata['pin_p']
    ppmask = ref_pin_power>0

    if use_angular_moments == False :
        for c in range(len(cases)) :
            if c < nDLPcases :
                tmp = pickle.load(open(dir_name+'/rf_data/dlp/rf_data'+str(c)+'.p', 'rb'))
            else :
                tmp = pickle.load(open(dir_name+'/rf_data/phi/rf_data'+str(c)+'.p', 'rb'))
            data[cases[c]] = tmp[cases[c]]
    elif moment_order == 0 :
        for c in range(len(cases)) :
            if c < nDLPcases :
                tmp = pickle.load(open(dir_name+'/rf_data/dlp/rf_data'+str(c)+'.p', 'rb'))
            else :
                tmp = pickle.load(open(dir_name+'/rf_data/partial/rf_data'+str(c)+'.p', 'rb'))
            data[cases[c]] = tmp[cases[c]]
    else :
        for c in range(len(cases)) :
            if c < nDLPcases :
                tmp = pickle.load(open(dir_name+'/rf_data/dlp/rf_data'+str(c)+'.p', 'rb'))
            else :
                tmp = pickle.load(open(dir_name+'/rf_data/angular/rf_data'+str(c)+'order'+str(moment_order)+'.p', 'rb'))
            data[cases[c]] = tmp[cases[c]]

    n = 10
    x = range(n) 

    for c in range(len(cases)) :
        data[cases[c]]['err_p_rel'] = np.zeros(n)
        data[cases[c]]['err_pp_rel'] = np.zeros(n)
        for o in range(n) :
            pwr =  data[cases[c]]['power'][o]
            data[cases[c]]['err_p_rel'][o] = np.max(np.abs(pwr[pmask]-ref_power[pmask])/ref_power[pmask])
            pwr =  data[cases[c]]['pin_p'][o]
            data[cases[c]]['err_pp_rel'][o] = np.max(np.abs(pwr[ppmask]-ref_pin_power[ppmask])/ref_pin_power[ppmask])

    leg = ['DLP', 'mDLP', 'Combined-Assemblies', 'Combined-Pins', 'Small-Core', 'Small-Assemblies', '1-D Approximation']

    plt.figure(0, figsize=(8, 6.5))
    plt.semilogy(x, 100.0 * data['dlp']['err_k'][0:n],            'k-',  
                 x, 100.0 * data['dlp1_L']['err_k'][0:n],         'k--', 
 				 x, 100.0 * data['assays']['err_k'][0:n],         'g-', 
				 x, 100.0 * data['pins']['err_k'][0:n],           'b-', 
				 x, 100.0 * data['small_core']['err_k'][0:n],     'r--',
				 x, 100.0 * data['small_assays']['err_k'][0:n],   'g--',
				 x, 100.0 * data['1D']['err_k'][0:n],             'b--', linewidth=1.85)
    plt.xlabel('order', fontsize=22)
    plt.ylabel('k absolute relative error (%)', fontsize=22)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.legend(leg, loc = 0, prop=fontP)
    plt.grid(True)
    if use_angular_moments == False :
        plt.savefig(dir_name+'/rf_plots/phi/energy_basis_comparison_keff-'+str(n)+'.pdf')
    elif moment_order == 0 :
        plt.savefig(dir_name+'/rf_plots/partial/partial_energy_basis_comparison_keff-'+str(n)+'.pdf')
    else :
        plt.savefig(dir_name+'/rf_plots/angular/angular'+str(moment_order)+'_energy_basis_comparison_keff-'+str(n)+'.pdf')
    
    
    
    plt.figure(1, figsize=(8, 6.5))
    plt.axis([0,n,1e-2, 1e2])
    plt.semilogy(x, 100.0 * data['dlp']['err_p_rel'][0:n],            'k-',  
                 x, 100.0 * data['dlp1_L']['err_p_rel'][0:n],         'k--', 
 				 x, 100.0 * data['assays']['err_p_rel'][0:n],         'g-', 
				 x, 100.0 * data['pins']['err_p_rel'][0:n],           'b-', 
				 x, 100.0 * data['small_core']['err_p_rel'][0:n],     'r--',
				 x, 100.0 * data['small_assays']['err_p_rel'][0:n],   'g--',
				 x, 100.0 * data['1D']['err_p_rel'][0:n],             'b--', linewidth=1.85)
    plt.xlabel('order', fontsize=22)
    plt.ylabel('fission density maximum relative error ($\%$)', fontsize=22) 
    plt.axhline(y=1e-1,xmin=0,xmax=1, c='k', ls='-')
    plt.annotate('Goal', xy=(4,1e-1), xytext=(1,4.7e-2))
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.legend(leg, loc=0, prop=fontP)
    plt.grid(True)
    if use_angular_moments == False :
        plt.savefig(dir_name+'/rf_plots/phi/energy_basis_comparison_fission-'+str(n)+'.pdf')
    elif moment_order == 0 :
        plt.savefig(dir_name+'/rf_plots/partial/partial_energy_basis_comparison_fission-'+str(n)+'.pdf')
    else :
        plt.savefig(dir_name+'/rf_plots/angular/moment'+str(moment_order)+'_energy_basis_comparison_fission-'+str(n)+'.pdf')
        
    plt.figure(2, figsize=(8, 6.5))
    plt.axis([0,n,1e-2, 1e2])
    plt.semilogy(x, 100.0 * data['dlp']['err_pp_rel'][0:n],            'k-',  
                 x, 100.0 * data['dlp1_L']['err_pp_rel'][0:n],         'k--', 
 				 x, 100.0 * data['assays']['err_pp_rel'][0:n],         'g-', 
				 x, 100.0 * data['pins']['err_pp_rel'][0:n],           'b-', 
				 x, 100.0 * data['small_core']['err_pp_rel'][0:n],     'r--',
				 x, 100.0 * data['small_assays']['err_pp_rel'][0:n],   'g--',
				 x, 100.0 * data['1D']['err_pp_rel'][0:n],             'b--', linewidth=1.85)
    plt.xlabel('order', fontsize=22)
    plt.ylabel('pin power maximum relative error ($\%$)', fontsize=22) 
    plt.axhline(y=1e-1,xmin=0,xmax=1, c='k', ls='-')
    plt.annotate('Goal', xy=(4,1e-1), xytext=(1,4.7e-2))
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.legend(leg, loc=0, prop=fontP)
    plt.grid(True)
    if use_angular_moments == False :
        plt.savefig(dir_name+'/rf_plots/phi/energy_basis_comparison_pinpower-'+str(n)+'.pdf')
    elif moment_order == 0 :
        plt.savefig(dir_name+'/rf_plots/partial/partial_energy_basis_comparison_pinpower-'+str(n)+'.pdf')
    else :
        plt.savefig(dir_name+'/rf_plots/angular/moment'+str(moment_order)+'_energy_basis_comparison_pinpower-'+str(n)+'.pdf')
    
def angular_comparison(tog44=False) :
    data = {}
    
    fontP = FontProperties()
    fontP.set_size('medium')
    
    # Load reference rf data
    rfdata =  pickle.load(open(dir_name+'/refdata/reference_rf_data.p', "rb"))
    ref_power = rfdata['power']

    for o in range(4) :
        data[o] = {}
        for c in range(len(cases)) :
            if o > 0 :
                if c > 1 :
                    tmp = pickle.load(open(dir_name+'/rf_data/angular/rf_data'+str(case)+'order'+str(o-1)+'.p', 'rb'))
                else :
                    tmp = pickle.load(open(dir_name+'/rf_data/partial/rf_data'+str(case)+'.p', 'rb')) 
                data[o][cases[c]] = tmp[cases[c]]
            else :
                tmp = pickle.load(open(dir_name+'/rf_data/phi/rf_data'+str(case)+'.p', 'rb'))
                data[o][cases[c]] = tmp[cases[c]]
    
    
    if tog44 == True :
        n = 44 - 1
    else :
        n = len(data[0]['core']['err_p']) - 1
    x = range(0, n) 
    for order in range(4) :
        for c in range(len(cases)) :
            data[order][cases[c]]['err_p_rel'] = np.zeros(n)
            for o in range(0, n) :
                pwr =  data[order][cases[c]]['power'][o]
                data[order][cases[c]]['err_p_rel'][o] = np.max(np.abs(pwr-ref_power)/ref_power)    

    leg = ['Only $\phi$', '$0^{th}$ moment', '$1^{st}$ moment', '$2^{nd}$ moment']
    
    for c in range(nDLPcases, len(cases)):
        plt.figure(0, figsize=(8, 6.5))
        plt.semilogy(x, 100.0 * data[0][cases[c]]['err_k'][0:n],      'k-', \
                     x, 100.0 * data[1][cases[c]]['err_k'][0:n],      'b-', \
     				 x, 100.0 * data[2][cases[c]]['err_k'][0:n],      'g-', \
				     x, 100.0 * data[3][cases[c]]['err_k'][0:n],      'r-', linewidth=1.85 )

        plt.xlabel('order')
        plt.ylabel('$k$ absolute relative error ($\%$)')
        plt.legend(leg, loc=0, prop=fontP)
        plt.grid(True)
        plt.savefig(dir_name+'/rf_plots/angular/angular_comparison_keff_'+cases[c]+'-'+str(n+1)+'.pdf')
        plt.clf()
        
        plt.figure(1, figsize=(8, 6.5))
        plt.axis([0,n,1e-7, 1e2])
        plt.semilogy(x, 100.0 * data[0][cases[c]]['err_p_rel'][0:n],   'k-', \
                     x, 100.0 * data[1][cases[c]]['err_p_rel'][0:n],   'b-', \
     				 x, 100.0 * data[2][cases[c]]['err_p_rel'][0:n],   'g-', \
				     x, 100.0 * data[3][cases[c]]['err_p_rel'][0:n],   'r-', linewidth=1.85 )
        plt.xlabel('order', fontsize=22)
        plt.ylabel('fission density maximum relative error ($\%$)', fontsize=22) 
        plt.axhline(y=1e-1,xmin=0,xmax=1, c='k', ls='-')
        if tog44 == True :
            plt.annotate('Goal', xy=(4,1e-1), xytext=(1,4.7e-2))
        else :
            plt.annotate('Goal', xy=(4,1e-1), xytext=(n-15,4.7e-2))
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.legend(leg, loc=3, prop=fontP)
        plt.grid(True)
        plt.savefig(dir_name+'/rf_plots/angular/angular_comparison_fission_'+cases[c]+'-'+str(n+1)+'.pdf')
        #plt.show()
        plt.clf()    
        
        
        
        
        
        
