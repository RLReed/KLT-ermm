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

cases = ['dlp1_L', 'dlp1_U', 'dlp1_M', 'dlp2_L', 'dlp2_U', 'dlp2_M', 'dlp',
         '10-pin', 'uo2', 'mox', '1-pin', '2-pin', '6-pin', 'combined_pin']
nDLPcases = 7
nKLTcases = int(len(cases) - nDLPcases)


def get_material() :
    """ Return the three material, 238 group library generated from SCALE 6.1.
        Materials 0 and 2 are UO2 and MOX while 1 is moderator.
    """
    io = IO_HDF5('cross_section_libraries/'+str(test_problem)+'_'+str(nGroups)+'_library.h5')
    mat = io.read_material()
    mat.compute_sigma_a()
    mat.compute_diff_coef()
    mat.finalize()
    
    return mat
    
def get_mesh(id) :
    """ Get a fuel, moderator, or core mesh
    """
    
    # Define the "pin coarse mesh edges.
    cm_pin = [0.09, 1.17, 1.26]

    # Define the base fine mesh count per coarse mesh.
    fm_pin = [3, 22, 3]
    
    # An assembly is composed of 10 adjacent pins.
    if id == 0 :
        n = 10
    elif id in [3, 4] :
        fm_pin = [15, 105, 15]
        n = 2
    elif id == 5 :
        fm_pin = [6, 44, 6]
        n = 6
    else :
        n = 1
        fm_pin = [30, 210, 30]
    
    cm = [0.0]
    fm = []
    cm_map = []
    for i in range(0, n) :
        for j in range(0, len(cm_pin)) :
            cm.append(cm_pin[j] + 1.26*float(i))
            fm.append(fm_pin[j])
        cm_map += [i,i,i]
    if id == 1 :
        pin_map = [0]
    elif id == 2 :
        pin_map = [1]
    elif id in [3, 4] :
        pin_map = [0, 1]
    elif id == 5 :
        pin_map = [0,0,0,  1,1,1]
    else :
        pin_map = [0,0,0,0,0,  1,1,1,1,1]
    if Comm.rank() == 0 :
        print 'pin_map = ', pin_map
    mts = []
    mts.append([1, 0, 1])
    mts.append([1, 2, 1])
    mts.append([1, 1, 1])

    mt = [] 
    for i in range(0, len(pin_map)) :
        mt += mts[pin_map[i]]
    mesh = Mesh1D.Create(fm, cm, mt)
    mesh.add_coarse_mesh_map("PINS", cm_map)

    return mesh
    
def get_db(case) :
	
    basis_inp = InputDB.Create("basis_data")
    """ Provides the base nodal db (or for use in the reference)
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
    db.put_str("bc_west",                        "fixed")
    db.put_str("bc_east",                        "fixed")
    db.put_int("erme_angular_expansion",         1)
    if case >= nDLPcases :
        A = common.KLT(case)
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
        Data = np.loadtxt(dir_name+'/refdata/10-pin_mg_flux')
    elif c == 1 :
        Data = np.loadtxt(dir_name+'/refdata/uo2_mg_flux')
    elif c == 2 :
        Data = np.loadtxt(dir_name+'/refdata/mox_mg_flux')
    elif c == 3 :
        Data = np.loadtxt(dir_name+'/refdata/1-pin_mg_flux')
    elif c == 4 :
        Data = np.loadtxt(dir_name+'/refdata/2-pin_mg_flux')
    elif c == 5 :
        Data = np.loadtxt(dir_name+'/refdata/6-pin_mg_flux')
    elif c == 6 :
        Data = np.loadtxt(dir_name+'/refdata/mox_mg_flux')
        Data1 = np.loadtxt(dir_name+'/refdata/uo2_mg_flux')
        Data = np.concatenate((Data, Data1), axis=1)
        Data1 = np.loadtxt(dir_name+'/refdata/1-pin_mg_flux')
        Data = np.concatenate((Data, Data1), axis=1)
    if Comm.rank() == 0 :
        print 'size = ', Data.shape
        if Comm.rank() == 0 :
            print 'removing duplicate snapshots'
        Data = Data.T
        b = np.ascontiguousarray(np.around(Data, decimals=10)).view(np.dtype((np.void, Data.dtype.itemsize * Data.shape[1])))
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
            angdata = np.loadtxt(dir_name+'/refdata/10-pin_mg_moment'+str(o))
        elif c == 1 :
            angdata = np.loadtxt(dir_name+'/refdata/uo2_mg_moment'+str(o))
        elif c == 2 :
            angdata = np.loadtxt(dir_name+'/refdata/mox_mg_moment'+str(o))
        elif c == 3 :
            angdata = np.loadtxt(dir_name+'/refdata/1-pin_mg_moment'+str(o))
        elif c == 4 :
            angdata = np.loadtxt(dir_name+'/refdata/2-pin_mg_moment'+str(o))
        elif c == 5 :
            angdata = np.loadtxt(dir_name+'/refdata/6-pin_mg_moment'+str(o))
        elif c == 6 :
            angdata = np.loadtxt(dir_name+'/refdata/mox_mg_moment'+str(o))
            angdata1 = np.loadtxt(dir_name+'/refdata/uo2_mg_moment'+str(o))
            angdata = np.concatenate((angdata, angdata1), axis=1)
            angdata1 = np.loadtxt(dir_name+'/refdata/1-pin_mg_moment'+str(o))
            angdata = np.concatenate((angdata, angdata1), axis=1)
        Data = np.concatenate((Data, angdata), axis=1)
        if Comm.rank() == 0 :
            print 'size = ', Data.shape
    if Comm.rank() == 0 :
        print 'removing duplicate snapshots'
    Data = Data.T
    b = np.ascontiguousarray(np.around(Data, decimals=10)).view(np.dtype((np.void, Data.dtype.itemsize * Data.shape[1])))
    Data = np.unique(b).view(Data.dtype).reshape(-1, Data.shape[1]).T
    if Comm.rank() == 0 :
        print 'reduced size = ', Data.shape
    return Data
        
def get_nodes(inp, energy_order, rfdb, dbgen) :
    mat = get_material()
    # meshes
    mesh0 = get_mesh(1)
    mesh1 = get_mesh(2)
    # orders
    so = [[0],[0]] 
    po = 1
    po = [po,po]
    ao = [0, 0]
    eo = [energy_order, energy_order]
    # nodes
    w  = [mesh0.total_width_x(), 1.0, 1.0]
    nodes = []
    if rfdb == True :
        if Comm.rank() == 0 :
            print 'Using Cartesian Nodes'
        nodes.append(CartesianNode.Create(1, 'node0', so, po, ao, eo, w))
        nodes.append(CartesianNode.Create(1, 'node1', so, po, ao, eo, w))
    else :
        if Comm.rank() == 0 :
            print 'Using Detran Nodes'
        nodes.append(CartesianNodeDetran.Create(1, 'node0', so, po, ao, eo, w, inp, mat, mesh0))
        nodes.append(CartesianNodeDetran.Create(1, 'node1', so, po, ao, eo, w, inp, mat, mesh1))
    if dbgen == True:
        nodal_map = [0, 1]
    else :
        nodal_map = [0,0,0,0,0, 1,1,1,1,1]
    # global boundary conditions
    bc = [Node.REFLECT, Node.REFLECT]
    nodes = build_model(nodes, nodal_map, bc)
    return nodes

def run_reference() :
    """ Run the reference eigenvalue problem using Detran.  Also, produce
        the reference modes for producing an orthogonal basis.
    """
    db   = get_db(0)
    db.put_int('adjoint',      0)
    db.put_int("store_angular_flux",             1)
    
    mat  = get_material()

    #--------------------------------------------------------------------------#
    # MODELS
    #--------------------------------------------------------------------------#
    phi = []
    numcells = []
    keff = []
    pin_power = []
    angularmoment = []
    nModels = len(cases)-1
    names = cases[nDLPcases:nModels]
    print names
    number = 0
    if nGroups == 238 :
        thermal_cutoff_group = 203
    elif nGroups == 44 :
        thermal_cutoff_group = 27
    for n in names :
        if n == '1-pin' :
            db.put_str("outer_solver", "GS")
            db.put_str("bc_west",      "periodic")
            db.put_str("bc_east",      "periodic")
        else :
            db.put_str("outer_solver", "GMRES")
            db.put_str("bc_west",      "reflect")
            db.put_str("bc_east",      "reflect") 
        print n
        x = []
        mesh = get_mesh(number) 
        number += 1  
        solver = Eigen1D(db, mat, mesh)
        solver.solve()
        if n == '10-pin' :
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
        np.savetxt(dir_name+'/refdata/'+n+'_mg_flux', phig)
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
            np.savetxt(dir_name+'/refdata/'+n+'_mg_moment'+str(order), angmom[order])
        angularmoment.append(angmom)
    
    #--------------------------------------------------------------------------#
    # PLOTS AND DATA
    #--------------------------------------------------------------------------#

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
        E = []
        S = []
        for c in range(len(names)) :
            E.append(np.mean(angularmoment[c][order], axis=0))
            S.append(np.mean(angularmoment[c][order], axis=1))
        Eave.append(E)
        Save.append(S)

    A = []
    # Data for Phi plots 
    for a in range(len(phi)): 
        y, A0 = common.barchart(EG, np.mean(phi[a], axis=1))
        A.append(A0)
    C = []
    for order in range(nOrder) :
        CC = []
        for core in range(len(names)) :
            y, B00 = common.barchart(EG, Save[order][core])
            CC.append(B00)
        C.append(CC)

    groups = range(0, nGroups)
    fontP = FontProperties()
    fontP.set_size('medium')
    plt.figure(0, figsize=(8, 6.5))
    plt.semilogy(groups, np.mean(phi[0], axis=1),      'k-', \
                 groups, np.mean(phi[1], axis=1),      'b-', \
                 groups, np.mean(phi[2], axis=1),      'g-', \
                 groups, np.mean(phi[3], axis=1),      'r-'  )
    plt.xlabel('$g$')
    plt.ylabel('$\phi_g$')
    leg = ['10-pin', 'UO$_2$', 'MOX', '1-pin']
    plt.legend(leg, loc = 0, prop=fontP)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(dir_name+'/reference_figures/'+str(nGroups)+'group_spectra.pdf')

    # Spectra Plot
    leg = ['10-pin', 'UO$_2$', 'MOX']
    plt.figure(1, figsize=(8, 6.5))
    plt.loglog(y, A[0],     'k-', \
               y, A[1],     'r-', \
               y, A[2],     'g-'  )
    plt.xlabel('energy in $eV$')
    plt.ylabel('$\phi_g$')
    plt.legend(leg, loc = 0, prop=fontP)
    plt.grid(True) 
    plt.tight_layout()
    plt.savefig(dir_name+'/reference_figures/'+str(nGroups)+'group_spectra_energy.pdf')   
    
    # Spatially Averaged Moment Plots
    # Moment 0 Plot
    plt.figure(2, figsize=(8, 6.5))
    plt.loglog(y, C[0][0],     'k-', \
               y, C[0][1],     'r-', \
               y, C[0][2],     'g-'  )
    plt.xlabel('energy in $eV$')
    plt.ylabel('${moment0}_g$')
    plt.legend(leg, loc = 0, prop=fontP)
    plt.grid(True) 
    plt.tight_layout()
    plt.savefig(dir_name+'/reference_figures/'+str(nGroups)+'monent0_spectra_energy.pdf')
    
    # Moment 1 Plot
    plt.figure(3, figsize=(8, 6.5))
    plt.loglog(y, C[1][0],     'k-', \
               y, C[1][1],     'r-', \
               y, C[1][2],     'g-'  )
    plt.xlabel('energy in $eV$')
    plt.ylabel('${moment1}_g$')
    plt.legend(leg, loc = 0, prop=fontP)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(dir_name+'/reference_figures/'+str(nGroups)+'monent1_spectra_energy.pdf')
    
    # Moment 2 Plot
    plt.figure(4, figsize=(8, 6.5))
    plt.loglog(y, C[2][0],     'k-', \
               y, C[2][1],     'r-', \
               y, C[2][2],     'g-'  )
    plt.xlabel('energy in $eV$')
    plt.ylabel('${moment2}_g$')
    plt.legend(leg, loc = 0, prop=fontP)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(dir_name+'/reference_figures/'+str(nGroups)+'monent2_spectra_energy.pdf')
    
    # Energy Averaged Plots
    x = numcells[0] 
    
    leg = ['Fast Phi', 'Thermal Phi']
    plt.figure(5, figsize=(8, 6.5))
    plt.plot(x, np.mean(phi[0][:thermal_cutoff_group][0:len(x)],axis=0), 'k-', \
             x, np.mean(phi[0][thermal_cutoff_group:][0:len(x)],axis=0), 'g-'  )
    plt.xlabel('position in cm')
    plt.ylabel('$\phi$')
    plt.legend(leg, loc = 0, prop=fontP)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(dir_name+'/reference_figures/spectra_full_energyaveraged.pdf')
    
    # Moment 0 Plot
    
    leg = ['$0^{th}$ moment', '$1^{st}$ moment', '$2^{nd}$ moment']
    plt.figure(6, figsize=(8, 6.5))
    plt.plot(x, Eave[0][0],     'k-', \
             x, Eave[1][0],     'r-', \
             x, Eave[2][0],     'g-'  )
    plt.xlabel('position in cm')
    plt.ylabel('${moment0}_g$')
    plt.legend(leg, loc = 0, prop=fontP)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(dir_name+'/reference_figures/'+str(nGroups)+'monent_spectra_full.pdf')
    
    # Moment 1 Plot
    x = numcells[1]
    n = len(x)
    plt.figure(7, figsize=(8, 6.5))
    plt.plot(x, Eave[0][1],     'k-', \
             x, Eave[1][1],     'r-', \
             x, Eave[2][1],     'g-'  )
    plt.xlabel('position in cm')
    plt.ylabel('${moment1}_g$')
    plt.legend(leg, loc = 0, prop=fontP)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(dir_name+'/reference_figures/'+str(nGroups)+'monent_spectra_uo2.pdf')
    
    # Moment 2 Plot
    x = numcells[2]
    n = len(x)
    plt.figure(8, figsize=(8, 6.5))
    plt.plot(x, Eave[0][2],     'k-', \
             x, Eave[1][2],     'r-', \
             x, Eave[2][2],     'g-'  )
    plt.xlabel('position in cm')
    plt.ylabel('${moment2}_g$')
    plt.legend(leg, loc = 0, prop=fontP)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(dir_name+'/reference_figures/'+str(nGroups)+'monent_spectra_mox.pdf')   
    
    data = {}
    data['pin_power'] = pin_power
    data['phi_full'] = np.mean(phi[0],axis=1)
    data['phi_uo2'] = np.mean(phi[1],axis=1)
    data['phi_mox'] = np.mean(phi[2],axis=1)
    data['keff'] = keff
    pickle.dump(data, open(dir_name+'/refdata/reference_data.p', 'wb'))
    
def process_rf_data(tog44=False) :
    data = {}
    
    fontP = FontProperties()
    fontP.set_size('medium')
    
    # Load reference rf data
    rfdata =  pickle.load(open(dir_name+'/refdata/reference_rf_data.p', "rb"))
    ref_power = rfdata['power']

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
    
    if tog44 == True :
        n = 44 - 1
    else :
        n = len(data['dlp1_L']['err_p']) - 1
    x = range(n) 

    for c in range(len(cases)) :
        data[cases[c]]['err_p_rel'] = np.zeros(n)
        for o in range(0, n) :
            pwr =  data[cases[c]]['power'][o]
            data[cases[c]]['err_p_rel'][o] = np.max(np.abs(pwr-ref_power)/ref_power)

    leg = ['DLP', 'mDLP', 'Full-Assembly', 'UO$_2$-Pin', 'MOX-Pin', 'Combined-Pins','1-pin', '2-pin', '3-pin']

    plt.figure(0, figsize=(8, 6.5))
    plt.semilogy(x, 100.0 * data['dlp']['err_k'][0:n],           'k-', \
                 x, 100.0 * data['dlp1_L']['err_k'][0:n],        'k--', \
 				 x, 100.0 * data['10-pin']['err_k'][0:n],        'r-', \
 				 x, 100.0 * data['uo2']['err_k'][0:n],           'b-', \
				 x, 100.0 * data['mox']['err_k'][0:n],           'b--', \
 				 x, 100.0 * data['combined_pin']['err_k'][0:n],  'g-', \
				 x, 100.0 * data['1-pin']['err_k'][0:n],         'g--', \
				 x, 100.0 * data['2-pin']['err_k'][0:n],         'g:', \
 				 x, 100.0 * data['6-pin']['err_k'][0:n],         'r--', linewidth=1.85  )
    plt.xlabel('order', fontsize=22)
    plt.ylabel('$k$ absolute relative error ($\%$)', fontsize=22)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.legend(leg, loc = 0, prop=fontP)
    plt.grid(True)
    plt.tight_layout()
    if use_angular_moments == False :
        plt.savefig(dir_name+'/energy_basis_comparison_keff-'+str(n+1)+'.pdf')
    elif moment_order == 0 :
        plt.savefig(dir_name+'/partial_energy_basis_comparison_keff-'+str(n+1)+'.pdf')
    else :
        plt.savefig(dir_name+'/angular'+str(moment_order)+'_energy_basis_comparison_keff-'+str(n+1)+'.pdf')
    
    plt.figure(1, figsize=(8, 6.5))
    plt.axis([0,n,1e-7, 1e2])
    plt.semilogy(x, 100.0 * data['dlp']['err_p_rel'][0:n],          'k-', \
                 x, 100.0 * data['dlp1_L']['err_p_rel'][0:n],       'k--', \
 				 x, 100.0 * data['10-pin']['err_p_rel'][0:n],       'r-', \
				 x, 100.0 * data['uo2']['err_p_rel'][0:n],          'b-', \
 				 x, 100.0 * data['mox']['err_p_rel'][0:n],          'b--', \
				 x, 100.0 * data['combined_pin']['err_p_rel'][0:n], 'g-', \
				 x, 100.0 * data['1-pin']['err_p_rel'][0:n],        'g--', \
 				 x, 100.0 * data['2-pin']['err_p_rel'][0:n],        'g:', \
				 x, 100.0 * data['6-pin']['err_p_rel'][0:n],        'r--', linewidth=1.85  )
    plt.xlabel('order', fontsize=22)
    plt.ylabel('fission density maximum relative error ($\%$)', fontsize=22) 
    plt.axhline(y=1e-1,xmin=0,xmax=1, c='k', ls='-')
    plt.annotate('Goal', xy=(4,1e-1), xytext=(1,4.7e-2))
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.legend(leg, loc=0, prop=fontP)
    plt.grid(True)
    plt.tight_layout()
    if use_angular_moments == False :
        plt.savefig(dir_name+'/rf_plots/phi/energy_basis_comparison_fission-'+str(n+1)+'.pdf')
    elif moment_order == 0 :
        plt.savefig(dir_name+'/rf_plots/partial/partial_energy_basis_comparison_fission-'+str(n+1)+'.pdf')
    else :
        plt.savefig(dir_name+'/rf_plots/angular/moment'+str(moment_order)+'_energy_basis_comparison_fission-'+str(n+1)+'.pdf')
    
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
        n = len(data[0]['dlp1_L']['err_p']) - 1
    x = range(0, n) 
    for order in range(4) :
        for c in range(len(cases)) :
            data[order][cases[c]]['err_p_rel'] = np.zeros(n)
            for o in range(0, n) :
                pwr =  data[order][cases[c]]['power'][o]
                data[order][cases[c]]['err_p_rel'][o] = np.max(np.abs(pwr-ref_power)/ref_power)    

    leg = ['Only $\phi$', '0th moment', '1st moment', '2nd moment']
    
    for c in range(2, len(cases)):
        plt.figure(0, figsize=(8, 6.5))
        plt.semilogy(x, 100.0 * data[0][cases[c]]['err_k'][0:n],      'k-', \
                     x, 100.0 * data[1][cases[c]]['err_k'][0:n],      'b-', \
     				 x, 100.0 * data[2][cases[c]]['err_k'][0:n],      'g-', \
				     x, 100.0 * data[3][cases[c]]['err_k'][0:n],      'r-', linewidth=1.85 )

        plt.xlabel('order')
        plt.ylabel('$k$ absolute relative error ($\%$)')
        plt.legend(leg, loc=0, prop=fontP)
        plt.grid(True)
        plt.tight_layout()
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
        plt.tight_layout()
        plt.savefig(dir_name+'/rf_plots/angular/angular_comparison_fission_'+cases[c]+'-'+str(n+1)+'.pdf')
        #plt.show()
        plt.clf()    
        
        
        
        
        
        
