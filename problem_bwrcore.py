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
    
def get_mesh(id) :
    """ Get a fuel, moderator, or core mesh
    """
    
    if id == 0 :
        n = 70
        if config_number == 0 :
            core_map = [0,1,0,1,0,1,0]
        elif config_number == 1 :
            core_map = [0,2,0,2,0,2,0]
        elif config_number == 2 :
            core_map = [0,3,0,3,0,3,0]
    elif id == 1 :
        n = 10
        core_map = [0]
    elif id == 2 :
        n = 10
        core_map = [1]
    elif id == 3 :
        n = 10
        core_map = [2]
    elif id == 4 :
        n = 10
        core_map = [3]      
    
    assay_map = []  
    assay_map.append([4, 0, 0, 0, 0, 0, 0, 0, 0, 4])
    assay_map.append([4, 0, 0, 1, 1, 1, 1, 0, 0, 4])
    assay_map.append([4, 1, 1, 2, 2, 2, 2, 1, 1, 4])
    assay_map.append([4, 2, 2, 2, 2, 2, 2, 2, 2, 4])
    
    pin_map = []
    fm_pin = [6, 16, 6]
    if id < 5 :
        am_map = []
        for i in range(len(core_map)):
            for j in range(len(assay_map[core_map[i]])):
                pin_map.append(assay_map[core_map[i]][j])
                am_map += [i,i,i]       
    else :
        n = 1
        fm_pin = [60, 160, 60]
        if id == 5 :
            pin_map = [0]
        elif id == 6 :
            pin_map = [1]
        elif id == 7 :
            pin_map = [2]
        elif id == 8 :
            pin_map = [3]
    if Comm.rank() == 0 :
        print pin_map
    # Define the "pin coarse mesh edges.
    cm_pin = [0.27, 0.99, 1.26]
    
    # Define the base fine mesh count per coarse mesh.
    
    # An assembly is composed of 10 adjacent pins.
    
    cm = [0.0]
    fm = []
    cm_map = []
    for i in range(n) :
        for j in range(len(cm_pin)) :
            cm.append(cm_pin[j] + 1.26*float(i))
            fm.append(fm_pin[j])
        cm_map += [i,i,i]     
        
    """
    Material  0 is UO_2 at 4.5% enrichment
    Material  1 is UO_2 at 2.5% enrichment
    Material  2 is UO_2 at 4.5% enrichment w/ Gd at 5%
    Material  3 is UO_2 at 2.5% enrichment w/ Gd at 5%
    Material  4 is moderator
    """
    mts = []
    for i in range(0, 7, 2) :
        mts.append([i+1, i, i+1])
    mts.append([1, 1, 1])

    mt = [] 
    for i in range(0, len(pin_map)) :
        mt += mts[pin_map[i]]
    
    mesh = Mesh1D.Create(fm, cm, mt)
    mesh.add_coarse_mesh_map("PINS", cm_map)
    if id < 5 :
        mesh.add_coarse_mesh_map("ASSAYS", am_map)
    
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
        Data = np.loadtxt(dir_name+'/refdata/core_flux')
    elif c == 1 :
        Data = np.loadtxt(dir_name+'/refdata/assay1_flux')
        Data1 = np.loadtxt(dir_name+'/refdata/assay'+str(2+config_number)+'_flux')
        Data = np.concatenate((Data, Data1), axis=1)
    elif c == 2 :
        Data = np.loadtxt(dir_name+'/refdata/fuel1_flux')
        if config_number == 0 :    
            Data1 = np.loadtxt(dir_name+'/refdata/fuel2_flux')
        elif config_number == 2 :
            Data1 = np.loadtxt(dir_name+'/refdata/fuel3_flux')
        elif config_number == 1 :
            Data1 = np.loadtxt(dir_name+'/refdata/fuel2_flux')
            Data2 = np.loadtxt(dir_name+'/refdata/fuel3_flux')
            Data1 = np.concatenate((Data1, Data2), axis=1)
        Data = np.concatenate((Data, Data1), axis=1)
    if Comm.rank() == 0 :
        print 'size = ', Data.shape
        
    if use_angular_moments == False :
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
        if case == 2 :
            angdata = np.loadtxt(dir_name+'/refdata/core_moment'+str(o))
        elif case == 3 :
            angdata = np.loadtxt(dir_name+'/refdata/assay1_moment'+str(o))
            angdata1 = np.loadtxt(dir_name+'/refdata/assay'+str(2+config_number)+'_moment'+str(o))
            angdata = np.concatenate((angdata, angdata1), axis=1)
        elif case == 4 :
            angdata = np.loadtxt(dir_name+'/refdata/fuel1_moment'+str(o))
            if config_number == 0 :    
                angdata1 = np.loadtxt(dir_name+'/refdata/fuel2_moment'+str(o))
            elif config_number == 2 :
                angdata1 = np.loadtxt(dir_name+'/refdata/fuel3_moment'+str(o))
            elif config_number == 1 :
                angdata1 = np.loadtxt(dir_name+'/refdata/fuel2_moment'+str(o))
                angdata2 = np.loadtxt(dir_name+'/refdata/fuel3_moment'+str(o))
                angdata1 = np.concatenate((angdata1, angdata2), axis=1)
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
    if config_number == 0 :
        mesh1 = get_mesh(2)
    if config_number == 1 :
        mesh1 = get_mesh(3)
    if config_number == 2 :
        mesh1 = get_mesh(4)
    # orders
    so = [[0],[0]]
    po = 3 
    po = [po,po]
    ao = [0, 0]
    eo = [energy_order, energy_order]
    # nodes
    w  = [mesh0.total_width_x(), 1.0, 1.0]
    nodes = []
    if rfdb == True :
        if Comm.rank() == 0 :
            print 'Using Cartesian Nodes'
        node = CartesianNode.Create(1, 'node0', so, po, ao, eo, w)
        node.set_number_pins(10)
        nodes.append(node)
        node = CartesianNode.Create(1, 'node1', so, po, ao, eo, w)
        node.set_number_pins(10)
        nodes.append(node)
    else :
        if Comm.rank() == 0 :
            print 'Using Detran Nodes'
        nodes.append(CartesianNodeDetran.Create(1, 'node0', so, po, ao, eo, w, inp, mat, mesh0))
        nodes.append(CartesianNodeDetran.Create(1, 'node1', so, po, ao, eo, w, inp, mat, mesh1))
    if dbgen == True:
        nodal_map = [0,1]
    else :
        nodal_map = [0,1,0,1,0,1,0]
    # global boundary conditions
    bc = [Node.VACUUM, Node.VACUUM]
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
    angularmoment = []
    if nGroups == 238 :
        thermal_cutoff_group = 203
    elif nGroups == 44 :
        thermal_cutoff_group = 27
    for n in range(9) :
        if n == 0 :
            db.put_str("bc_west",      "vacuum")
            db.put_str("bc_east",      "vacuum")
        else :
            db.put_str("bc_west",      "reflect")
            db.put_str("bc_east",      "reflect") 
        x = []
        mesh = get_mesh(n)
        solver = Eigen1D(db, mat, mesh)
        solver.solve()
        if n == 0 :
            keff = solver.state().eigenvalue()
            rates = ReactionRates(mat, mesh, solver.state())
            pin_power = rates.region_power("PINS")
            assay_power = rates.region_power("ASSAYS")
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
        elif n < 5 :
            np.savetxt(dir_name+'/refdata/assay'+str(n)+'_flux', phig)
        else :
            np.savetxt(dir_name+'/refdata/fuel'+str(n-4)+'_flux', phig)
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
            elif n < 5 :
                np.savetxt(dir_name+'/refdata/assay'+str(n)+'_moment'+str(order), angmom[order])
            else :
                np.savetxt(dir_name+'/refdata/fuel'+str(n-4)+'_moment'+str(order), angmom[order])
        angularmoment.append(angmom)
    
    #--------------------------------------------------------------------------#
    # PLOTS AND DATA
    #--------------------------------------------------------------------------#
    
    if config_number == 0 :
        phimap = [0,1,2]
    elif config_number == 1 :
        phimap = [0,1,3]
    elif config_number == 2 :
        phimap = [0,1,4]

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
    for a in phimap: 
        y, A0 = common.barchart(EG, np.mean(phi[a], axis=1))
        A.append(A0)
    C = []
    for order in range(nOrder) :
        y, B00 = common.barchart(EG, Save[order])
        C.append(B00)

    groups = range(0, nGroups)
    fontP = FontProperties()
    fontP.set_size('medium')
    plt.figure(0, figsize=(8, 6.5))
    plt.semilogy(groups, np.mean(phi[phimap[0]], axis=1),      'k-',
                 groups, np.mean(phi[phimap[1]], axis=1),      'b-',
                 groups, np.mean(phi[phimap[2]], axis=1),      'g-')
    plt.xlabel('$g$')
    plt.ylabel('$\phi_g$')
    plt.xlim([1e-5, 1e8])
    leg = ['core', 'assay1', 'assay'+str(phimap[2])]
    plt.legend(leg, loc = 0, prop=fontP)
    plt.grid(True)
    plt.savefig(dir_name+'/reference_figures/'+str(nGroups)+'group_spectra.pdf')

    # Spectra Plot
    plt.figure(1, figsize=(8, 6.5))
    plt.loglog(y, A[0],     'k-',
               y, A[1],     'b-',
               y, A[2],     'g-')
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
    
    leg = ['Fast Phi', 'Thermal Phi']
    plt.figure(3, figsize=(8, 6.5))
    plt.plot(x, np.mean(phi[0][:thermal_cutoff_group][0:len(x)],axis=0), 'k-', \
             x, np.mean(phi[0][thermal_cutoff_group:][0:len(x)],axis=0), 'g-'  )
    plt.xlabel('position in cm')
    plt.ylabel('$\phi$')
    plt.legend(leg, loc = 0, prop=fontP)
    plt.grid(True)
    plt.savefig(dir_name+'/reference_figures/spectra_core_energyaveraged.pdf')
    
    # Moment 0 Plot
    
    leg = ['$0^{th}$ moment', '$1^{st}$ moment', '$2^{nd}$ moment']
    plt.figure(6, figsize=(8, 6.5))
    plt.plot(x, np.mean(angularmoment[0][0], axis=0),     'k-', \
             x, np.mean(angularmoment[0][1], axis=0),     'r-', \
             x, np.mean(angularmoment[0][2], axis=0),     'g-'  )
    plt.xlabel('position in cm')
    plt.ylabel('${moment0}_g$')
    plt.legend(leg, loc = 0, prop=fontP)
    plt.grid(True)
    plt.savefig(dir_name+'/reference_figures/'+str(nGroups)+'monent_spectra_full.pdf')
    
    data = {}
    data['pin_power'] = pin_power
    data['assay_power'] = assay_power
    data['phi_full'] = np.mean(phi[0],axis=1)
    data['keff'] = keff
    pickle.dump(data, open(dir_name+'/refdata/reference_data.p', 'wb'))   
    
def process_rf_data(tog44=False) :
    data = {}
    
    fontP = FontProperties()
    fontP.set_size('medium')
    
    # Load reference rf data
    rfdata =  pickle.load(open(dir_name+'/refdata/reference_rf_data.p', "rb"))
    ref_power = rfdata['power']
    ref_pin_power = rfdata['pin_p']

    if use_angular_moments == False :
        for c in range(len(cases)) :
            if c < nDLPcases :
                tmp = pickle.load(open(dir_name+'/rf_data/dlp/rf_data'+str(case)+'.p', 'rb'))
            else :
                tmp = pickle.load(open(dir_name+'/rf_data/phi/rf_data'+str(case)+'.p', 'rb'))
            data[cases[c]] = tmp[cases[c]]
    elif moment_order == 0 :
        for c in range(len(cases)) :
            if c < nDLPcases :
                tmp = pickle.load(open(dir_name+'/rf_data/dlp/rf_data'+str(case)+'.p', 'rb'))
            else :
                tmp = pickle.load(open(dir_name+'/rf_data/partial/rf_data'+str(case)+'.p', 'rb'))
            data[cases[c]] = tmp[cases[c]]
    else :
        for c in range(len(cases)) :
            if c < nDLPcases :
                tmp = pickle.load(open(dir_name+'/rf_data/dlp/rf_data'+str(case)+'.p', 'rb'))
            else :
                tmp = pickle.load(open(dir_name+'/rf_data/angular/rf_data'+str(case)+'order'+str(moment_order)+'.p', 'rb'))
            data[cases[c]] = tmp[cases[c]]
    
    if tog44 == True :
        n = 44 - 1
    else :
        #n = len(data['dlp1_L']['err_p']) - 1
        n = len(data['dlp1_L']['err_p'])
    x = range(n) 

    for c in range(len(cases)) :
        data[cases[c]]['err_p_rel'] = np.zeros(n)
        data[cases[c]]['err_pp_rel'] = np.zeros(n)
        for o in range(0, n) :
            pwr =  data[cases[c]]['power'][o]
            data[cases[c]]['err_p_rel'][o] = np.max(np.abs(pwr-ref_power)/ref_power)
            pwr =  data[cases[c]]['pin_p'][o]
            data[cases[c]]['err_pp_rel'][o] = np.max(np.abs(pwr-ref_pin_power)/ref_pin_power)

    leg = ['DLP', 'mDLP', 'Full-Core', 'Combined-Assemblies', 'Combined-Pins']

    plt.figure(0, figsize=(8, 6.5))
    plt.axis([0,n,1e-7, 1e2])
    plt.semilogy(x, 100.0 * data['dlp']['err_k'][0:n],             'k-',  \
                 x, 100.0 * data['dlp1_L']['err_k'][0:n],          'k--', \
 				 x, 100.0 * data['core']['err_k'][0:n],            'r-',  \
				 x, 100.0 * data['assays']['err_k'][0:n],          'g-', \
				 x, 100.0 * data['pins']['err_k'][0:n],            'b-', linewidth=1.85  )
    plt.xlabel('order', fontsize=22)
    plt.ylabel('$k$ absolute relative error ($\%$)', fontsize=22)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.legend(leg, loc = 0, prop=fontP)
    plt.grid(True)
    if use_angular_moments == False :
        print dir_name+'/rf_plots/phi/energy_basis_comparison_keff-'+str(n+1)+'.pdf'
        plt.savefig(dir_name+'/rf_plots/phi/energy_basis_comparison_keff-'+str(n+1)+'.pdf')
    elif moment_order == 0 :
        print dir_name+'/rf_plots/partial/partial_energy_basis_comparison_keff-'+str(n+1)+'.pdf'
        plt.savefig(dir_name+'/rf_plots/partial/partial_energy_basis_comparison_keff-'+str(n+1)+'.pdf')
    else :
        print dir_name+'/rf_plots/angular/angular'+str(moment_order)+'_energy_basis_comparison_keff-'+str(n+1)+'.pdf'
        plt.savefig(dir_name+'/rf_plots/angular/angular'+str(moment_order)+'_energy_basis_comparison_keff-'+str(n+1)+'.pdf')
    
    plt.figure(1, figsize=(8, 6.5))
    plt.axis([0,n,1e-7, 1e2])
    plt.semilogy(x, 100.0 * data['dlp']['err_p_rel'][0:n],             'k-',  \
                 x, 100.0 * data['dlp1_L']['err_p_rel'][0:n],          'k--', \
 				 x, 100.0 * data['core']['err_p_rel'][0:n],            'r-',  \
				 x, 100.0 * data['assays']['err_p_rel'][0:n],          'g-', \
				 x, 100.0 * data['pins']['err_p_rel'][0:n],            'b-', linewidth=1.85)
    plt.xlabel('order', fontsize=22)
    plt.ylabel('fission density maximum relative error ($\%$)', fontsize=22) 
    plt.axhline(y=1e-1,xmin=0,xmax=1, c='k', ls='-')
    plt.annotate('Goal', xy=(4,1e-1), xytext=(1,4.7e-2))
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.legend(leg, loc=0, prop=fontP)
    plt.grid(True)
    if use_angular_moments == False :
        print dir_name+'/rf_plots/phi/energy_basis_comparison_fission-'+str(n+1)+'.pdf'
        plt.savefig(dir_name+'/rf_plots/phi/energy_basis_comparison_fission-'+str(n+1)+'.pdf')
    elif moment_order == 0 :
        print dir_name+'/rf_plots/partial/partial_energy_basis_comparison_fission-'+str(n+1)+'.pdf'
        plt.savefig(dir_name+'/rf_plots/partial/partial_energy_basis_comparison_fission-'+str(n+1)+'.pdf')
    else :
        print dir_name+'/rf_plots/angular/moment'+str(moment_order)+'_energy_basis_comparison_fission-'+str(n+1)+'.pdf'
        plt.savefig(dir_name+'/rf_plots/angular/moment'+str(moment_order)+'_energy_basis_comparison_fission-'+str(n+1)+'.pdf')
      
    plt.figure(2, figsize=(8, 6.5))
    plt.axis([0,n,1e-7, 1e2])
    plt.semilogy(x, 100.0 * data['dlp']['err_pp_rel'][0:n],             'k-',  \
                 x, 100.0 * data['dlp1_L']['err_pp_rel'][0:n],          'k--', \
 				 x, 100.0 * data['core']['err_pp_rel'][0:n],            'r-',  \
				 x, 100.0 * data['assays']['err_pp_rel'][0:n],          'g-', \
				 x, 100.0 * data['pins']['err_pp_rel'][0:n],            'b-', linewidth=1.85)
    plt.xlabel('order', fontsize=22)
    plt.ylabel('pin power maximum relative error ($\%$)', fontsize=22) 
    plt.axhline(y=1e-1,xmin=0,xmax=1, c='k', ls='-')
    plt.annotate('Goal', xy=(4,1e-1), xytext=(1,4.7e-2))
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.legend(leg, loc=0, prop=fontP)
    plt.grid(True)
    if use_angular_moments == False :
        print dir_name+'/rf_plots/phi/energy_basis_comparison_pinpower-'+str(n+1)+'.pdf'
        plt.savefig(dir_name+'/rf_plots/phi/energy_basis_comparison_pinpower-'+str(n+1)+'.pdf')
    elif moment_order == 0 :
        print dir_name+'/rf_plots/partial/partial_energy_basis_comparison_pinpower-'+str(n+1)+'.pdf'
        plt.savefig(dir_name+'/rf_plots/partial/partial_energy_basis_comparison_pinpower-'+str(n+1)+'.pdf')
    else :
        print dir_name+'/rf_plots/angular/moment'+str(moment_order)+'_energy_basis_comparison_pinpower-'+str(n+1)+'.pdf'
        plt.savefig(dir_name+'/rf_plots/angular/moment'+str(moment_order)+'_energy_basis_comparison_pinpower-'+str(n+1)+'.pdf')
    
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

    leg = ['Only $\phi$', '0th moment', '1st moment', '2nd moment']
    
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
        print dir_name+'/rf_plots/angular/angular_comparison_keff_'+cases[c]+'-'+str(n+1)+'.pdf'
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
        print dir_name+'/rf_plots/angular/angular_comparison_fission_'+cases[c]+'-'+str(n+1)+'.pdf'
        plt.savefig(dir_name+'/rf_plots/angular/angular_comparison_fission_'+cases[c]+'-'+str(n+1)+'.pdf')
        #plt.show()
        plt.clf()     
