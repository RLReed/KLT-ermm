from parameters import test_problem, nGroups, use_angular_moments, case, moment_order, config_number
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import cPickle as pickle
import os
import h5py
import sys

if test_problem == '10-pin' :
    dir_name = str(test_problem)+'/'+str(nGroups)+'group'
elif test_problem == 'bwrcore' :
    dir_name = str(test_problem)+'/'+str(nGroups)+'group/config'+str(config_number)
elif test_problem == 'c5g7' :
    dir_name = str(test_problem)+'/'+str(nGroups)+'group'

if test_problem == '10-pin' : from problem_10_pin import *
elif test_problem == 'bwrcore' : from problem_bwrcore import *
elif test_problem == 'c5g7' : from problem_c5g7 import *
    
def create_dirs() :
    cwd = os.getcwd()
    if not os.path.exists(str(cwd)+'/'+dir_name+'/databases') :
        os.makedirs(str(cwd)+'/'+dir_name+'/databases')
    if not os.path.exists(str(cwd)+'/'+dir_name+'/refdata') :
        os.makedirs(str(cwd)+'/'+dir_name+'/refdata')
    if not os.path.exists(str(cwd)+'/'+dir_name+'/reference_figures') :
        os.makedirs(str(cwd)+'/'+dir_name+'/reference_figures')
    if not os.path.exists(str(cwd)+'/'+dir_name+'/basis_functions') :
        os.makedirs(str(cwd)+'/'+dir_name+'/basis_functions')
    if not os.path.exists(str(cwd)+'/'+dir_name+'/basis_functions/phi') :
        os.makedirs(str(cwd)+'/'+dir_name+'/basis_functions/phi')
    if not os.path.exists(str(cwd)+'/'+dir_name+'/basis_functions/partial') :
        os.makedirs(str(cwd)+'/'+dir_name+'/basis_functions/partial')
    if not os.path.exists(str(cwd)+'/'+dir_name+'/basis_functions/angular') :
        os.makedirs(str(cwd)+'/'+dir_name+'/basis_functions/angular')
    if not os.path.exists(str(cwd)+'/'+dir_name+'/rf_data/dlp') :
        os.makedirs(str(cwd)+'/'+dir_name+'/rf_data/dlp')
    if not os.path.exists(str(cwd)+'/'+dir_name+'/rf_data/phi') :
        os.makedirs(str(cwd)+'/'+dir_name+'/rf_data/phi')
    if not os.path.exists(str(cwd)+'/'+dir_name+'/rf_data/partial') :
        os.makedirs(str(cwd)+'/'+dir_name+'/rf_data/partial')
    if not os.path.exists(str(cwd)+'/'+dir_name+'/rf_data/angular') :
        os.makedirs(str(cwd)+'/'+dir_name+'/rf_data/angular')
    if not os.path.exists(str(cwd)+'/'+dir_name+'/rf_plots/phi') :
        os.makedirs(str(cwd)+'/'+dir_name+'/rf_plots/phi')
    if not os.path.exists(str(cwd)+'/'+dir_name+'/rf_plots/partial') :
        os.makedirs(str(cwd)+'/'+dir_name+'/rf_plots/partial')
    if not os.path.exists(str(cwd)+'/'+dir_name+'/rf_plots/angular') :
        os.makedirs(str(cwd)+'/'+dir_name+'/rf_plots/angular')
    
def KLT(case) :
    
    Data = get_snapshots(case)
    size = Data.shape
    nSnap = size[1]

    DataT = Data.T
    A = DataT.dot(Data)
    w,A = np.linalg.eig(A)
    idx = w.argsort()[::-1] #sorts eigenvalues and vectors max->min
    w = w[idx]
    A = A[:,idx]
    A = np.real(Data.dot(A))
    
    datatest = False
    orth = 0
    while datatest == False :
        A,r = np.linalg.qr(A, mode='complete')
        sze = A.shape    
        I = np.identity(sze[0])
        T = np.real(A.dot(A.T))
        datatest = np.allclose(T, I, atol=1e-10)
        orth += 1
    if Comm.rank() == 0 : print 'nLoops = ', orth

    A = A.T
    if use_angular_moments == False :
        np.savetxt(dir_name+'/basis_functions/phi/functions_case'+str(case)+'.txt', A)
    elif moment_order == 0 :
        np.savetxt(dir_name+'/basis_functions/partial/functions_case'+str(case)+'.txt', A)
    else :
        np.savetxt(dir_name+'/basis_functions/angular/functions_case'+str(case)+'order'+str(moment_order)+'.txt', A)
    
    fontP = FontProperties()
    fontP.set_size('small')
    
    plt.figure(0, figsize=(8, 6.5))
    
    io = IO_HDF5('cross_section_libraries/'+str(test_problem)+'_'+str(nGroups)+'_library.h5')
    db = io.read_input()
    EG = db.get_vec_dbl('neutron_energy_bounds')
    x, A0 = barchart(EG, A[0])
    x, A1 = barchart(EG, A[1])
    x, A2 = barchart(EG, A[2])
    x, A3 = barchart(EG, A[3])   
    
    plt.semilogx(x, A0,   'k-', \
                 x, A1,   'r-', \
		         x, A2,   'b-', \
		         x, A3,   'g-')
    plt.xlabel('group')
    plt.ylabel('Normalized Basis Function') 
    leg = ['A$_0$', 'A$_1$', 'A$_2$', 'A$_3$']
    plt.legend(leg, loc=0, prop=fontP)
    plt.grid(True)
    if use_angular_moments == False :
        plt.savefig(dir_name+'/basis_functions/phi/basis_function_case'+str(case)+'.pdf')
    elif moment_order == 0 :
        plt.savefig(dir_name+'/basis_functions/partial/basis_function_case'+str(case)+'.pdf')
    else :
        plt.savefig(dir_name+'/basis_functions/angular/basis_function_case'+str(case)+'order'+str(moment_order)+'.pdf')
    #plt.show()
    
    return A
    
def barchart(x, y) :
    X = np.zeros(2*len(y))
    Y = np.zeros(2*len(y))
    for i in range(0, len(y)) :
        X[2*i]   = x[i]
        X[2*i+1] = x[i+1]
        Y[2*i]   = y[i]
        Y[2*i+1] = y[i]
    return X, Y
    
def generate_database(case, kstudy=False, num_keff=4, kdel=0.15) :

    order_reduction = 0
    db = InputDB.Create()
    db.put_int("dimension",                     1)
    if test_problem == 'c5g7':
        db.put_int("dimension",                 2)
    db.put_int("erme_maximum_iterations",       20)
    db.put_int("comm_local_groups",             1)
    # Be aware of this parameter
    db.put_int("erme_order_reduction",          order_reduction)
    
    refdata = pickle.load(open(dir_name+'/refdata/reference_data.p', "rb"))
    zeroth = []
    zeroth.append(refdata['phi_full'])
    if test_problem == '10-pin' :
        zeroth.append(refdata['phi_uo2'])
        zeroth.append(refdata['phi_mox'])
    k = refdata['keff']
    min_keff = k - kdel 
    max_keff = k + kdel
    
    if Comm.rank() == 0 : print 'Case = ', case

    # Loop over energy orders
    data = {}
    
    manager = ManagerERME.Create(sys.argv)     
    manager.build_comm(db)  
    node_db = get_db(case)  
    if case > nDLPcases - 1 :
        node_db.put_str("basis_e_type", "UserBasis")
    elif case == nDLPcases-1 :
        node_db.put_str("basis_e_type", "dlp")
    else :
        node_db.put_str("basis_e_type", "trans")
    if case < 3 :
        node_db.put_int("basis_e_transformed_option", 1)
    else :
        node_db.put_int("basis_e_transformed_option", 0)
    if case < nDLPcases - 1 :
        node_db.put_vec_dbl("basis_e_zeroth", zeroth[case % 3])
    
    if test_problem == 'c5g7':
        nodes = get_nodes(node_db, 10-1, False, True)
    else :
        nodes = get_nodes(node_db, nGroups-1, False, True)
    manager.build_erme(nodes)
    
    # The ResponseServer actually manages all the response sources.  The 
    # way we'll do this (at least for now) is to update *all* responses for 
    # a given keff, and then copy the resulting responses to some interim 
    # storage container (HDF5 or otherwise).  From the interim storage, you
    # can build the HDF5 file in whatever way you need.  The only thing you 
    # need to know is how to index into a response array.  Consider the 
    # fission response, which is just a vector of length (eo+1)(so+1)(ao+1)...
    # If you need only zeroth order in energy, you need to know which elements
    # are zeroth order in energy.  

    server = manager.get_server()  
    if kstudy == False :
        if use_angular_moments == False :
            dbname = dir_name+'/databases/keffdb'+str(case)+'.h5'
        elif moment_order == 0 :
            dbname = dir_name+'/databases/keffdb'+str(case)+'partial.h5'
        else :
            dbname = dir_name+'/databases/keffdb'+str(case)+'order'+str(moment_order)+'.h5'
    else :
        dbname = dir_name+'/databases/keffdb'+str(case)+'nk'+str(num_keff)+'kd'+str(int(kdel*100))+'.h5'
    
    if Comm.rank() == 0 :
        f = h5py.File(dbname, 'w')
    for n in range(nodes.number_unique_local_nodes()) :
        if nodes.node(n).is_fuel() == True :
            keff = np.linspace(min_keff, max_keff, num_keff)
        else :
            keff = [1.]
        if Comm.rank() == 0 :
            g = 0.
            g = f.create_group('/' + 'node' + str(n))
            g.attrs['number_keffs'] = len(keff)
            nSurface = nodes.node(n).number_surfaces()
            nPins = nodes.node(n).number_pins()
            g.attrs['number_surfaces'] = nSurface
            g.attrs['number_pins'] = nPins
            eo = nodes.node(n).energy_order(0)
            po = nodes.node(n).polar_order(0)
            ao = nodes.node(n).azimuthal_order(0)
            so = nodes.node(n).spatial_order(0,0)
            RO = (eo+1)*(po+1)*(ao+1)*(so+1) * nSurface
            g.attrs['response_size'] = RO
            g.attrs['scheme'] = 1
            g.attrs['ao'] = ao
            g.attrs['so'] = so
            g.attrs['eo'] = eo
            g.attrs['po'] = po
            A = np.zeros((len(keff), RO))
            F = np.zeros((len(keff), RO))
            R = np.zeros((len(keff), RO, RO))
            L = np.zeros((len(keff), nSurface, RO))
            NP = np.zeros((len(keff), RO))
            PP = np.zeros((len(keff), nPins, RO))
        for k in range(len(keff)) :
            if Comm.rank() == 0 : print 'n=', n, ' k=', k 
            server.update(keff[k])
            if Comm.rank() == 0 :
                #server.response(n).display()
                for i in range(RO) :
                    A[k][i] = server.response(n).absorption_response(i)
                    F[k][i] = server.response(n).fission_response(i)
                    NP[k][i] = server.response(n).nodal_power(i)
                    for j in range(nPins) :
                        PP[k][j][i] = server.response(n).pin_power(j, i)
                    for j in range(RO) :
                        R[k][i][j] = server.response(n).boundary_response(i,j)
                    for j in range(nSurface) :
                        L[k][j][i] = server.response(n).leakage_response(j,i)
        if Comm.rank() == 0 :
            g['A'] = A
            g['F'] = F
            g['L'] = L
            g['R'] = R
            g['nodal_power'] = NP
            g['pin_power'] = PP
            g['keffs'] = keff
    if Comm.rank() == 0 :
        f.close()
        print 'database ', case, 'completed'
        return   
  
    
def dbslice(case, eo, kstudy=False, num_keff=8, kdel=0.15) :
    if kstudy == False :
        if use_angular_moments == False :
            dbname = dir_name+'/databases/keffdb'+str(case)+'.h5'
            dName = dir_name+'/databases/keffdbslice'+str(case)+'.h5'
        elif moment_order == 0 :
            dbname = dir_name+'/databases/keffdb'+str(case)+'partial.h5'
            dName = dir_name+'/databases/keffdbslice'+str(case)+'partial.h5'
        else :
            dbname = dir_name+'/databases/keffdb'+str(case)+'order'+str(moment_order)+'.h5'
            dName = dir_name+'/databases/keffdbslice'+str(case)+'order'+str(moment_order)+'.h5'
    else :
        dbname = dir_name+'/databases/keffdb'+str(case)+'nk'+str(num_keff)+'kd'+str(int(kdel*100))+'.h5'
        dName = dir_name+'/databases/keffdbslice'+str(case)+'nk'+str(num_keff)+'kd'+str(int(kdel*100))+'.h5'
        
    f = h5py.File(dbname, 'r')
    g = h5py.File(dName, 'w')
    for name in f :
        node = f['/' + name]
        so = node.attrs['so']
        po = node.attrs['po']
        ao = node.attrs['ao']
        ro = node.attrs['response_size']
        nK = node.attrs['number_keffs']
        nS = node.attrs['number_surfaces']
        nPins = node.attrs['number_pins']
        scheme = node.attrs['scheme']
        new_ro = (eo+1)*(po+1)*(ao+1)*(so+1) * nS
        A = node['A'][:]
        F = node['F'][:]
        R = node['R'][:]
        L = node['L'][:]
        NP = node['nodal_power'][:]
        PP = node['pin_power'][:]
        keffs = node['keffs'][:]
        AA = np.zeros((nK, new_ro))
        FF = np.zeros((nK, new_ro))
        RR = np.zeros((nK, new_ro, new_ro))
        LL = np.zeros((nK, nS, new_ro))
        NPNP = np.zeros((nK, new_ro))
        PPPP = np.zeros((nK, nPins, new_ro))
        
        h = 0
        h = g.create_group('/' + name)
        h.attrs['response_size'] = new_ro
        h.attrs['number_keffs'] = nK
        h.attrs['number_surfaces'] = nS
        h.attrs['number_pins'] = nPins
        h.attrs['scheme'] = scheme
        h.attrs['so'] = so
        h.attrs['eo'] = eo
        h.attrs['ao'] = ao
        h.attrs['po'] = po
        print name, 'ro =', new_ro, 'of', ro
        for s in range(nS) :
            for r in range(new_ro/nS) :
                AA[:,r+new_ro/nS*s]  = A[:,r+ro/nS*s]
                FF[:,r+new_ro/nS*s]  = F[:,r+ro/nS*s]
                NPNP[:,r+new_ro/nS*s]  = NP[:,r+ro/nS*s]
                PPPP[:,:,r+new_ro/nS*s]  = PP[:,:,r+ro/nS*s]
                LL[:,:,r+new_ro/nS*s]  = L[:,:,r+ro/nS*s]
                for ss in range(nS) :
                    for rr in range(new_ro/nS) :
                        RR[:,r+new_ro/nS*s,rr+new_ro/nS*ss]  = R[:,r+ro/nS*s,rr+ro/nS*ss]
        h['A'] = AA[...]
        h['F'] = FF[...]
        h['nodal_power'] = NPNP[...]
        h['pin_power'] = PPPP[...]
        h['R'] = RR[...]
        h['L'] = LL[...]  
        h['keffs'] = keffs[...]             
    f.close()
    g.close()

    return dName
    
def run_ref_rf() :
    """ Produce a reference response matrix solution using a full
        multigroup energy treatment.
    """
    if test_problem == 'c5g7':
        keff = 1.0675061198499389
    else :
        refdata = pickle.load(open(dir_name+'/refdata/reference_data.p', "rb"))
        keff = refdata['keff']
        
    node_db = get_db(0)
    nodes = get_nodes(node_db, nGroups-1, False, False)
    db = InputDB.Create()
    if test_problem == 'c5g7':
        db.put_int("dimension",             2)
        db.put_dbl("erme_tolerance",        1e-8)
        db.put_dbl("erme_inner_tolerance",  1e-8)
    else:
        db.put_int("dimension",             1)
        db.put_dbl("erme_tolerance",        1e-12)
        db.put_dbl("erme_inner_tolerance",  1e-12)
    db.put_dbl("erme_initial_keff",         keff)
    db.put_int("erme_maximum_iterations",   20)
    db.put_int("comm_local_groups",         1)
    manager = ManagerERME.Create(sys.argv)
    manager.build_comm(db)
    manager.build_erme(nodes)
    manager.solve()
    
    postprocess = PostProcess(manager)
    data = {}
    if test_problem == 'c5g7':
        data['ass_p'] = np.asarray(postprocess.nodal_power(4.0))
        pin_power   = np.asarray(postprocess.pin_power(1056.) )
        pin_power = pin_power * 1056.0 / sum(sum(pin_power))
        data['pin_p'] = pin_power
        #data['2to1_pin_p'] = two2one(pin_power)
        data['keff']  = manager.get_keff()
    elif test_problem == 'bwrcore':
        data['power'] = np.asarray(postprocess.nodal_fission_density(1.0))
        data['pin_p'] = np.asarray(postprocess.pin_power(1.0))
        data['keff']  = manager.get_keff()
    else:
        data['power'] = np.asarray(postprocess.nodal_fission_density(1.0))
        data['keff']  = manager.get_keff()
    pickle.dump(data, open(dir_name+'/refdata/reference_rf_data.p', 'wb'))
        
def run(case, kstudy=False, num_keff=8, kdel=0.15) :
    """ Run study of various energy bases.
    """
    case = int(case)
    usedb = 1
    refdata = pickle.load(open(dir_name+'/refdata/reference_data.p', "rb"))
    keff = refdata['keff']

    db = InputDB.Create()
    db.put_int("dimension",                 1)
    db.put_dbl("erme_initial_keff",         keff)
    db.put_int("erme_maximum_iterations",   20)
    db.put_dbl("erme_tolerance",            1e-12)
    db.put_dbl("erme_inner_tolerance",      1e-12)
    db.put_int("comm_local_groups",         1)
    if test_problem == 'c5g7':
        db.put_int("dimension",             2)
        db.put_dbl("erme_tolerance",        1e-8)
        db.put_dbl("erme_inner_tolerance",  1e-8)
    
    # Load the zeroth order for each case
    refdata = pickle.load(open(dir_name+'/refdata/reference_data.p', "rb"))
    zeroth = []
    zeroth.append(refdata['phi_full'])
    if test_problem == '10-pin' :
        zeroth.append(refdata['phi_uo2'])
        zeroth.append(refdata['phi_mox'])

    # Load reference rf data
    rfdata =  pickle.load(open(dir_name+'/refdata/reference_rf_data.p', "rb"))
    if test_problem == 'c5g7':
        ref_power = rfdata['ass_p']
        ref_pin_power = rfdata['pin_p']
    else :
        ref_power = rfdata['power']
    ref_keff  = rfdata['keff']
    
    # Loop over energy orders
    data = {}
                       
    node_db = get_db(case)        
    for c in range(case, case+1) :    
        # Setup cases
        data[cases[c]] = {}
        if case > nDLPcases - 1 :
            node_db.put_str("basis_e_type", "UserBasis")
        elif case == nDLPcases-1 :
            node_db.put_str("basis_e_type", "dlp")
        else :
            node_db.put_str("basis_e_type", "trans")
        if case < 3 :
            node_db.put_int("basis_e_transformed_option", 1)
        else :
            node_db.put_int("basis_e_transformed_option", 0)
        if case < nDLPcases - 1 :
            node_db.put_vec_dbl("basis_e_zeroth", zeroth[case % 3])       
        # Loop over orders
        data[cases[c]]['err_p'] = np.zeros(nGroups)
        data[cases[c]]['err_k'] = np.zeros(nGroups)
        data[cases[c]]['power'] = []
        data[cases[c]]['keff']  = np.zeros(nGroups)   
        if test_problem == '10-pin' :
            N = nGroups
        elif test_problem == 'bwrcore':
            data[cases[c]]['pin_p'] = []   
            N = nGroups
        elif test_problem == 'c5g7':
            data[cases[c]]['pin_p'] = []  
            N = 10
        for o in range(N) :
            print " case = ", c, " order =", o
            if usedb == 1 :
                if kstudy == False :
                    dName = dbslice(case, o)
                else :
                    dName = dbslice(case, o, kstudy, num_keff, kdel)
                db.put_str("response_db_name",               dName)
                db.put_int("response_db_order",              3)
                db.put_str("response_db_interpolation_type", 'i') # 'n', 'i', 'e', 's', 'ie', 'is', 'es', 'ies'
                nodes = get_nodes(node_db, o, True, False)
            else :
                nodes = get_nodes(node_db, o, False, False)
            if test_problem != '10-pin' :
                tmp = process_erme_c5g7(db, nodes, ref_power, ref_keff, ref_pin_power)
                data[cases[c]]['pin_p'].append(tmp['pin_p'])
            else :
                tmp = process_erme(db, nodes, ref_power, ref_keff)  
            data[cases[c]]['err_p'][o] = tmp['err_p']
            data[cases[c]]['err_k'][o] = tmp['err_k']
            data[cases[c]]['power'].append(tmp['power'])
            data[cases[c]]['keff'][o] = tmp['keff']
            if usedb == 1 :
                if os.path.isfile(dName) :
                    os.remove(dName)
        if kstudy == False :
            if case < nDLPcases :
                pickle.dump(data, open(dir_name+'/rf_data/dlp/rf_data'+str(case)+'.p', 'wb'))
            elif use_angular_moments == False :
                pickle.dump(data, open(dir_name+'/rf_data/phi/rf_data'+str(case)+'.p', 'wb'))
            elif moment_order == 0 :
                pickle.dump(data, open(dir_name+'/rf_data/partial/rf_data'+str(case)+'.p', 'wb'))
            else :
                pickle.dump(data, open(dir_name+'/rf_data/angular/rf_data'+str(case)+'order'+str(moment_order)+'.p', 'wb'))
        else :
            pickle.dump(data, open(dir_name+'/rf_data/dlp/rf_data'+str(case)+'nk'+str(num_keff)+'kd'+str(int(kdel*100))+'.p', 'wb'))
    return
      
def process_erme(db, nodes, ref_power, ref_keff) :
    manager = ManagerERME.Create(sys.argv)
    manager.build_comm(db)
    manager.build_erme(nodes)
    manager.solve()
    postprocess = PostProcess(manager)
    if Comm.rank() == 0 :
        tmp = {}
        tmp['power'] = np.asarray(postprocess.nodal_fission_density(1.0))
        tmp['keff']  = manager.get_keff()
        tmp['err_p'] = np.linalg.norm(ref_power-tmp['power']) 
        tmp['err_k'] = np.abs(ref_keff-tmp['keff'])/ref_keff
        return tmp

def process_erme_c5g7(db, nodes, ref_power, ref_keff, ref_pinp) :
    manager = ManagerERME.Create(sys.argv)
    manager.build_comm(db)
    manager.build_erme(nodes)
    manager.solve()
    postprocess = PostProcess(manager)
    if Comm.rank() == 0 :
        tmp = {}
        tmp['power'] = np.asarray(postprocess.nodal_power(4.0))
        print tmp['power']
        pin_power = np.asarray(postprocess.pin_power(1056.))
        tmp['pin_p'] = pin_power * 1056.0 / sum(sum(pin_power))
        print tmp['pin_p']
        tmp['keff']  = manager.get_keff()
        tmp['err_p'] = np.linalg.norm(ref_power-tmp['power']) 
        tmp['err_k'] = np.abs(ref_keff-tmp['keff'])/ref_keff
        return tmp



