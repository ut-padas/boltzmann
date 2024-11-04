## Recombination cross-section computation. 

import scipy.optimize
import basis
import cross_section
import scipy.constants
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.optimize

CRS_DATA   = cross_section.read_cross_section_data("lxcat_data/eAr_crs.6sp_Tg_0.5eV")
CRS_SPS    = cross_section.read_available_species("lxcat_data/eAr_crs.6sp_Tg_0.5eV")

EQL_NAME   = {
                "E + Ar -> E + E + Ar(+)":     "Ground",
                "E + Ar*(1) -> E + E + Ar(+)": "Metastable",
                "E + Ar*(2) -> E + E + Ar(+)": "Resonant",
                "E + Ar*(3) -> E + E + Ar(+)": "4p",}

def read_eqr_consts(eqc_fname, cs_dict):
    kB         = scipy.constants.Boltzmann
    me         = scipy.constants.electron_mass
    qe         = scipy.constants.electron_volt
    c_gamma    = np.sqrt(2 * qe / me)
    
    ev_to_K    = (qe/kB)
    eqc_name   = EQL_NAME
    
    ff         = h5py.File(eqc_fname, "r")
    eqc        = dict()

    for key, cs_metadata in cs_dict.items():
        if cs_metadata["type"] == "IONIZATION":
            eqc_data       = ff[eqc_name[key]][()]
            eqc_data[:, 0] = eqc_data[:, 0]/ev_to_K
            eqc[key]       = (scipy.interpolate.interp1d(eqc_data[:, 0], eqc_data[:, 1], kind="linear", bounds_error=True), eqc_data)
    
    return eqc

#EQL_CONSTS = read_eqr_consts("lxcat_data/equilibrium_constants_eAr6sp.h5", CRS_DATA)
EQL_CONSTS  = read_eqr_consts("lxcat_data/equilibrium_constants_eAr6sp_ext.h5", CRS_DATA)


def spline_basis(num_p, sp_order, v_domain):
    return basis.BSpline(v_domain, sp_order, num_p, sig_pts=None, knots_vec=None, dg_splines=False, verbose=True, extend_domain=False)

def forward_operators(num_p, sp_order, ev_domain, Te_d, q_per_knot, energy_threshold_ev):
    """
    Te_d : Te data points
    
    outputs
    Q_mat  : Quadrature matrix
    Vr_mat : B-spline Vandermonde matrix (eval at Q_mat points)
    Lp_mat : Laplace on the B-splines (eval at Q_mat_points)
    
    """
    
    
    num_d      = len(Te_d)
    kB         = scipy.constants.Boltzmann
    me         = scipy.constants.electron_mass
    qe         = scipy.constants.electron_volt
    c_gamma    = np.sqrt(2 * qe / me)
    
    ev_to_K    = (qe/kB)
    vth_e      = lambda Te : np.sqrt(Te) * c_gamma
    vth        = vth_e(np.max(Te_d))
    
    v_domain   = (np.sqrt(ev_domain[0]) * c_gamma / vth, np.sqrt(ev_domain[1]) * c_gamma / vth)
    bb         = spline_basis(num_p, sp_order, v_domain)
    
    gx, gw     = bb.Gauss_Pn(bb._num_knot_intervals * q_per_knot)
    vth_d      = vth_e(Te_d)
    
    Vr_mat     = np.array([bb.Pn(idx)(gx, l=0) for idx in range(num_p)]).reshape((num_p, len(gx)))
    #Lp_mat     = np.array([bb.derivative(idx, 2)(gx) for idx in range(num_p)]).reshape((num_p, len(gx)))
    sp_order   = bb._sp_order
    
    Lp_mat     = np.zeros((num_p, num_p))
    for i in range(num_p):
        for j in range(max(0, i - (sp_order+3) ), min(num_p, i + (sp_order+3))):
            Lp_mat[i,j] = np.dot(gw, gx**2 * bb.derivative(i, 1)(gx) * bb.derivative(j, 1)(gx))
    
    f0         = 4* np.pi * np.array([(1 / ((vth_d[i] * np.sqrt(np.pi))**3)) * np.exp(-(gx * vth/vth_d[i]) ** 2) for i in range(num_d)]).reshape((num_d, len(gx)))
    mass       = (vth **3 * np.dot(gx**2 * f0 , gw)).reshape((-1, 1))
    f0_n       = f0/mass
    avg_energy = (0.5 * me ) * vth ** 5 * np.dot(gx**4 * f0_n , gw) 
    temp       = (2.0/3.0/kB) * avg_energy / (qe/kB)
    
    #print("computed Te : \n", temp, " data points : \n", Te_d, "\n", end="")
    print("rel error (Te): %.6E"%(np.linalg.norm(temp - Te_d)/np.linalg.norm(Te_d)))
    
    vi         = (energy_threshold_ev**0.5) * c_gamma / vth
    Q_mat      = np.array([ gw * vth**7 * np.sqrt(2 * gx**2 + vi**2) * 2 * gx**5 * f0[i]**2 for i in range(num_d) ])
    
    return bb, Q_mat, Vr_mat.T, Lp_mat.T, vth, gx, gw

def newton_solver(x, residual, jacobian, atol, rtol, iter_max, xp=np):
    x0       = xp.copy(x)
    jac      = jacobian(x0)
    jac_inv  = xp.linalg.inv(jac)
    
    ns_info  = dict()
    alpha    = 1.0e0
    x        = x0
    count    = 0
    r0       = residual(x)
    rr       = xp.copy(r0)
    norm_rr  = norm_r0 = xp.linalg.norm(r0)
    converged = ((norm_rr/norm_r0 < rtol) or (norm_rr < atol))
    
    while ( alpha > 1e-6 ):
        count = 0
        x     = x0
        rr    = r0
        
        while( (not converged) and (count < iter_max)):
            xk       = x  - alpha * xp.dot(jac_inv, rr).reshape(x.shape)
            rk       = residual(xk)
            norm_rk  = xp.linalg.norm(rk)
        
            x         = xk
            rr        = rk
            norm_rr   = norm_rk
            count    += 1
            converged = ((norm_rr/norm_r0 < rtol) or (norm_rr < atol))
        
            #print("alpha = %.6E iter =%d norm_rr = %.6E norm_rr/norm_r0 = %.6E "%(alpha, count, norm_rr, norm_rr/norm_r0))
            if ((np.isnan(norm_rr)) or (norm_rr > norm_r0 * 1e2)):
                break
        
        
        if (not converged):
            alpha = 0.25 * alpha
        else:
            break

    if (not converged):
        print("Newton solver failed !!!: {0:d}: ||res|| = {1:.14e}, ||res||/||res0|| = {2:.14e}".format(count, norm_rr, norm_rr/norm_r0), "alpha ", alpha, xp.linalg.norm(xk))
        ns_info["status"] = converged
        ns_info["x"]      = x
        ns_info["atol"]   = norm_rr
        ns_info["rtol"]   = norm_rr/norm_r0
        ns_info["alpha"]  = alpha
        ns_info["iter"]   = count
        return ns_info
    
    ns_info["status"] = converged
    ns_info["x"]      = x
    ns_info["atol"]   = norm_rr
    ns_info["rtol"]   = norm_rr/norm_r0
    ns_info["alpha"]  = alpha
    ns_info["iter"]   = count
    return ns_info
    
def nllsqr(x0, residual, jacobian, atol, rtol, iter_max, xp=np):
    
    iter          = 0
    r0            = residual(x0)
    norm_r0       = xp.linalg.norm(r0)
    abs_e , rel_e = norm_r0, 1
    converged     = (rel_e < rtol or abs_e < atol)
    x1            = xp.copy(x0)
    x2            = xp.copy(x0) + 1
    
    while((iter < iter_max) and not converged):
        
        if (iter > 0):
            x2 = np.copy(x1)
        
        alpha      = 1e0
        rr         = residual(x1)
        norm_rr    = np.linalg.norm(rr)
        
        # abs_e      = norm_rr
        # rel_e      = norm_rr/norm_r0
        
        
        Jmat       = jacobian(x1)
        JTJ        = xp.dot(Jmat.T, Jmat)
        JTJ_inv    = xp.linalg.pinv(JTJ, rcond=1e-12)
    
        dx         = -np.dot(JTJ_inv, np.dot(Jmat.T, rr))
        
        while (alpha> 1e-8):
            if (np.linalg.norm(residual(x1 + alpha * dx)) > norm_rr):
                alpha = alpha * 0.1
            else:
                break
        
            if (alpha < 1e-8):
                break
        
        x1         = x1 + dx * alpha
        
        rr         = residual(x1)
        norm_rr    = np.linalg.norm(rr)
        abs_e      = norm_rr
        rel_e      = norm_rr/norm_r0
        rr_x       = np.linalg.norm(x1-x2)/max(1, np.linalg.norm(x1))
        converged  = rel_e < rtol or abs_e < atol or rr_x < rtol
        
        if (iter%100 == 0):
            print("[NLLS] iter = %06d abs. error = %.10E rel. error = %.10E ||x1-x2||/||x2|| = %.10E "%(iter, abs_e, rel_e, rr_x))
        
        if (converged):
            print("[NLLS] iter = %06d abs. error = %.10E rel. error = %.10E ||x1-x2||/||x2|| = %.10E "%(iter, abs_e, rel_e, rr_x))
            break
        
        
        iter+=1
        
    return x1
        

def recomb_crs_fit(Te_all, Te_obs, Nr, sp_order, q_per_knot, cs_dict, eql_const, theta, fname):
    
    # probably want this to be passed as an argument
    eqc_name = {
                "E + Ar -> E + E + Ar(+)":     "Ground",
                "E + Ar*(1) -> E + E + Ar(+)": "Metastable",
                "E + Ar*(2) -> E + E + Ar(+)": "Resonant",
                "E + Ar*(3) -> E + E + Ar(+)": "4p",
                }
    
    kB       = scipy.constants.Boltzmann
    me       = scipy.constants.electron_mass
    qe       = scipy.constants.electron_volt
    ev_to_K  = (qe/kB)
    c_gamma  = np.sqrt(2 * qe / me)
    VTH      = lambda Te : np.sqrt(Te) * c_gamma
    
    cs_data  = list()
    eqc_data = list()
    process_data = list()
    for key, cs_metadata  in cs_dict.items():
        if cs_metadata["type"] == "IONIZATION":
            process_data.append(key)
            cs_data.append(scipy.interpolate.interp1d(cs_metadata["energy"], cs_metadata["cross section"], kind="linear", bounds_error=False, fill_value=0.0))
            eqc_data.append(eql_const[key][0])
    
    
    rf = cross_section.compute_mw_reaction_rates(Te_obs, cs_data)
    a1 = np.zeros((len(Te_obs), len(cs_data)))
    
    for i in range(len(cs_data)):
        a1[:, i] = eqc_data[i](Te_obs)
    
    rb      = rf / a1 / scipy.constants.Avogadro
    
    rf_all = cross_section.compute_mw_reaction_rates(Te_all, cs_data)
    a1     = np.zeros((len(Te_all), len(cs_data)))
    
    for i in range(len(cs_data)):
        a1[:, i] = eqc_data[i](Te_all)
    
    rb_all = rf_all / a1 / scipy.constants.Avogadro
    
    plt.figure(figsize=(8, 4), dpi=200)
    plt.subplot(1, 2, 1)
    for i in range(len(process_data)):
        plt.semilogy(Te_all,rf_all[:, i], '-', markersize=2, label=r"%s"%(process_data[i]))

    plt.subplot(1, 2, 2)
    for i in range(len(process_data)):
        plt.semilogy(Te_all,rb_all[:, i], '-', markersize=2, label=r"reverse (%s)"%(process_data[i]))
    
    plt.subplot(1, 2, 1)
    plt.xlabel(r"temperature (eV)")
    plt.ylabel(r"reaction rate ($m^3s^{-1}$)")
    plt.title(r"$e + Ar^{*}/Ar \rightarrow e + e + Ar^{+}$")
    plt.grid(visible=True)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title(r"$e + e + Ar^{+} \rightarrow e + Ar^{*}/Ar$")
    plt.xlabel(r"temperature (eV)")
    plt.ylabel(r"reaction rate ($m^6s^{-1}$)")
    plt.grid(visible=True)
    plt.tight_layout()
    #plt.show()
    plt.savefig("lxcat_data/forward_backward_rates.png")
    plt.close()
    
    rb_norms                      = np.array([np.linalg.norm(rb[:, rs_idx]) for rs_idx in range(len(process_data))])
    rb_obs                        = rb / rb_norms
    
    
    rb_fit = list()
    Br_fit = list()
    
    for rs_idx in range(0, len(process_data)):
        
        process_key                       = process_data[rs_idx]
        ev_domain                         = (0, 8 * Te_obs[-1])
        bb, Qmat, Vq, Lp, vth_b, gx, gw   = forward_operators(Nr, sp_order, ev_domain, Te_obs, q_per_knot, cs_dict[process_key]["threshold"])
        QTQ                               = np.dot(Qmat.T, Qmat)
        
        #print(np.linalg.norm(Qmat))
        rg_theta                      = theta #* np.linalg.norm(Qmat)
        rg_lip_const                  = 10
        #rg_jac                       = np.kron(np.ones(Te_obs.shape[0]), np.dot(gw, Lp)).reshape((len(Te_obs), Nr))
    
        def residual(x):
            y       = np.exp(np.dot(Vq, x))
            return  (np.dot(Qmat, y) - rb_obs[:, rs_idx]) + rg_theta * np.dot(x, np.dot(Lp, x))
            
    
        def jacobian(x):
            jd = np.dot(Qmat, np.exp(np.dot(Vq, x)).reshape((-1, 1)) * Vq)
            jr = np.kron(np.ones(Te_obs.shape[0]), 2*np.dot(Lp, x)).reshape((len(Te_obs), Nr))
            return  jd +  rg_theta * jr
            
        x0 = np.ones(Nr) 
        x1 = nllsqr(x0, residual, jacobian, atol=1e-20, rtol=1e-14, iter_max=300, xp=np)
    
    
        assert np.max(Te_all) == np.max(Te_obs)
        bb, qmat_full, Vq, Lp, vth, gx, gw  = forward_operators(Nr, sp_order, ev_domain, Te_all, q_per_knot, cs_dict[process_key]["threshold"])
        
        ev_r                                = (gx * vth/c_gamma)**2
        Br_ev                               = np.exp(np.dot(Vq, x1)) * rb_norms[rs_idx] 
        rb_pred                             = np.dot(qmat_full, Br_ev)
        
        rb_fit.append(rb_pred)
        Br_fit.append(Br_ev)

        # if rs_idx == 0:
        #     plt.figure(figsize=(10, 6), dpi=200)
            
        # plt.subplot(1, 2, 1)
        # plt.semilogy(Te_all , rb_all[:, rs_idx]                         ,      '-'  , label=r"$k_b = \lambda k_f$ (%s)"%(eqc_name[process_data[rs_idx]])        , markersize=2)
        # plt.semilogy(Te_all , rb_pred                                   ,      'o--' , label=r"nl-lsqr-fit (%s)"%((eqc_name[process_data[rs_idx]]))             , markersize=2)
        # plt.legend()
        # plt.grid(visible=True)
        # plt.ylabel(r"rate coefficient ($m^6s^{-1}$)")
        # plt.xlabel(r"temperature (eV)")
        
        # plt.subplot(1, 2, 2)
        # plt.loglog(ev_r, Br_ev, label=r"recomb (%s)"%(process_data[rs_idx]))
        # plt.legend()
        # plt.grid(visible=True)
        # plt.tight_layout()
        
        # plt.ylabel(r"$B_R$")
        # plt.xlabel(r"energy (eV)")
        # plt.tight_layout()
    
    #plt.show()
    #plt.savefig("lxcat_data/recomb_kernel.png")
    #plt.close()
    
    rb_fit= np.array(rb_fit).T
    
    ff = h5py.File("lxcat_data/%s.h5"%(fname), 'w')
    ff.create_dataset(name = "energy(eV)"   , data = (gx * vth/ c_gamma) ** 2)
    ff.create_dataset(name = "recomb_kernel", data = np.array(Br_fit).T)
    
    ff.create_dataset(name = "Te_obs[eV]"       , data=Te_obs)
    ff.create_dataset(name = "kb_obs[m^6s^-1]"  , data=rb_obs * rb_norms)
    
    ff.create_dataset(name = "Te_full[eV]"      , data=Te_all)
    ff.create_dataset(name = "kb_data[m^6s^-1]" , data=rb_all)
    ff.create_dataset(name = "kb_fit[m^6s^-1]"  , data=rb_fit)
    ff.create_dataset(name = "processes"        , data = np.array(process_data, dtype='S'))
    ff.close()
    
    
    
    

nr            = 64
sp_order      = 3
qpts_per_knot = 80

# plt.figure(figsize=(6, 4), dpi=200)
# for w in range(5, 21, 5):
#     Te_obs   = list(EQL_CONSTS.items())[0][1][1][0::w, 0]
#     bb       = spline_basis(nr, sp_order, (0, 5))
#     qmat     = qmat_bsplines(Te_obs, bb, qpts_per_knot)
#     u, s, vh = np.linalg.svd(qmat, full_matrices=False)
#     plt.semilogy(s, 'o-',label=r"data points = %d"%(len(Te_obs)), markersize=2)
    
# plt.xlabel(r"index")
# plt.ylabel(r"singular value")
# plt.grid(visible=True)
# plt.legend()
# plt.savefig("qmat_svd.png")
# plt.close()

Te_all = list(EQL_CONSTS.items())[0][1][1][:, 0]    
Te_obs = Te_all[np.array(range(0, len(Te_all), 64))]
Te_obs = np.append(Te_obs, np.array([Te_all[-1]]))

theta_vals = [1e-6, 1e-8, 1e-10, 0.0]
for rt in theta_vals:
    recomb_crs_fit(Te_all, Te_obs, nr, sp_order, qpts_per_knot, CRS_DATA, EQL_CONSTS, theta=rt, fname="rb_train_%d_rg_%.2E"%(len(Te_obs), rt))

for rs_idx in range(1, 4):
    plt.figure(figsize=(10, 6), dpi=200)
    for rt_idx, rt in enumerate(theta_vals):
        fname = "rb_train_%d_rg_%.2E"%(len(Te_obs), rt)
        with h5py.File("lxcat_data/%s.h5"%(fname), 'r') as ff:
            Te_all = ff["Te_full[eV]"][()]
            kb_all = ff["kb_data[m^6s^-1]"][()]
            kb_fit = ff["kb_fit[m^6s^-1]"][()]
            ev     = ff["energy(eV)"][()]
            Br     = ff["recomb_kernel"][()]
            prs    = ff["processes"][()]
            ff.close()
            
            plt.subplot(1, 2, 1)
            if(rt_idx == 0):
                plt.semilogy(Te_all, kb_all[:, 1], label = r"$k_b=\lambda k_f$")
            
            plt.semilogy(Te_all, kb_fit[:, rs_idx], label=r"$\theta$=%.2E"%(rt))
            plt.xlabel(r"temperature [eV]")
            plt.ylabel(r"reverse rate [$m^6s^{-1}$]")
            plt.grid(visible=True)
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.loglog(ev, Br[:, rs_idx], label=r"$\theta$=%.2E"%(rt))
            plt.xlabel(r"energy ($\varepsilon$) [eV]")
            plt.ylabel(r"$B_{recomb}(\varepsilon)$")
            plt.grid(visible=True)
            plt.legend()
    
    plt.suptitle(r"%s"%(prs[rs_idx]))
    plt.tight_layout()
    #plt.show()
    plt.savefig("lxcat_data/crs_recomb_%02d.png"%(rs_idx))
    plt.close()
        
    

    
    
    
    
    
    
    
    