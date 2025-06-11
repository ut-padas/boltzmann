"""
Module to handle different closure models. 
"""
import numpy as np
import cross_section
import spec_spherical as sp
import basis
import utils as bte_utils
import plot_scripts.plot_utils as plot_utils
import scipy.constants
import matplotlib.pyplot as plt

def compute_radial_components(spec_sp, ff, ev: np.array, Te0):
    
    t_pts    = ff.shape[0]
    
    ff_lm    = ff
    c_gamma  = np.sqrt(2 * (scipy.constants.elementary_charge/ scipy.constants.electron_mass))
    vth      = np.sqrt(Te0) * c_gamma
    
    vr       = np.sqrt(ev) * c_gamma / vth
    num_p    = spec_sp._p +1 
    num_sh   = len(spec_sp._sph_harm_lm)
    n_pts    = ff.shape[2]
    
    output   = np.zeros((t_pts, n_pts, num_sh, len(vr)))
    Vqr      = spec_sp.Vq_r(vr,0,1)
    
    mm_fac   = np.sqrt(4 * np.pi) 
    
    scale    = np.array([1 / mm_fac  for idx in range(t_pts)])
    ff_lm_n  = np.array([ff_lm[idx]/scale[idx] for idx in range(t_pts)])
    
    for idx in range(t_pts):
        ff_lm_T  = ff_lm_n[idx].T
        for l_idx, lm in enumerate(spec_sp._sph_harm_lm):
            output[idx, :, l_idx, :] = np.dot(ff_lm_T[:,l_idx::num_sh], Vqr)


    if spec_sp.get_radial_basis_type() == basis.BasisType.LAGUERRE:
        output *= (1/np.pi**1.5) * np.exp(-vr**2)
        
    return output

def quadrature_grid(spec_sp : sp.SpectralExpansionSpherical, num_vr, num_vt, num_vp, coord="spherical"):
    [glx,glw]    = basis.Legendre().Gauss_Pn(num_vt)
    vt_q, vt_w   = np.arccos(glx), glw
    
    [glx,glw]    = basis.Legendre().Gauss_Pn(num_vp)
    vp_q, vp_w   = np.pi * glx + np.pi , np.pi * glw

    vr_q, vr_w   = spec_sp._basis_p.Gauss_Pn(num_vr)

    vg           = np.meshgrid(vr_q, vt_q, vp_q, indexing="ij")

    if (coord == "spherical"):
        return vg, (vr_w, vt_w, vp_w)
    elif (coord == "cartesian"):
        vg = (vg[0] * np.sin(vg[1]) * np.cos(vg[2]), vg[0] * np.sin(vg[1]) * np.sin(vg[2]), vg[0] * np.cos(vg[1]))
        return vg, (vr_w, vt_w, vp_w) 
    else:
        raise NotImplementedError
    
def max_entropy_reconstruction(spec_sp:sp.SpectralExpansionSpherical, x0, num_vr, num_vt, num_vp, m_vec : np.array, m_funcs : list, xp:np, rtol, atol, iter_max):
    num_m = m_vec.shape[0]
    num_x = m_vec.shape[1]

    ###
    # We seek solutions of the form R = (<exp(\lambda_k \phi_k) \phi> - m)
    # find lambda values that minimizes the R. i.e., maximum entropy approximation that best
    # match the moments
    ###

    x                             = xp.copy(x0)
    vg, (vr_w, vt_w, vp_w)        = quadrature_grid(spec_sp, num_vr, num_vt, num_vp)
    mf                            = np.zeros(tuple([num_m]) + vg[0].shape)

    for i in range(num_m):
       mf[i]  = m_funcs[i](vg[0], vg[1], vg[2])
    
    if spec_sp.get_radial_basis_type() == basis.BasisType.SPLINES:
        wx    = vg[0]**2 
        beta  = 0.0 
        nfac  = 2.0

    elif spec_sp.get_radial_basis_type() == basis.BasisType.LAGUERRE:
        wx   = np.ones_like(vg[0])
        beta = 1.0
        nfac = 4 * xp.pi**1.5 
    else:
        raise NotImplementedError

    def residual(x):
        xmf = xp.einsum("ix,iklm->xklm", x, mf)
        y   = 2 * xp.exp(xmf) * (xp.exp(beta * vg[0]**2) * wx)[xp.newaxis, :, :, :]
        y   = xp.einsum("iklm,xklm->ixklm", mf, y)
        y   = xp.einsum("ixklm,k,l,m->ix", y, vr_w, vt_w, vp_w) - m_vec
        return y

    def jacobian(x):
        xmf = xp.einsum("ix,iklm->xklm", x, mf)
        y   = 2 * xp.exp(xmf) * (xp.exp(beta * vg[0]**2) * wx)[xp.newaxis, :, :, :]
        y   = xp.einsum("iklm,jklm,xklm->ijxklm", mf, mf, y)
        y   = xp.einsum("ijxklm,k,l,m->ijx", y, vr_w, vt_w, vp_w)
        y   = (y.reshape((num_m * num_m, num_x)).T).reshape((num_x, num_m, num_m))
        return y
    
    count     = 0
    r0        = residual(x)
    rr        = xp.copy(r0)
    norm_rr   = norm_r0 = xp.linalg.norm(r0)
    converged = ((norm_rr/norm_r0 < rtol) or (norm_rr < atol))

    alpha     = 1e0
    while( not converged and (count < iter_max) ):
        jinv = xp.linalg.inv(jacobian(x))
        # line search
        while ( alpha > 1e-8 ):
            xk       = x  - alpha * xp.einsum("xil,lx->ix", jinv, rr)
            rk       = residual(xk)
            norm_rk  = xp.linalg.norm(rk)
      
            if ( norm_rk < norm_rr ):
                break
            else:
                alpha = 0.25 * alpha
    
        x         = xk
        rr        = rk
        norm_rr   = norm_rk
        count    += 1
        converged = ((norm_rr/norm_r0 < rtol) or (norm_rr < atol))
        print("iter = %04d norm_rr = %.4E norm_rr/norm_r0 = %.4E"%(count, norm_rr, norm_rr/norm_r0))
        #print(rr)
    
    #print(x)
    num_p        = spec_sp._p +1
    sph_harm_lm  = spec_sp._sph_harm_lm
    num_sh       = len(sph_harm_lm)
    mm           = spec_sp.compute_mass_matrix()

    xmf          = xp.einsum("ix,iklm->xklm", x, mf)
    q            = nfac * xp.exp(xmf) * (xp.exp(beta * vg[0]**2) * wx)[xp.newaxis, :, :, :]

    Vq           = xp.zeros(tuple([num_p * num_sh]) + vg[0].shape)
    
    for p in range(num_p):
        for lm_idx, lm in enumerate(spec_sp._sph_harm_lm):
            Vq[p * num_sh + lm_idx] = spec_sp.basis_eval_radial(vg[0], p, l=0) * spec_sp.basis_eval_spherical(vg[1], vg[2], lm[0], lm[1])
    
    fklm = np.einsum("ixklm,k,l,m->ix", np.einsum("iklm,xklm->ixklm", Vq, q), vr_w, vt_w, vp_w)
    fklm = xp.linalg.solve(mm, fklm)
    return x, fklm
    
def relative_entropy(spec_sp_psh: sp.SpectralExpansionSpherical, psh,
                     spec_sp_qsh: sp.SpectralExpansionSpherical, qsh,
                     qg, xp:np, type="KL"):
    """
    psh : true distribution function
    qsh : model distribution function
    """

    [glx,glw]    = basis.Legendre().Gauss_Pn(qg[1])
    vt_q, vt_w   = np.arccos(glx), glw
    
    [glx,glw]    = basis.Legendre().Gauss_Pn(qg[2])
    vp_q, vp_w   = np.pi * glx + np.pi , np.pi * glw

    vr_q, vr_w   = spec_sp_psh._basis_p.Gauss_Pn(qg[0])
    vg           = np.meshgrid(vr_q, vt_q, vp_q, indexing="ij")


    if spec_sp_psh.get_radial_basis_type() == basis.BasisType.SPLINES:
        wx    = vg[0]**2 
        beta  = 0.0 
        nfac  = 1.0
    elif spec_sp_psh.get_radial_basis_type() == basis.BasisType.LAGUERRE:
        wx   = np.ones_like(vg[0])
        beta = 1.0
        nfac = 4 * xp.pi**1.5
    else:
        raise NotImplementedError
    
    def compute_vq(spec_sp, vg):
        num_p        = spec_sp._p +1
        sph_harm_lm  = spec_sp._sph_harm_lm
        num_sh       = len(sph_harm_lm)
        Vq           = xp.zeros(tuple([num_p * num_sh]) + vg[0].shape)
        
        for p in range(num_p):
            for lm_idx, lm in enumerate(spec_sp._sph_harm_lm):
                Vq[p * num_sh + lm_idx] = spec_sp.basis_eval_radial(vg[0], p, l=0) * spec_sp.basis_eval_spherical(vg[1], vg[2], lm[0], lm[1])

        return Vq
    

    psh_vg = nfac * xp.einsum("ix,iklm->xklm", psh, compute_vq(spec_sp_psh, vg))
    qsh_vg = nfac * xp.einsum("ix,iklm->xklm", qsh, compute_vq(spec_sp_qsh, vg))

    psh_vg[psh_vg < 0] = 1e-15
    qsh_vg[qsh_vg < 0] = 1e-15

    return xp.einsum("xklm,k,l,m", wx * psh_vg * xp.log(xp.abs(psh_vg/qsh_vg)), vr_w, vt_w, vp_w)

def max_entropy_closure(args, data):
    pass

def bte_0d_closure(args, E, fprefix):
    """
    computes the time normalized EEDFs with 0D bte
    """
    
    import cupy as cp
    from   bte_0d3v_batched import bte_0d3v_batched
    import glow1d_utils

    class bte_0d_params():
        param           = glow1d_utils.parameters(args)
        threads         = 4
        out_fname       = fprefix
        solver_type     = "transient"
        l_max           = int(args["l_max"])
        collisions      = args.collisions
        sp_order        = int(args["sp_order"])
        spline_qpts     = int(args["spline_qpts"])
        atol            = float(args["atol"])
        rtol            = float(args["rtol"])
        max_iter        = 1000
        Te              = float(args["Te"])
        n0              = param.n0 * param.np0
        ee_collisions   = 0 
        use_dg          = 0
        ev_max          = float(args["ev_max"])
        Nr              = int(args["Nr"])
        profile         = 0
        store_eedf      = 1
        store_csv       = 1
        plot_data       = 1
        verbose         = 1
        use_gpu         = 1
        cycles          = 100
        dt              = 1e-2
        Efreq           = 13.56e6
        n_grids         = 1
        n_pts           = int(args.Np)

    args_0d_bte = bte_0d_params()

    lm_modes    = [[l,0] for l in range(args_0d_bte.l_max+1)]    
    bte_solver  = bte_0d3v_batched(args, args_0d_bte.ev_max, args_0d_bte.Te,
                                   args_0d_bte.Nr, lm_modes, args_0d_bte.n_grids, args_0d_bte.collisions)
    
    grid_idx    = 0
    n_pts       = args_0d_bte.n_pts
    f0          = bte_solver.initialize(grid_idx, n_pts,"maxwellian")

    bte_solver.set_boltzmann_parameter(grid_idx, "n0"       , np.ones(n_pts) * args_0d_bte.n0)
    bte_solver.set_boltzmann_parameter(grid_idx, "ne"       , np.ones(n_pts) * 1.0)
    bte_solver.set_boltzmann_parameter(grid_idx, "ns_by_n0" , np.ones(n_pts))
    bte_solver.set_boltzmann_parameter(grid_idx, "Tg"       , np.ones(n_pts) * args_0d_bte.param.Tg)
    bte_solver.set_boltzmann_parameter(grid_idx, "eRe"      , np.ones(n_pts) * 0.0)
    bte_solver.set_boltzmann_parameter(grid_idx, "eIm"      , np.ones(n_pts) * 0.0)
    bte_solver.set_boltzmann_parameter(grid_idx, "f0"       , f0)
    bte_solver.set_boltzmann_parameter(grid_idx,  "E"       , np.ones(n_pts) * 0.0)

    if args.use_gpu==1:
        dev_id   = 0
        bte_solver.host_to_device_setup(dev_id, grid_idx)


    # do the time-steping till convergence





data_folder               = "1dglow_hybrid_Nx400/1Torr300K_100V_Ar_3sp2r_cycle"
l_max                     = 2
Nr                        = 128
num_qvr, num_qvt, num_qvp = 120, 16, 8
data                      = plot_utils.load_data_bte(data_folder, range(0, 101), eedf_idx=None, read_cycle_avg=False, use_ionization=1)
args                      = data[0]
spec_bspline, col_list    = plot_utils.gen_spec_sp(args)

bb                        = basis.BSpline(spec_bspline._basis_p._kdomain,
                                          spec_bspline._basis_p._sp_order,
                                          Nr+1,
                                          sig_pts=spec_bspline._basis_p._sig_pts, knots_vec=None, 
                                          dg_splines=False, verbose = 0, extend_domain_with_log=True)

Lp                        = basis.gLaguerre(alpha=0.5)
sph_lm                    = [[l,0] for l in range(l_max+1)]
spec_sp                   = sp.SpectralExpansionSpherical(Nr, bb, sph_lm)
num_qvr                   = spec_sp._basis_p._num_knot_intervals * 5
nscale                    = 1

# spec_sp                   = sp.SpectralExpansionSpherical(Nr-1, Lp, sph_lm)
# nscale                    = (1/np.pi**1.5)/2



v                         = data[2]
bte_op                    = data[3]
#vsh                      = np.einsum("lo,tox->tlx", bte_op["po2sh"], v)
Te0                       = float(args["Te"])
c_gamma                   = np.sqrt(2 * (scipy.constants.elementary_charge/ scipy.constants.electron_mass))

mfuncs                    = [lambda vr, vt, vp: np.ones_like(vr),
                            lambda vr, vt, vp : vr * np.cos(vt),
                            lambda vr, vt, vp : vr **2,
                            lambda vr, vt, vp : vr * np.einsum("k,l,m->klm", col_list[1].total_cross_section(np.unique(vr) **2 * Te0), np.unique(vt), np.unique(vp)) * 3.22e22
                            ]

mnt_ops                   = bte_utils.assemble_moment_ops(spec_sp, num_qvr, num_qvt, num_qvp, mfuncs, scale = nscale)
mnt_ops_bspline           = bte_utils.assemble_moment_ops(spec_bspline, spec_bspline._basis_p._num_knot_intervals * 5,
                                                        num_qvt, num_qvp, mfuncs, scale = 1.0)


tidx        = 0
vsh         = bte_op["po2sh"] @ v[tidx]
mnt_rhs     = mnt_ops_bspline @ vsh
mnt_rhs     = mnt_rhs[:, 0::10]
x, fsh      = max_entropy_reconstruction(spec_sp, np.zeros_like(mnt_rhs), num_qvr, num_qvt, num_qvp, mnt_rhs, mfuncs, np, atol=1e-20, rtol=1e-16, iter_max=1000)
rel_entropy = relative_entropy(spec_bspline, vsh[:, 0::10], spec_sp, fsh, (num_qvr, num_qvt, num_qvp), np, "KL")

ev         = np.linspace(0, 40, 512)
vsh_n_rc   = compute_radial_components(spec_bspline, vsh[np.newaxis, :, 0::10], ev, Te0)
fsh_n_rc   = compute_radial_components(spec_sp     , fsh[np.newaxis, :, :]    , ev, Te0)

plt.semilogy(ev, np.abs(vsh_n_rc[0, 10, 0])   ,'-', label=r"1D-BTE")
plt.semilogy(ev, np.abs(fsh_n_rc[0, 10, 0])        ,'--', label=r"max entropy recons")
plt.show()
plt.close()














