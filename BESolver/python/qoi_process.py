import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
import os
import scipy.interpolate
import h5py
import sys
import basis
import utils as bte_utils
import cross_section
import collisions
import toml
import spec_spherical as sp
class op():
    def __init__(self,Np, args):
      self.Np  = Np
      self.deg = self.Np-1
      self.xp  = -np.cos(np.pi*np.linspace(0,self.deg,self.Np)/self.deg)
      #self.xp = np.linspace(-1,1, self.Np)
      from numpy.polynomial import chebyshev as cheb
      # Operators
      ident = np.identity(self.Np)

      # V0p: Coefficients to values at xp
      self.V0p = np.polynomial.chebyshev.chebvander(self.xp, self.deg)

      # V0pinv: xp values to coefficients
      self.V0pinv = np.linalg.solve(self.V0p, ident)

      # V1p: coefficients to derivatives at xp
      self.V1p = np.zeros((self.Np,self.Np))
      for i in range(0,self.Np):
          self.V1p[:,i] = cheb.chebval(self.xp, cheb.chebder(ident[i,:], m=1))

      # Dp: values at xp to derivatives at xp
      self.Dp = self.V1p @ self.V0pinv
      
      # V2p: coefficients to 2nd derivatives at xp
      self.V2p = np.zeros((self.Np,self.Np))
      for i in range(0,self.Np):
          self.V2p[:,i] = cheb.chebval(self.xp, cheb.chebder(ident[i,:], m=2))

      # Lp: values at xp to 2nd derivatives at xp
      self.Lp = self.V2p @ self.V0pinv
      
      # self.xp = np.linspace(-1,1,self.Np)
      # self.Dp = np.eye(self.Np)
      # self.Lp = np.eye(self.Np)
      
      # LpD: values at xp to 2nd derivatives at xc, with identity
      # for top and bottom row (for Dirichlet BCs)
      self.LpD = np.identity(self.Np)
      self.LpD[1:-1,:] = self.Lp[1:-1,:]
      self.LpD_inv     = np.linalg.solve(self.LpD, np.eye(self.Np))
      
      tp = toml.load(args["par_file"])["glow_1d"]
      self.L     = 0.5 * tp["L"]             # m 
      self.V0    = tp["V0"]                  # V
      self.f     = tp["freq"]                # Hz
      self.tau   = (1/self.f)                # s
      self.qe    = scipy.constants.e         # C
      self.eps0  = scipy.constants.epsilon_0 # eps_0 
      self.kB    = scipy.constants.Boltzmann # J/K
      self.ev_to_K = scipy.constants.electron_volt / scipy.constants.Boltzmann
      self.me    = scipy.constants.electron_mass
      self.Teb   = tp["Teb"]
      self.gamma = tp["gamma"]
      
      self.Tg    = tp["Tg"]  #K
      self.p0    = tp["p0"]  #Torr

      if self.Tg !=0: 
        self.n0    = self.p0 * scipy.constants.torr / (scipy.constants.Boltzmann * self.Tg) #3.22e22                   #m^{-3}
      else:
        self.n0    = 3.22e22                   #m^{-3}
        
      self.np0   = 8e16                      #"nominal" electron density [1/m^3]

      # raw transport coefficients 
      self._De    = (3.86e22) * 1e2 / self.n0 #m^{2}s^{-1}
      self._mu_e  = (9.66e21) * 1e2 / self.n0 #V^{-1} m^{2} s^{-1} 
        
      self.Di    = (2.07e18) * 1e2 / self.n0 #m^{2} s^{-1}
      self.mu_i  = (4.65e19) * 1e2 / self.n0 #V^{-1} m^{2} s^{-1}
      
      self.eps0  = scipy.constants.epsilon_0 # eps_0 
      self.alpha = self.np0 * self.L**2 * self.qe / (self.eps0 * self.V0)
      
      self.use_tab_data=1
      
      self.n0 = self.n0/self.np0
      
      if(self.use_tab_data==1):
        ki   = np.genfromtxt("Ar3species/Ar_1Torr_300K/Ionization.300K.txt" , delimiter=",", skip_header=True) # m^3/s
        mu_e = np.genfromtxt("Ar3species/Ar_1Torr_300K/Mobility.300K.txt"   , delimiter=",", skip_header=True) # mu * n0 (1/(Vms))
        De   = np.genfromtxt("Ar3species/Ar_1Torr_300K/Diffusivity.300K.txt", delimiter=",", skip_header=True) # D  * n0 (1/(ms))
        
        # non-dimentionalized QoIs
        ki  [:, 0]  *= (1/self.ev_to_K)
        mu_e[:, 0]  *= (1/self.ev_to_K)
        De [:, 0]   *= (1/self.ev_to_K)
        
        ki  [:, 1]  *= (self.np0 * self.tau)
        mu_e[:, 1]  *= (self.V0 * self.tau / (self.L**2 * self.n0 * self.np0))
        De [:, 1]   *= (self.tau / (self.L**2 *self.n0 * self.np0) )
        
        ki_data   = ki
        mu_e_data = mu_e
        De_data   = De
        
        # non-dimensional QoI interpolations and their derivatives, w.r.t., ne, nTe
        ki            = scipy.interpolate.UnivariateSpline(ki[:,0],  ki  [:,1], k=1, s=0, ext="const")
        ki_d          = ki.derivative(n=1)
        
        self.ki       =  lambda nTe, ne : ki(nTe/ne)
        self.ki_ne    =  lambda nTe, ne : ki_d(nTe/ne) * (-nTe/(ne**2))
        self.ki_nTe   =  lambda nTe, ne : ki_d(nTe/ne) * (1/ne)
        
        mu_e          = scipy.interpolate.UnivariateSpline(mu_e[:,0],  mu_e[:,1], k=1, s=0, ext="const")
        mu_e_d        = mu_e.derivative(n=1)
        self.mu_e     = lambda nTe, ne : mu_e(nTe/ne)
        self.mu_e_ne  = lambda nTe, ne : mu_e_d(nTe/ne) * (-nTe/(ne**2))
        self.mu_e_nTe = lambda nTe, ne : mu_e_d(nTe/ne) * (1/ne)
        
        De            = scipy.interpolate.UnivariateSpline(De [:,0],   De [:,1], k=1, s=0, ext="const")
        De_d          = De.derivative(n=1)
        self.De       = lambda nTe, ne : De(nTe/ne)
        self.De_ne    = lambda nTe, ne : De_d(nTe/ne) * (-nTe/(ne**2))
        self.De_nTe   = lambda nTe, ne : De_d(nTe/ne) * (1/ne)
        
        self.mu_fac   = (1.0) / ( (self.V0 * self.tau/(self.L**2)) )
        self.D_fac    = (1.0) / (self.tau/(self.L**2))
        self.r_fac    = 1/(self.np0 * self.tau)
      
    def solve_poisson(self, ne, ni, time):
        """Solve Gauss' law for the electric potential.

        Inputs:
          ne   : Values of electron density at xp
          ni   : Values of ion density at xp
          time : Current time

        Outputs: None (sets self.phi to computed potential)
        """
        xp    = np#self.xp_module
        r     = - self.alpha * (ni-ne)
        r[0]  = xp.sin(2 * xp.pi * time) #+ self.params.verticalShift
        r[-1] = 0.0
        return xp.dot(self.LpD_inv, r)

def load_run_args(folder):
    args   = dict()
    
    f  = open(folder+"/1d_glow_args.txt")
    st = f.read().strip()
    st = st.split(":")[1].strip()
    st = st.split("(")[1]
    st = st.split(")")[0].strip()
    st = st.split(",")
    
    for s in st:
        kv = s.split("=")
        if len(kv)!=2 or kv[0].strip()=="collisions":
            continue
        args[kv[0].strip()]=kv[1].strip().replace("'", "")

    return args

def gen_spec_sp(args):
    sig_pts             = list()
    bs_coll_list        = list()
    collision_names     = list()
    coll_list           = list()
    
    collision_str       = "lxcat_data/eAr_crs.Biagi.3sp2r"

    avail_species                     = cross_section.read_available_species(collision_str)
    cross_section.CROSS_SECTION_DATA  = cross_section.read_cross_section_data(collision_str)
    cross_section_data                = cross_section.CROSS_SECTION_DATA
      
    print("==========read collissions=====================")
    collision_count = 0
    for col_str, col_data in cross_section_data.items():
        print(col_str, col_data["type"])
        g = collisions.electron_heavy_binary_collision(col_str, collision_type=col_data["type"])
        g.reset_scattering_direction_sp_mat()
        coll_list.append(g)
        collision_names.append("C%d"%(collision_count)) 
        collision_count+=1
    print("===============================================")
    print("number of total collisions = %d " %(len(coll_list)))
    num_collisions = len(coll_list)
    bs_coll_list   = coll_list
    
    for col_idx, g in enumerate(bs_coll_list):
        g  = bs_coll_list[col_idx]
        if g._reaction_threshold != None and g._reaction_threshold >0:
            sig_pts.append(g._reaction_threshold)
    
    Te               = (float)(args["Te"])
    vth              = collisions.electron_thermal_velocity(Te * (scipy.constants.elementary_charge / scipy.constants.Boltzmann))
    c_gamma          = np.sqrt(2 * (scipy.constants.elementary_charge/ scipy.constants.electron_mass))
      
    sig_pts          = np.sort(np.array(list(set(sig_pts))))
    maxwellian       = bte_utils.get_maxwellian_3d(vth, 1.0)
    
    dg_nodes         = np.sqrt(np.array(sig_pts)) * c_gamma / vth
    ev_range         = (0, (float)(args["ev_max"])) #((0 * vth /self.c_gamma)**2, (self.vth_fac * vth /self.c_gamma)**2)
    k_domain         = (np.sqrt(ev_range[0]) * c_gamma / vth, np.sqrt(ev_range[1]) * c_gamma / vth)
    use_ee           = (int)(args["ee_collisions"])
    bs_use_dg        = 0 
    
    if bs_use_dg==1:
        print("DG boltzmann grid v-space ev=", ev_range, " v/vth=",k_domain)
        print("DG points \n", sig_pts)
    else:
        print("CG boltzmann grid v-space ev=", ev_range, " v/vth=",k_domain)
    
    
    sp_order    = (int)(args["sp_order"])
    nr          = (int)(args["Nr"])
    spline_qpts = (int)(args["spline_qpts"])
    l_max       = (int)(args["l_max"])
    
    bs_lm       = [(l, 0) for l in range(l_max+1)]
    ev_extend   = 2
    verbose     = 1 
    
    # construct the spectral class 
    if (ev_extend==0):
        bb                     = basis.BSpline(k_domain, sp_order, nr + 1, sig_pts=dg_nodes, knots_vec=None, dg_splines=bs_use_dg, verbose = verbose)
    elif (ev_extend==1):
        print("using uniform grid extention")
        bb                     = basis.BSpline(k_domain, sp_order, nr + 1, sig_pts=dg_nodes, knots_vec=None, dg_splines=bs_use_dg, verbose = verbose, extend_domain=True)
    else:
        assert ev_extend==2
        print("using log spaced grid extention")
        bb                     = basis.BSpline(k_domain, sp_order, nr + 1, sig_pts=dg_nodes, knots_vec=None, dg_splines=bs_use_dg, verbose = verbose,extend_domain_with_log=True)
    
    spec_sp                = sp.SpectralExpansionSpherical(nr, bb, bs_lm)
    spec_sp._num_q_radial  = bb._num_knot_intervals * spline_qpts
    
    return spec_sp, coll_list

def load_data_bte(folder, cycles, read_cycle_avg=False):
    
    args   = load_run_args(folder)

    bte_op = dict()
    bte_op["mass"]      = np.load(folder+"/1d_glow_bte_mass_op.npy")
    bte_op["temp"]      = np.load(folder+"/1d_glow_bte_temp_op.npy")
    bte_op["mobility"]  = np.load(folder+"/1d_glow_bte_mobility.npy")
    bte_op["diffusion"] = np.load(folder+"/1d_glow_bte_diffusion.npy")
    bte_op["cmat"]      = np.load(folder+"/1d_glow_bte_cmat.npy")
    bte_op["g0"]        = np.load(folder+"/1d_glow_bte_op_g0.npy")
    bte_op["g2"]        = np.load(folder+"/1d_glow_bte_op_g2.npy")
    
    bte_op["po2sh"]     = np.load(folder+"/1d_glow_bte_po2sh.npy")
    bte_op["psh2o"]     = np.load(folder+"/1d_glow_bte_psh2o.npy")
    
    u  = list()
    v  = list()
    
    for idx in cycles:
        if read_cycle_avg:
            u.append(np.load(folder+"/1d_glow_%04d_u_avg.npy"%(idx)))
            v.append(np.load(folder+"/1d_glow_%04d_v_avg.npy"%(idx)))
        else:
            u.append(np.load(folder+"/1d_glow_%04d_u.npy"%(idx)))
            v.append(np.load(folder+"/1d_glow_%04d_v.npy"%(idx)))
        
    u  = np.array(u)
    v  = np.array(v)
    return [args, u, v, bte_op]

def compute_rate_coefficients(args, bte_op, v, collisions):
    n_pts    = v.shape[0] 
    mm_fac   = np.sqrt(4 * np.pi) 
    mm_op    = bte_op["mass"]
    c_gamma  = np.sqrt(2 * (scipy.constants.elementary_charge/ scipy.constants.electron_mass))
    vth      = (float) (args["Te"])**0.5 * c_gamma
    v_lm     = np.array([np.dot(bte_op["po2sh"], v[idx]) for idx in range(n_pts)])
    scale    = np.array([np.dot(mm_op / mm_fac, v_lm[idx]) * (2 * (vth/c_gamma)**3) for idx in range(n_pts)])
    v_lm_n   = np.array([v_lm[idx]/scale[idx] for idx in range(n_pts)])
    num_sh   = (int) (args["l_max"]) + 1
    return [np.dot(bte_op[col_type], v_lm_n[:, 0::num_sh, :]) for col_type in collisions]

def compute_mean_velocity(spec_sp, args, bte_op, v, n0):
    n_pts    = v.shape[0] 
    mm_fac   = np.sqrt(4 * np.pi) 
    mm_op    = bte_op["mass"]
    c_gamma  = np.sqrt(2 * (scipy.constants.elementary_charge/ scipy.constants.electron_mass))
    vth      = (float) (args["Te"])**0.5 * c_gamma
    v_lm     = np.array([np.dot(bte_op["po2sh"], v[idx]) for idx in range(n_pts)])
    ne       = np.array([np.dot(mm_op, v_lm[idx])  for idx in range(n_pts)])
    num_sh   = (int) (args["l_max"]) + 1
    
    num_q_vr = (int(args["sp_order"]) + 3) * spec_sp._basis_p._num_knot_intervals
    vop      = bte_utils.mean_velocity_op(spec_sp, NUM_Q_VR = num_q_vr, NUM_Q_VT = 32, NUM_Q_VP = 16, scale=1)
    
    vx      = np.einsum("l,ilk->ik",vop[0], v_lm) * vth / ne #[ms-1] 
    vy      = np.einsum("l,ilk->ik",vop[1], v_lm) * vth / ne #[ms-1]
    vz      = np.einsum("l,ilk->ik",vop[2], v_lm) * vth / ne #[ms-1]
    
    u       = [vx, vy, vz]
    
    cv_lm   = n0 * np.einsum("ml,ilk->imk",bte_op["cmat"], v_lm) # s^{-1}
    
    c_mom_x = np.einsum("l,ilk->ik",vop[0], cv_lm) * vth / ne / vx 
    c_mom_y = np.einsum("l,ilk->ik",vop[1], cv_lm) * vth / ne / vy 
    c_mom_z = np.einsum("l,ilk->ik",vop[2], cv_lm) * vth / ne / vz 
    
    c_mom   = [c_mom_x, c_mom_y, c_mom_z]
    return u, c_mom

def compute_kinetic_energy_tensor(spec_sp, args, bte_op, v):
    
    n_pts    = v.shape[0] 
    mm_fac   = np.sqrt(4 * np.pi) 
    mm_op    = bte_op["mass"]
    c_gamma  = np.sqrt(2 * (scipy.constants.elementary_charge/ scipy.constants.electron_mass))
    vth      = (float) (args["Te"])**0.5 * c_gamma
    v_lm     = np.array([np.dot(bte_op["po2sh"], v[idx]) for idx in range(n_pts)])
    ne       = np.array([np.dot(mm_op, v_lm[idx])  for idx in range(n_pts)])
    num_sh   = (int) (args["l_max"]) + 1
    
    num_q_vr = (int(args["sp_order"]) + 4) * spec_sp._basis_p._num_knot_intervals
    
    # compute stress tensor in eV
    a1       = (vth**2) * 0.5 * scipy.constants.electron_mass * (2/3/scipy.constants.Boltzmann) / collisions.TEMP_K_1EV
    vop      = 1.5 * a1 * bte_utils.kinetic_stress_tensor_op(spec_sp, NUM_Q_VR = num_q_vr, NUM_Q_VT = 32, NUM_Q_VP = 16, scale=1) 
    
    P        = np.array([np.einsum("l,ilk->ik",vop[i], v_lm) / ne for i in range(vop.shape[0])]) # [eV] 
    return P    
    
def compute_diffusion_and_effective_mobility(args, bte_op, v, n0):
    n_pts    = v.shape[0] 
    mm_fac   = np.sqrt(4 * np.pi) 
    mm_op    = bte_op["mass"]
    c_gamma  = np.sqrt(2 * (scipy.constants.elementary_charge/ scipy.constants.electron_mass))
    vth      = (float) (args["Te"])**0.5 * c_gamma
    v_lm     = np.array([np.dot(bte_op["po2sh"], v[idx]) for idx in range(n_pts)])
    scale    = np.array([np.dot(mm_op / mm_fac, v_lm[idx]) * (2 * (vth/c_gamma)**3) for idx in range(n_pts)])
    v_lm_n   = np.array([v_lm[idx]/scale[idx] for idx in range(n_pts)])
    num_sh   = (int) (args["l_max"]) + 1
    
    muE_op   = bte_op["mobility"]
    De_op    = bte_op["diffusion"]
    
    mue_E    = np.einsum("l,ilk->ik", muE_op, v_lm_n[:, 1::num_sh, :]) * ((c_gamma / (3 * ( 1 / n0)))) /n0
    De       = np.einsum("l,ilk->ik", De_op , v_lm_n[:, 0::num_sh, :]) * (c_gamma / 3.) / n0 
    
    return mue_E, De

def compute_radial_components(args, spec_sp, bte_op, ev_grid, v, n0):
    n_pts    = v.shape[0] 
    mm_fac   = np.sqrt(4 * np.pi) 
    mm_op    = bte_op["mass"]
    c_gamma  = np.sqrt(2 * (scipy.constants.elementary_charge/ scipy.constants.electron_mass))
    vth      = (float) (args["Te"])**0.5 * c_gamma
    v_lm     = np.array([np.dot(bte_op["po2sh"], v[idx]) for idx in range(n_pts)])
    scale    = np.array([np.dot(mm_op / mm_fac, v_lm[idx]) * (2 * (vth/c_gamma)**3) for idx in range(n_pts)])
    v_lm_n   = np.array([v_lm[idx]/scale[idx] for idx in range(n_pts)])
    
    vr       = np.sqrt(ev_grid) * c_gamma/ vth
    num_p    = spec_sp._p +1 
    num_sh   = len(spec_sp._sph_harm_lm)
    n_xpts   = v_lm_n.shape[-1]
        
    output   = np.zeros((n_pts, n_xpts, num_sh, len(vr)))
    Vqr      = spec_sp.Vq_r(vr, 0, 1)
    
    
    for l_idx, lm in enumerate(spec_sp._sph_harm_lm):
        output [:, :, l_idx, :] = np.einsum("tvx,vu->txu",v_lm_n[:, l_idx::num_sh, :],  Vqr)

    return output, v_lm, v_lm_n

def compute_E(ne, ni, tt, cheb):
    phi  = np.array([cheb.solve_poisson(ne[i], ni[i], tt[i]) for i in range(len(tt))])
    E    = - (cheb.V0/cheb.L) * np.dot(cheb.Dp, phi.T).T
    
    return E

def Etx_interpolate(E, tt, xx, tt_old, xx_old):
    cheb = op(E.shape[1])
    #assert cheb.xp == xx_old
    P    = np.dot(np.polynomial.chebyshev.chebvander(xx, cheb.deg), cheb.V0pinv)
    
    Ex   = np.dot(P, E.T).T
    
    Et   = [scipy.interpolate.interp1d(tt_old, Ex[:, k]) for k in range(Ex.shape[1])]
    
    Etx  = np.zeros((len(tt), len(xx)))
    for k in range(Ex.shape[1]):
        Etx[:, k] = Et[k](tt)
    
    return Etx

def time_average(qoi, tt):
    """
    computes the time average on grids
    """
    # check if tt is uniform
    nT   = len(tt)
    T    = (tt[-1]-tt[0])
    dt   = T/(nT-1)
    
    assert abs(tt[1] -tt[0] - dt) < 1e-10
    
    tw    = np.ones_like(tt) * dt
    tw[0] = 0.5 * tw[0]; tw[-1] = 0.5 * tw[-1];
    
    assert (T-np.sum(tw)) < 1e-12
    
    return np.dot(tw, qoi)

print("args:", sys.argv)
folder_name = sys.argv[1]
idx_range   = list(range((int)(sys.argv[2]), (int)(sys.argv[3]), (int)(sys.argv[4])))
run_type    = sys.argv[5]
glist       = list(["g0", "g2"])
col_names   = ["elastic", "ionization"]
#run arguments
args = load_run_args(folder_name)
print("run args \n", args)
Np        = int(args["Np"])
cheb      = op(Np, args)
N0        = cheb.n0 * cheb.np0
xc        = cheb.xp
dt        = float(args["cfl"])
io_freq   = int(float(args["io_cycle_freq"])/dt)

tb        = io_freq * idx_range[0] * dt
te        = io_freq * idx_range[-1] * dt

print("Nx= ", Np, " n0 = %.8E"%(N0), "t=(%.4E, %.4E)"%(tb, te))


if (run_type == "bte"):
    spec_sp, coll_list = gen_spec_sp(args)
    ev_grid           = np.linspace(0, float(args["ev_max"]), 512)
    #ev_grid            = np.linspace(0, 2 * float(args["ev_max"]), 1023)

    d                  = load_data_bte(folder_name, idx_range, read_cycle_avg=False)
    ki                 = compute_rate_coefficients(d[0], d[3], d[2], collisions=glist)
    u , Cu             = compute_mean_velocity(spec_sp,d[0], d[3], d[2], N0)
    mueE, De           = compute_diffusion_and_effective_mobility(d[0], d[3], d[2], N0)
    P                  = compute_kinetic_energy_tensor(spec_sp, d[0], d[3], d[2])
    tt                 = np.linspace(tb, te, d[1][:, :, 0].shape[0])
    ne                 = d[1][:, :, 0]
    ni                 = d[1][:, :, 1]
    Te                 = d[1][:, :, 2]
    Ef                 = compute_E(d[1][:, :, 0], d[1][:, :, 1],tt, cheb)
    assert Np == d[1][:, :, 0].shape[1]

    COMPUTE_RADIAL_COMP = 1
    F = h5py.File("%s/macro_idx_%04d_to_%04d.h5"%(folder_name, idx_range[0], idx_range[-1]), 'w')
    F.create_dataset("time[T]"      , data = tt)
    F.create_dataset("x[-1,1]"      , data = xc)
    F.create_dataset("E[Vm^-1]"     , data = Ef)
    F.create_dataset("ne[m^-3]"     , data = cheb.np0 * ne)
    F.create_dataset("ni[m^-3]"     , data = cheb.np0 * ni)
    F.create_dataset("Te[eV]"		, data = Te)
    F.create_dataset("ke[m^3s^{-1}]", data = ki[0])
    F.create_dataset("ki[m^3s^{-1}]", data = ki[1])

    F.create_dataset("ux[ms^{-1}]"  , data = u[0])
    F.create_dataset("uy[ms^{-1}]"  , data = u[1])
    F.create_dataset("uz[ms^{-1}]"  , data = u[2])
    F.create_dataset("Cux[ms^{-1}]" , data = Cu[0])
    F.create_dataset("Cuy[ms^{-1}]" , data = Cu[1])
    F.create_dataset("Cuz[ms^{-1}]" , data = Cu[2])

    F.create_dataset("mueE[ms^{-1}]"   , data = mueE)
    F.create_dataset("De  [m^2s^{-1}]" , data = De)
    F.create_dataset("P   [m2s^{-2}]"  , data = P)

    if (COMPUTE_RADIAL_COMP==1):
        fl_comp , f_lm, f_lm_n  = compute_radial_components(d[0], spec_sp, d[3], ev_grid , d[2], N0)
        F.create_dataset("evgrid[eV]" , data=ev_grid)
        F.create_dataset("fl[eV^-1.5]", data=fl_comp)
        F.create_dataset("Ftvx"       , data=f_lm_n)


    F.create_dataset("avg_E[Vm^-1]"       , data = time_average(Ef , tt))
    F.create_dataset("avg_ne[m^-3]"       , data = cheb.np0 * time_average(ne , tt))
    F.create_dataset("avg_ni[m^-3]"       , data = cheb.np0 * time_average(ni , tt))
    F.create_dataset("avg_Te[eV]"		  , data = time_average(Te , tt))
    F.create_dataset("avg_ke[m^3s^{-1}]"  , data = time_average(ki[0] , tt))
    F.create_dataset("avg_ki[m^3s^{-1}]"  , data = time_average(ki[1] , tt))

    F.create_dataset("avg_energy_density[eVkgm^{-3}]", data = 1.5 * cheb.np0 * scipy.constants.electron_mass * time_average(Te * ne, tt))
    F.create_dataset("avg_elastic [m^-3s^{-1}]"      , data = cheb.n0 * cheb.np0 **2 * time_average(ki[0] * ne, tt))
    F.create_dataset("avg_ion_prod[m^-3s^{-1}]"      , data = cheb.n0 * cheb.np0 **2 * time_average(ki[1] * ne, tt))
    F.close()
elif (run_type == "fluid"):
    U           = np.array([np.load("%s/1d_glow_%04d.npy"%(folder_name,idx)) for idx in range(0, 101)])
    ne          = U[:, :, 0]    
    ni          = U[:, :, 1]
    Te          = U[:, :, 2] / U[:, :, 0]
    tt          = np.linspace(0, 1, ne.shape[0])
    Ef          = compute_E(ne, ni, tt, cheb)
    ki          = np.array([cheb.ki(Te[i], 1) * cheb.r_fac for i in range(len(tt))])

    F = h5py.File("%s/macro.h5"%(folder_name), 'w')
    F.create_dataset("time[T]"      , data = tt)
    F.create_dataset("x[-1,1]"      , data = xc)
    F.create_dataset("E[Vm^-1]"     , data = Ef)
    F.create_dataset("ne[m^-3]"     , data = cheb.np0 * ne)
    F.create_dataset("ni[m^-3]"     , data = cheb.np0 * ni)
    F.create_dataset("Te[eV]"		, data = Te)
    F.create_dataset("ki[m^3s^{-1}]", data = ki)
    F.create_dataset("avg_E[Vm^-1]"       , data = time_average(Ef , tt))
    F.create_dataset("avg_ne[m^-3]"       , data = cheb.np0 * time_average(ne , tt))
    F.create_dataset("avg_ni[m^-3]"       , data = cheb.np0 * time_average(ni , tt))
    F.create_dataset("avg_Te[eV]"		  , data = time_average(Te , tt))
    F.create_dataset("avg_ki[m^3s^{-1}]"  , data = time_average(ki , tt))
    F.create_dataset("avg_energy_density[eVkgm^{-3}]", data = 1.5 * cheb.np0 * scipy.constants.electron_mass * time_average(Te * ne, tt))
    F.create_dataset("avg_ion_prod[m^-3s^{-1}]"      , data = cheb.n0 * cheb.np0 **2 * time_average(ki * ne, tt))
    F.close()
else:
    raise NotImplementedError

ff = h5py.File("%s/macro_idx_%04d_to_%04d.h5"%(folder_name, idx_range[0], idx_range[-1]))
xx     = ff["x[-1,1]"][()]
avg_E  = ff["avg_energy_density[eVkgm^{-3}]"][()]
avg_ne = ff["avg_ne[m^-3]"][()]
avg_Te = ff["avg_Te[eV]"][()]
ff.close()

plt.figure(figsize=(12, 4), dpi=200)
plt.subplot(1,3, 1)
plt.semilogy(xx, avg_ne)
plt.xlabel(r"x")
plt.ylabel(r"number density [$m^{-3}$]")
plt.grid(visible=True)

plt.subplot(1,3, 2)
plt.semilogy(xx, avg_E)
plt.xlabel(r"x")
plt.ylabel(r"energy mass density [$eV Kg m^{-3}$]")
plt.grid(visible=True)

plt.subplot(1,3, 3)
plt.semilogy(xx, avg_Te)
plt.xlabel(r"x")
plt.ylabel(r"average temp [$eV$]")
plt.grid(visible=True)

plt.tight_layout()
plt.savefig("%s/macro_idx_%04d_to_%04d.png"%(folder_name, idx_range[0], idx_range[-1]))
plt.close()