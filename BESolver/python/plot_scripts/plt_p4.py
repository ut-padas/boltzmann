import numpy as np
import matplotlib.pyplot as plt
import sys

import scipy.constants
import plot_utils
import scipy.interpolate
from itertools import cycle

plt.rcParams.update({
    #"text.usetex": True,
    "font.size": 16,
    #"ytick.major.size": 3,
    #"font.family": "Helvetica",
    "lines.linewidth":2.0
})
import sys
sys.path.append("../.")
import basis
import cross_section
import utils as bte_utils
import spec_spherical as sp
import collisions

def load_data_bte(folder, cycles, eedf_idx=None, read_cycle_avg=False):
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
        args[kv[0].strip()]=kv[1].strip()
    
    bte_op = dict()
    bte_op["mass"]  = np.load(folder+"/1d_glow_bte_mass_op.npy")
    bte_op["temp"]  = np.load(folder+"/1d_glow_bte_temp_op.npy")
    bte_op["g0"]    = np.load(folder+"/1d_glow_bte_op_g0.npy")
    
    if use_ionization==1:
        bte_op["g2"]    = np.load(folder+"/1d_glow_bte_op_g2.npy")
    
    bte_op["po2sh"] = np.load(folder+"/1d_glow_bte_po2sh.npy")
    bte_op["psh2o"] = np.load(folder+"/1d_glow_bte_psh2o.npy")
    
    try:
        bte_op["mobility"]  = np.load(folder+"/1d_glow_bte_mobility.npy")
        bte_op["diffusion"] = np.load(folder+"/1d_glow_bte_diffusion.npy")
    except:
        bte_op["mobility"]  = None
        bte_op["diffusion"] = None 
    
    
    u  = list()
    v  = list()
    v1 = list()
    
    read_eedf = not(eedf_idx is None)
    
    for idx in cycles:
        if read_cycle_avg:
            u.append(np.load(folder+"/1d_glow_%04d_u_avg.npy"%(idx)))
            v.append(np.load(folder+"/1d_glow_%04d_v_avg.npy"%(idx)))
        else:
            u.append(np.load(folder+"/1d_glow_%04d_u.npy"%(idx)))
            v.append(np.load(folder+"/1d_glow_%04d_v.npy"%(idx)))
        
    if read_eedf:
        for idx in cycles:
            w = np.load(folder+"/1d_glow_%04d_v_eedf.npy"%(idx))
            v1.append(w[eedf_idx])
        
    u  = np.array(u)
    v  = np.array(v)
    v1 = np.array(v1)
    return [args, u, v, bte_op, v1]

def compute_rate_coefficients(args, bte_op, v, collisions_list):
    t_pts    = v.shape[0] 
    mm_fac   = np.sqrt(4 * np.pi) 
    mm_op    = bte_op["mass"]
    c_gamma  = np.sqrt(2 * (scipy.constants.elementary_charge/ scipy.constants.electron_mass))
    vth      = (float) (args["Te"])**0.5 * c_gamma
    v_lm     = np.array([np.dot(bte_op["po2sh"], v[idx]) for idx in range(t_pts)])
    scale    = np.array([np.dot(mm_op / mm_fac, v_lm[idx]) * (2 * (vth/c_gamma)**3) for idx in range(t_pts)])
    v_lm_n   = np.array([v_lm[idx]/scale[idx] for idx in range(t_pts)])
    num_sh   = (int) (args["l_max"]) + 1
    return [np.dot(bte_op[col_type], v_lm_n[:, 0::num_sh, :]) for col_type in collisions_list]

def compute_mobility(args, bte_op, v, EbyN):
    t_pts    = v.shape[0] 
    mm_fac   = np.sqrt(4 * np.pi) 
    mm_op    = bte_op["mass"]
    c_gamma  = np.sqrt(2 * (scipy.constants.elementary_charge/ scipy.constants.electron_mass))
    vth      = (float) (args["Te"])**0.5 * c_gamma
    v_lm     = np.array([np.dot(bte_op["po2sh"], v[idx]) for idx in range(t_pts)])
    scale    = np.array([np.dot(mm_op / mm_fac, v_lm[idx]) * (2 * (vth/c_gamma)**3) for idx in range(t_pts)])
    v_lm_n   = np.array([v_lm[idx]/scale[idx] for idx in range(t_pts)])
    num_sh   = (int) (args["l_max"]) + 1
    return np.dot(bte_op["mobility"], v_lm_n[:, 1::num_sh, :]) * (-(c_gamma / (3 * EbyN)))

def compute_diffusion(args, bte_op, v):
    t_pts    = v.shape[0] 
    mm_fac   = np.sqrt(4 * np.pi) 
    mm_op    = bte_op["mass"]
    c_gamma  = np.sqrt(2 * (scipy.constants.elementary_charge/ scipy.constants.electron_mass))
    vth      = (float) (args["Te"])**0.5 * c_gamma
    v_lm     = np.array([np.dot(bte_op["po2sh"], v[idx]) for idx in range(t_pts)])
    scale    = np.array([np.dot(mm_op / mm_fac, v_lm[idx]) * (2 * (vth/c_gamma)**3) for idx in range(t_pts)])
    v_lm_n   = np.array([v_lm[idx]/scale[idx] for idx in range(t_pts)])
    num_sh   = (int) (args["l_max"]) + 1
    return np.dot(bte_op["diffusion"], v_lm_n[:, 0::num_sh, :]) * (c_gamma / 3.)

def gen_spec_sp(args):
    sig_pts             = list()
    bs_coll_list        = list()
    collision_names     = list()
    coll_list           = list()
    
    collision_str       = "../lxcat_data/eAr_crs.Biagi.3sp2r"

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
    
    return spec_sp

def compute_radial_components(args, bte_op, spec_sp, ev: np.array, ff):
    
    t_pts    = ff.shape[0]
    
    ff_lm    = np.array([np.dot(bte_op["po2sh"], ff[idx]) for idx in range(t_pts)])
    Te       = (float)(args["Te"])
    vth      = collisions.electron_thermal_velocity(Te * (scipy.constants.elementary_charge / scipy.constants.Boltzmann))
    c_gamma  = np.sqrt(2 * (scipy.constants.elementary_charge/ scipy.constants.electron_mass))
    
    vth      = vth
    spec_sp  = spec_sp
    
    vr       = np.sqrt(ev) * c_gamma / vth
    num_p    = spec_sp._p +1 
    num_sh   = len(spec_sp._sph_harm_lm)
    n_pts    = ff.shape[2]
    
    output   = np.zeros((t_pts, n_pts, num_sh, len(vr)))
    Vqr      = spec_sp.Vq_r(vr,0,1)
    
    
    mm_op    = bte_op["mass"]
    mm_fac   = np.sqrt(4 * np.pi) 
    
    scale    = np.array([np.dot(mm_op / mm_fac, ff_lm[idx]) * (2 * (vth/c_gamma)**3) for idx in range(t_pts)])
    ff_lm_n  = np.array([ff_lm[idx]/scale[idx] for idx in range(t_pts)])
    
    for idx in range(t_pts):
        ff_lm_T  = ff_lm_n[idx].T
        for l_idx, lm in enumerate(spec_sp._sph_harm_lm):
            output[idx, :, l_idx, :] = np.dot(ff_lm_T[:,l_idx::num_sh], Vqr)

    return output


folder_name    = sys.argv[1]
step_begin     = (int)(sys.argv[2])
step_end       = (int)(sys.argv[3])
step_sz        = (int)(sys.argv[4])
use_ionization = (int)(sys.argv[5])
dt_scaling_fac = (float)(sys.argv[6])
out_fname      = (sys.argv[7])
idx_range      = list(range(step_begin, step_end, step_sz))

collisions_list     = ["g0"] 
if use_ionization ==1:
    collisions_list.append("g2")
    
op         = plot_utils.op(200)    
d0         = load_data_bte(folder_name, idx_range, list(np.arange(200)), read_cycle_avg=True)
ki0        = compute_rate_coefficients(d0[0], d0[3], d0[2], collisions_list)
mu_e       = compute_mobility(d0[0], d0[3], d0[2], (1/(op.n0 * op.np0)))
De         = compute_diffusion(d0[0], d0[3], d0[2])

u          = d0[1]
v          = d0[2] 

spec_sp    = gen_spec_sp(d0[0])


u_f        = np.load("/home/mfernando/Research/bte-docs/1dglow/fluid_Ar_3_species_100V/1d_glow_0008_avg.npy")
ne_f       = u_f[:, 0]
ni_f       = u_f[:, 1]
Te_f       = u_f[:, 2]/ne_f

plt.figure(figsize=(8, 8), dpi=200)
plt.plot(op.xp, u[0, : , 0] * op.np0, "r-",  label=r"$n_e (BTE)$")
plt.plot(op.xp, u[0, : , 1] * op.np0, "b-",  label=r"$n_i (BTE)$")

plt.plot(op.xp, ne_f * op.np0, "r--", label=r"$n_e (fluid)$")
plt.plot(op.xp, ni_f * op.np0, "b--", label=r"$n_i (fluid)$")

plt.xlabel(r"x")
plt.ylabel(r"density [$m^3$]")
plt.grid()
plt.legend()
plt.savefig("%s/%s_density.png"%(folder_name, out_fname))
plt.close()

plt.figure(figsize=(8,8), dpi=200)
plt.plot(op.xp, u[0, : , 2], 'r-', label=r"BTE")
plt.plot(op.xp, Te_f, 'b--', label=r"fluid")
plt.xlabel(r"x")
plt.ylabel(r"temperature [$eV$]")
plt.grid()
plt.legend()
plt.savefig("%s/%s_temp.png"%(folder_name, out_fname))
plt.close()


plt.figure(figsize=(8,8), dpi=200)
plt.semilogy(op.xp, ki0[1][0,:] * u[0, : , 0] * op.np0, 'b-')
plt.xlabel(r"x")
plt.ylabel(r"$k_i n_e$ [$s^{-1}$]")
plt.grid()
plt.savefig("%s/%s_g2.png"%(folder_name, out_fname))
plt.close()


plt.figure(figsize=(8,8), dpi=200)
E = -np.dot(op.Dp, op.solve_poisson(u[0, :, 0], u[0, :, 1], 0))
plt.plot(op.xp, E * (op.V0/op.L), 'b-')
plt.grid(visible=True)
plt.xlabel(r"x")
plt.ylabel(r"E (V/m)")
plt.savefig("%s/%s_E.png"%(folder_name, out_fname))
plt.close()

#--------------------------------------------------------------------------------------------
plt.figure(figsize=(24, 8), dpi=200)
ev_grid=np.linspace(0,  100, 1024)
ff_r = compute_radial_components(d0[0], d0[3], spec_sp, ev_grid, d0[2])

print(ff_r[0, 1, 0])
print(d0[4][0, 1, 0])
import matplotlib as mpl
def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

c1='red' 
c2='blue' 
n=100

plt.subplot(1, 3, 1)
for i in range(0, 100, 5):
    plt.semilogy(ev_grid, np.abs(ff_r[0, i, 0]), color=colorFader(c1,c2,i/n))
#plt.xlim((0, 50))
plt.xlabel(r"eV")
plt.ylabel(r"$f_0$ $[eV^{-3/2}]$")
plt.grid(visible=True)


plt.subplot(1, 3, 2)
for i in range(0, 100, 5):
    plt.semilogy(ev_grid, np.abs(ff_r[0, i, 1]), color=colorFader(c1,c2,i/n))
#plt.xlim((0, 50))
plt.xlabel(r"eV")
plt.ylabel(r"$f_1$ $[eV^{-3/2}]$")
plt.grid(visible=True)

plt.subplot(1, 3, 3)
for i in range(0, 100, 5):
    plt.semilogy(ev_grid, np.abs(ff_r[0, i, 2]), color=colorFader(c1,c2,i/n))
#plt.xlim((0, 50))
plt.xlabel(r"eV")
plt.ylabel(r"$f_2$ $[eV^{-3/2}]$")
plt.grid(visible=True)
plt.savefig("%s/%s_eedf_avg.png"%(folder_name, out_fname))
plt.close()

#-------------------------------------------------------------------------------------------


plt.figure(figsize=(24, 8), dpi=200)
d1   = load_data_bte(folder_name, idx_range, list(np.arange(200)), read_cycle_avg=False)
ff_r = compute_radial_components(d1[0], d1[3], spec_sp, ev_grid, d1[2])

plt.subplot(1, 3, 1)
for i in range(0, 100, 5):
    plt.semilogy(ev_grid, np.abs(ff_r[0, i, 0]), color=colorFader(c1,c2,i/n))
plt.xlabel(r"eV")
plt.ylabel(r"$f_0$ $[eV^{-3/2}]$")
plt.grid(visible=True)


plt.subplot(1, 3, 2)
for i in range(0, 100, 5):
    plt.semilogy(ev_grid, np.abs(ff_r[0, i, 1]), color=colorFader(c1,c2,i/n))
plt.xlabel(r"eV")
plt.ylabel(r"$f_1$ $[eV^{-3/2}]$")
plt.grid(visible=True)

plt.subplot(1, 3, 3)
for i in range(0, 100, 5):
    plt.semilogy(ev_grid, np.abs(ff_r[0, i, 2]), color=colorFader(c1,c2,i/n))
plt.xlabel(r"eV")
plt.ylabel(r"$f_2$ $[eV^{-3/2}]$")
plt.grid(visible=True)
plt.savefig("%s/%s_eedf.png"%(folder_name, out_fname))
plt.close()
