import numpy as np
import sys
import scipy.constants
import cross_section
import collisions
import utils as bte_utils
import basis
import spec_spherical as sp

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
    bte_op["mass"]      = np.load(folder+"/1d_glow_bte_mass_op.npy")
    bte_op["temp"]      = np.load(folder+"/1d_glow_bte_temp_op.npy")
    bte_op["mobility"]  = np.load(folder+"/1d_glow_bte_mobility.npy")
    bte_op["diffusion"] = np.load(folder+"/1d_glow_bte_diffusion.npy")
    bte_op["cmat"]      = np.load(folder+"/1d_glow_bte_cmat.npy")
    
    
    bte_op["g0"]    = np.load(folder+"/1d_glow_bte_op_g0.npy")
    
    if (int)(sys.argv[5])==1:
        bte_op["g2"]    = np.load(folder+"/1d_glow_bte_op_g2.npy")
    
    bte_op["po2sh"] = np.load(folder+"/1d_glow_bte_po2sh.npy")
    bte_op["psh2o"] = np.load(folder+"/1d_glow_bte_psh2o.npy")
    
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

def compute_mean_velocity(spec_sp, args, bte_op, v):
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
    
    n0      = float(sys.argv[6])
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

col_names={"g0":"elastic", "g2":"ionization"}
collisons=["g0"]
idx_range = list(range((int)(sys.argv[2]), (int)(sys.argv[3]), (int)(sys.argv[4]) ))
if (int)(sys.argv[5])==1:
    collisons.append("g2")
n0        = float(sys.argv[6])
print(n0)

d         = load_data_bte(sys.argv[1], idx_range, None, read_cycle_avg=False)
ki        = compute_rate_coefficients(d[0], d[3], d[2], collisons)

spec_sp, coll_list = gen_spec_sp(d[0])
u , Cu    = compute_mean_velocity(spec_sp,d[0], d[3], d[2])
mueE, De  = compute_diffusion_and_effective_mobility(d[0], d[3], d[2], n0)
P         = compute_kinetic_energy_tensor(spec_sp, d[0], d[3], d[2])

for g_idx, g in enumerate(collisons):
    np.save("%s/rates_%s.npy"%(sys.argv[1], col_names[g]),ki[g_idx])

np.save("%s/species_densities.npy"%(sys.argv[1]), d[1][:, :, 0:2])
np.save("%s/Te.npy"%(sys.argv[1]), d[1][:,:, 2])

np.save("%s/u_x.npy"%(sys.argv[1]), u[0])
np.save("%s/u_y.npy"%(sys.argv[1]), u[1])
np.save("%s/u_z.npy"%(sys.argv[1]), u[2])

np.save("%s/C_mom_x.npy"%(sys.argv[1]), Cu[0])
np.save("%s/C_mom_y.npy"%(sys.argv[1]), Cu[1])
np.save("%s/C_mom_z.npy"%(sys.argv[1]), Cu[2])


np.save("%s/mueE.npy"%(sys.argv[1]), mueE)
np.save("%s/De.npy"%(sys.argv[1])  , De)
np.save("%s/E.npy"%(sys.argv[1])  , P)

# ne=d[1][:, :, 0:2][:,:, 0]
# Te=d[1][:, :, 2]
# #print(Te)
# ne = ne[-1]
# Te = Te[-1]
# #print("ne", np.min(ne[Te<2]), np.max(ne[Te<2]))
    
d         = load_data_bte(sys.argv[1], idx_range, None, read_cycle_avg=True)
ki        = compute_rate_coefficients(d[0], d[3], d[2], collisons)
u , Cu    = compute_mean_velocity(spec_sp,d[0], d[3], d[2])
P         = compute_kinetic_energy_tensor(spec_sp, d[0], d[3], d[2])
for g_idx, g in enumerate(collisons):
    np.save("%s/rates_avg_%s.npy"%(sys.argv[1], col_names[g]),ki[g_idx])

np.save("%s/species_densities_avg.npy"%(sys.argv[1]), d[1][:, :, 0:2])
np.save("%s/Te_avg.npy"%(sys.argv[1]), d[1][:,:, 2])

np.save("%s/u_x_avg.npy"%(sys.argv[1]), u[0])
np.save("%s/u_y_avg.npy"%(sys.argv[1]), u[1])
np.save("%s/u_z_avg.npy"%(sys.argv[1]), u[2])

np.save("%s/C_mom_x_avg.npy"%(sys.argv[1]), Cu[0])
np.save("%s/C_mom_y_avg.npy"%(sys.argv[1]), Cu[1])
np.save("%s/C_mom_z_avg.npy"%(sys.argv[1]), Cu[2])

np.save("%s/mueE_avg.npy"%(sys.argv[1]), mueE)
np.save("%s/De_avg.npy"%(sys.argv[1])  , De)
np.save("%s/E_avg.npy"%(sys.argv[1])  , P)

Np = int(d[0]["Np"])
import matplotlib.pyplot as plt
from plot_scripts import plot_utils
xp        = -np.cos(np.pi*np.linspace(0,Np-1,Np) / (Np-1))
ki_g0_avg = np.load("%s/rates_avg_%s.npy"%(sys.argv[1], col_names["g0"]))
ki_g2_avg = np.load("%s/rates_avg_%s.npy"%(sys.argv[1], col_names["g2"]))
ne_avg    = np.load("%s/species_densities_avg.npy"%(sys.argv[1]))[:, : , 0]
ni_avg    = np.load("%s/species_densities_avg.npy"%(sys.argv[1]))[:, : , 1]
Te_avg    = np.load("%s/Te_avg.npy"%(sys.argv[1]))

plt.figure(figsize=(8, 8), dpi=300)
plt.subplot(2, 2, 1)
plt.semilogy(xp, ne_avg[-1] * 8e16, label=r"$n_e$")
plt.semilogy(xp, ni_avg[-1] * 8e16, label=r"$n_i$")
plt.xlabel(r"2x/L -1")
plt.ylabel(r"number density [$m^{-3}$]")
plt.grid()
plt.legend()

plt.subplot(2, 2, 2)
plt.semilogy(xp, ki_g2_avg[-1], label=r"ionization")
plt.semilogy(xp, ki_g0_avg[-1], label=r"elastic")
plt.xlabel(r"2x/L -1")
plt.ylabel(r"reaction rate [$m^3s^{-1}$]")
plt.grid()
plt.legend()

plt.subplot(2, 2, 3)
plt.semilogy(xp, ki_g2_avg[-1] * ne_avg[-1] * 8e16, label=r"ionization")
plt.semilogy(xp, ki_g0_avg[-1] * ne_avg[-1] * 8e16, label=r"elastic")
plt.xlabel(r"2x/L -1")
plt.ylabel(r"production rate ($k_i n_e$) [$s^{-1}$]")
plt.grid()
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(xp, Te_avg[-1])
plt.xlabel(r"2x/L -1")
plt.ylabel(r"temperature [eV]")
plt.grid()

plt.suptitle(r"$V_0$=100V f=13.56MHz, 3 species model with elastic + ionization")
plt.tight_layout()
plt.savefig("%s/test.png"%(sys.argv[1]))
plt.close()

