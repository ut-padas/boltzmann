import numpy as np
import sys
import scipy.constants

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

col_names={"g0":"elastic", "g2":"ionization"}
collisons=["g0"]
idx_range = list(range((int)(sys.argv[2]), (int)(sys.argv[3]), (int)(sys.argv[4]) ))
if (int)(sys.argv[5])==1:
    collisons.append("g2")

d         = load_data_bte(sys.argv[1], idx_range, None, read_cycle_avg=False)
ki        = compute_rate_coefficients(d[0], d[3], d[2], collisons)
for g_idx, g in enumerate(collisons):
    np.save("%s/rates_%s.npy"%(sys.argv[1], col_names[g]),ki[g_idx])
    
    
ne=d[1][:, :, 0:2][:,:, 0]
Te=d[1][:, :, 2]
print(Te)
ne = ne[-1]
Te = Te[-1]
print("ne", np.min(ne[Te<2]), np.max(ne[Te<2]))
    
d         = load_data_bte(sys.argv[1], idx_range, None, read_cycle_avg=True)
ki        = compute_rate_coefficients(d[0], d[3], d[2], collisons)
for g_idx, g in enumerate(collisons):
    np.save("%s/rates_avg_%s.npy"%(sys.argv[1], col_names[g]),ki[g_idx])

np.save("%s/species_densities_avg.npy"%(sys.argv[1]), d[1][:, :, 0:2])
np.save("%s/Te_avg.npy"%(sys.argv[1]), d[1][:,:, 2])
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

