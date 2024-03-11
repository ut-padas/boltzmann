import numpy as np
import sys
import scipy.constants
import matplotlib.pyplot as plt
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

def compute_mobility(args, bte_op, v, EbyN):
    n_pts    = v.shape[0] 
    mm_fac   = np.sqrt(4 * np.pi) 
    mm_op    = bte_op["mass"]
    c_gamma  = np.sqrt(2 * (scipy.constants.elementary_charge/ scipy.constants.electron_mass))
    vth      = (float) (args["Te"])**0.5 * c_gamma
    v_lm     = np.array([np.dot(bte_op["po2sh"], v[idx]) for idx in range(n_pts)])
    scale    = np.array([np.dot(mm_op / mm_fac, v_lm[idx]) * (2 * (vth/c_gamma)**3) for idx in range(n_pts)])
    v_lm_n   = np.array([v_lm[idx]/scale[idx] for idx in range(n_pts)])
    num_sh   = (int) (args["l_max"]) + 1
    return np.dot(bte_op["mobility"], v_lm_n[:, 1::num_sh, :]) * (-(c_gamma / (3 * EbyN)))

def compute_diffusion(args, bte_op, v):
    n_pts    = v.shape[0] 
    mm_fac   = np.sqrt(4 * np.pi) 
    mm_op    = bte_op["mass"]
    c_gamma  = np.sqrt(2 * (scipy.constants.elementary_charge/ scipy.constants.electron_mass))
    vth      = (float) (args["Te"])**0.5 * c_gamma
    v_lm     = np.array([np.dot(bte_op["po2sh"], v[idx]) for idx in range(n_pts)])
    scale    = np.array([np.dot(mm_op / mm_fac, v_lm[idx]) * (2 * (vth/c_gamma)**3) for idx in range(n_pts)])
    v_lm_n   = np.array([v_lm[idx]/scale[idx] for idx in range(n_pts)])
    num_sh   = (int) (args["l_max"]) + 1
    return np.dot(bte_op["diffusion"], v_lm_n[:, 0::num_sh, :]) * (c_gamma / 3.)




col_names={"g0":"elastic", "g2":"ionization"}
collisons=["g0"]
folder_name    = sys.argv[1]
step_begin     = (int)(sys.argv[2])
step_end       = (int)(sys.argv[3])
step_sz        = (int)(sys.argv[4])
use_ionization = (int)(sys.argv[5])
dt_scaling_fac = (float)(sys.argv[6])
out_fname      = (sys.argv[7])
idx_range      = list(range(step_begin, step_end, step_sz))

if use_ionization ==1:
    collisons.append("g2")
    
op = plot_utils.op(200)    
d0         = load_data_bte(folder_name, idx_range, None, read_cycle_avg=False)
ki0        = compute_rate_coefficients(d0[0], d0[3], d0[2], collisons)
mu_e       = compute_mobility(d0[0], d0[3], d0[2], (1/(op.n0 * op.np0)))
De         = compute_diffusion(d0[0], d0[3], d0[2])

plt.figure(figsize=(36, 20), dpi=300)

u = d0[1]
v = d0[2]
colors    = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
ufluid    = np.load("../1dglow_fluid/r1_teb_flux/1d_glow_1000.npy")
Te_fluid  = ufluid[:,2]/ufluid[:,0]


plt.subplot(2, 3, 1)
plt.plot(op.xp, ufluid[:, 0] * op.np0, '--.', label=r"steady-state (fluid)")

plt.subplot(2, 3, 2)
plt.plot(op.xp, Te_fluid , '--.', label=r"steady-state (fluid)")

plt.subplot(2, 3, 3)
E = -np.dot(op.Dp, op.solve_poisson(ufluid[:, 0], ufluid[:, 1], 0.0))
plt.plot(op.xp, E * (op.V0/op.L), '--.', label=r"steady-state (fluid)")

plt.subplot(2, 3, 4)
plt.semilogy(op.xp, op.ki(Te_fluid, 1)    * op.r_fac, '--.' , label=r"0d-BTE ($T_e^{fluid}$)")
plt.semilogy(op.xp, op.ki(u[-1][:, 2], 1) * op.r_fac, '--o' , label=r"0d-BTE ($T_e^{BTE}$)")

plt.subplot(2, 3, 5)
E       = -np.dot(op.Dp, op.solve_poisson(u[-1][:, 0], u[-1][:, 1], 0.0))
Evm_mag = np.abs(E * op.V0/op.L)
plt.semilogy(op.xp, Evm_mag * op.mu_e(Te_fluid, 1)   * op.mu_fac, '--.', label=r"0d-BTE ($T_e^{fluid}$)")
plt.semilogy(op.xp, Evm_mag * op.mu_e(u[-1][:,2], 1) * op.mu_fac, '--o', label=r"0d-BTE ($T_e^{BTE}$)")

plt.subplot(2, 3, 6)
plt.semilogy(op.xp, op.De(Te_fluid   , 1) * op.D_fac, '--.', label=r"0d-BTE ($T_e^{fluid}$)")
plt.semilogy(op.xp, op.De(u[-1][:,2] , 1) * op.D_fac, '--o', label=r"0d-BTE ($T_e^{BTE}$)")

for idx, cycle in enumerate(idx_range):
    c = next(colors)
    plt.subplot(2, 3, 1)
    plt.plot(op.xp, u[idx][:, 0] * op.np0, label="t=%.2f T"%(cycle * dt_scaling_fac), color=c)
    #plt.semilogy(op.xp, (u[idx][:, 0] - u[max(0,idx-1)][:,0])/np.max(u[max(0,idx-1)][:,0]), label="t=%.2f T"%(cycle * dt_scaling_fac), color=c)
    plt.legend()
    plt.grid(visible=True)
    plt.xlabel(r"x")
    plt.ylabel(r"density ($m^{-3}$)")
    
    plt.subplot(2, 3, 2)
    plt.plot(op.xp, u[idx][:, 2], label="t=%.2f T"%(cycle * dt_scaling_fac), color=c)
    plt.legend()
    plt.grid(visible=True)
    plt.xlabel(r"x")
    plt.ylabel(r"temperature (eV)")
    
    plt.subplot(2, 3, 3)
    E = -np.dot(op.Dp, op.solve_poisson(u[idx][:, 0], u[idx][:, 1], cycle * dt_scaling_fac))
    plt.plot(op.xp, E * (op.V0/op.L), label="t=%.2f T"%(cycle * dt_scaling_fac), color=c)
    plt.legend()
    plt.grid(visible=True)
    plt.xlabel(r"x")
    plt.ylabel(r"E (V/m)")
    
    if idx==0:
        continue
    
    Evm     = E * op.V0/op.L
    Evm_mag = np.abs(E * op.V0/op.L)
    
    plt.subplot(2, 3, 4)
    plt.semilogy(op.xp, ki0[1][idx], label="t=%.2f T"%(cycle * dt_scaling_fac), color=c)
    plt.legend()
    plt.grid(visible=True)
    plt.xlabel(r"x")
    plt.ylabel(r"$k_{ion} (m^3s^{-1})$")
    
    plt.subplot(2, 3, 5)
    plt.semilogy(op.xp, Evm_mag * np.abs(mu_e[idx])/Evm_mag         , label="t=%.2f T"%(cycle * dt_scaling_fac), color=c)
    
    plt.legend()
    plt.grid(visible=True)
    plt.xlabel(r"x")
    #plt.ylabel(r"$n_0 \mu_{e} (V^{-1} m^{-1} s^{-1} )$")
    plt.ylabel(r"$n_0|E|\mu_{e} (s^{-1} )$")
    
    plt.subplot(2, 3, 6)
    plt.semilogy(op.xp, (De[idx]), label="t=%.2f T"%(cycle * dt_scaling_fac), color=c)
    plt.legend()
    plt.grid(visible=True)
    plt.xlabel(r"x")
    plt.ylabel(r"$n_0 D_{e} (m^{-1} s^{-1})$")
    
# plt.subplot(2, 3, 4)

# plt.legend()
# print(op.ki(u[idx, : , 2], 1))

plt.savefig("%s.png"%(sys.argv[7]))


# op = plot_utils.op(200)

# plt.figure(figsize=(24, 10), dpi=200)
# for idx in range(0, (int)(sys.argv[2])):
#     u0 =d0[1]
#     u1 =d1[1]
#     print(u0.shape)
    
#     plt.subplot(1, 3, 1)
#     plt.plot(op.xp, u0[idx][:,0] * op.np0, '-', label="t=%.2E"%(idx * 0.1))
#     plt.plot(op.xp, u1[idx][:,0] * op.np0, '--', label="t=%.2E"%(idx * 0.1))
#     plt.legend()
#     plt.grid(visible=True)
#     plt.xlabel(r"x")
#     plt.ylabel(r"density ($m^{-3}$)")
    
#     plt.subplot(1, 3, 2)
#     plt.plot(op.xp, u0[idx][:,2], '-', label="t=%.2E"%(idx * 0.1))
#     plt.plot(op.xp, u1[idx][:,2], '--', label="t=%.2E"%(idx * 0.1))
#     plt.legend()
#     plt.grid(visible=True)
#     plt.xlabel(r"x")
#     plt.ylabel(r"electron temperature ($eV$)")
    
#     plt.subplot(1, 3, 3)
#     plt.semilogy(op.xp, ki0[1][idx], '-', label="t=%.2E"%(idx * 0.1))
#     plt.semilogy(op.xp, ki1[1][idx], '--', label="t=%.2E"%(idx * 0.1))
#     plt.legend()
#     plt.grid(visible=True)
#     plt.xlabel(r"x")
#     plt.ylabel(r"reaction rates ($k_i (m^3s^{-1})$)")
    
# plt.savefig("t.png")