import numpy as np
import matplotlib.pyplot as plt
import plot_utils 
from itertools import cycle
import scipy.constants
import scipy.interpolate

folder = "../1dglow_hybrid/1Torr300K_Ar_3sp2r_l2_cycle/qoi_analysis"

ne   = np.load ("../1dglow_hybrid/1Torr300K_Ar_3sp2r_l2_cycle/species_densities.npy")[:, :, 0]
Te   = np.load ("../1dglow_hybrid/1Torr300K_Ar_3sp2r_l2_cycle/Te.npy")

u_z      = np.load ("../1dglow_hybrid/1Torr300K_Ar_3sp2r_l2_cycle/u_z.npy")
C_mom_z  = np.load ("../1dglow_hybrid/1Torr300K_Ar_3sp2r_l2_cycle/C_mom_z.npy")
mueE     = np.load ("../1dglow_hybrid/1Torr300K_Ar_3sp2r_l2_cycle/mueE.npy")
De       = np.load ("../1dglow_hybrid/1Torr300K_Ar_3sp2r_l2_cycle/De.npy")
P        = np.load ("../1dglow_hybrid/1Torr300K_Ar_3sp2r_l2_cycle/E.npy")



tt   = np.linspace(0, 1 + 5e-3, ne.shape[0])
op   = plot_utils.op(ne.shape[1])

me    = scipy.constants.electron_mass
L     = op.L
dt    = 5e-5
io_dt = 5e-3


colors  = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

idx_left  =  op.xp < -0.5
idx_mid   =  np.logical_and(op.xp >= -0.5, op.xp < 0.5)
idx_right =  op.xp>=0.5

for idx in range(0, 201, 20):
    plt.figure(figsize=(16,6), dpi=200)    
    plt.subplot(1, 3, 1)
    
    plt.semilogy(op.xp[idx_left]  ,  np.abs(P[0][idx][idx_left])  , label = r"$E_{xx}$")
    plt.semilogy(op.xp[idx_left]  ,  np.abs(P[5][idx][idx_left])  , label = r"$E_{zz}$")
    plt.legend()
    plt.grid(visible=True)
    plt.xlabel(r"x")
    plt.xlabel(r"energy (eV)")

    
    plt.subplot(1, 3, 2)
    plt.semilogy(op.xp[idx_mid]   ,  np.abs(P[0][idx][idx_mid])   , label = r"$E_{xx}$")
    plt.semilogy(op.xp[idx_mid]   ,  np.abs(P[5][idx][idx_mid])   , label = r"$E_{zz}$")
    plt.legend()
    plt.grid(visible=True)
    plt.xlabel(r"x")
    plt.xlabel(r"energy (eV)")
    
    plt.subplot(1, 3, 3)
    plt.semilogy(op.xp[idx_right] ,  np.abs(P[0][idx][idx_right]) , label = r"$E_{xx}$")
    plt.semilogy(op.xp[idx_right] ,  np.abs(P[5][idx][idx_right]) , label = r"$E_{zz}$")
    plt.legend()
    plt.grid(visible=True)
    plt.xlabel(r"x")
    plt.xlabel(r"energy (eV)")
    
    plt.suptitle("time = %2.4f T"%(io_dt * idx))
    plt.tight_layout()
    plt.savefig("%s/qoi_%04d.png"%(folder, idx))
    plt.close()    


div_uuT_z = (1/L) * np.dot(op.Dp, (u_z**2 * ne).T).T / (C_mom_z * u_z * ne)

for idx in range(0, 201, 20):
    
    lbl = r"$\frac{Div(\vec{u}\vec{u}^T)} {C_{mom} u n}$"
    
    plt.figure(figsize=(8,8), dpi=200)
    plt.semilogy(op.xp  ,  np.abs(div_uuT_z[idx])  , label = lbl)
    plt.legend()
    plt.grid(visible=True)
    plt.xlabel(r"x")
    
    plt.savefig("%s/qoi_divU_full_%04d.png"%(folder, idx))
    plt.close()
    
    
    
    plt.figure(figsize=(16,6), dpi=200)
    plt.subplot(1, 3, 1)
    
    plt.semilogy(op.xp[idx_left]  ,  np.abs(div_uuT_z[idx][idx_left])  , label = lbl)
    plt.legend()
    plt.grid(visible=True)
    plt.xlabel(r"x")
    
    
    plt.subplot(1, 3, 2)
    plt.semilogy(op.xp[idx_mid]  ,  np.abs(div_uuT_z[idx][idx_mid])    , label = lbl)
    plt.legend()
    plt.grid(visible=True)
    plt.xlabel(r"x")
    
    
    plt.subplot(1, 3, 3)
    plt.semilogy(op.xp[idx_right]  ,  np.abs(div_uuT_z[idx][idx_right]) , label = lbl)
    plt.legend()
    plt.grid(visible=True)
    plt.xlabel(r"x")
    
    plt.suptitle("time = %2.4f T"%(io_dt * idx))
    plt.tight_layout()
    plt.savefig("%s/qoi_divU_%04d.png"%(folder, idx))
    plt.close()

plt.figure(figsize=(16, 16), dpi=200)
tt  = np.linspace(0, 1, 201)
tt1 = np.linspace(1e-6, 1-1e-6, 201)
for idx in range(0, len(op.xp)+1, 25):
    idx = min(idx, len(op.xp)-1)
    
    plt.subplot(2, 2, 1)
    spl = scipy.interpolate.UnivariateSpline(tt, u_z[:, idx] * ne[:, idx], k=3)
    #plt.plot(tt, ne[:, idx] * op.np0 * u_z[:, idx]        , label=r"$u_z n_e$(x=%.3f)"%(op.xp[idx]))
    plt.plot(tt1, (op.np0 / op.tau) * spl.derivative()(tt1)  , label=r"$\partial_t(u_z n_e)$(x=%.3f)"%(op.xp[idx]))
    plt.legend()
    plt.grid(visible=True)
    plt.xlabel(r"time t/T")
    plt.ylabel(r"electron flux rate[$m^{-2}s^{-2}$]")
    
    plt.subplot(2, 2, 2)
    plt.plot(tt, ne[:, idx] * op.np0 * u_z[:, idx] * C_mom_z[:, idx]  , label=r"$C_{mom} u_z n_e$(x=%.3f)"%(op.xp[idx]))
    plt.legend()
    plt.grid(visible=True)
    plt.xlabel(r"time t/T")
    plt.ylabel(r"electron flux rate [$m^2s^{-2}$]")
    
    plt.subplot(2, 2, 3)
    plt.semilogy(tt, np.abs(u_z[:, idx]), label=r"$u_z$(x=%.3f)"%(op.xp[idx]))
    plt.legend()
    plt.grid(visible=True)
    plt.xlabel(r"time t/T")
    plt.ylabel(r"velocity [$ms^{-1}$]")
    
    plt.subplot(2, 2, 4)
    #plt.semilogy(tt, ne[:, idx] * op.np0, label=r"$n_e$(x=%.3f)"%(op.xp[idx]))
    plt.plot(tt, ne[:, idx] * op.np0 * u_z[:, idx]         , label=r"$u_z n_e$(x=%.3f)"%(op.xp[idx]))
    plt.legend()
    plt.grid(visible=True)
    plt.xlabel(r"time t/T")
    plt.ylabel(r"number density [$m^{-3}$]")
    
plt.savefig("%s/qoi_dtneUz.png"%(folder))
plt.close()


#Je    = op.np0 * (mueE * ne - (De/L) * np.dot(op.Dp, ne.T).T)
Je    = op.np0 * (mueE * ne - (1 / L) * np.dot(op.Dp, (ne * De).T).T)
ne_ue = op.np0 * (u_z * ne )

for idx in range(0, 201, 20):
    plt.figure(figsize=(8,8), dpi=200)    
    
    plt.plot(op.xp,     Je[idx]  , label = r"$J_e$")
    plt.plot(op.xp,  ne_ue[idx]  , label = r"$n_e u_e$")
    plt.xlabel(r"x")
    plt.ylabel(r"flux [$m^{-2}s^{-1}$]")
    plt.grid(visible=True)
    plt.legend()
    
    plt.savefig("%s/qoi_eFlux_full_%04d.png"%(folder, idx))
    plt.close()
    
    
    plt.figure(figsize=(16,6), dpi=200)    

    plt.subplot(1, 3, 1)
    plt.semilogy(op.xp[idx_left]  ,  np.abs(   Je[idx][idx_left])  , label = r"$J_e$")
    plt.semilogy(op.xp[idx_left]  ,  np.abs(ne_ue[idx][idx_left])  , label = r"$n_e u_e$")
    plt.xlabel(r"x")
    plt.ylabel(r"flux [$m^{-2}s^{-1}$]")
    
    plt.grid(visible=True)
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.semilogy(op.xp[idx_mid]  ,  np.abs(   Je[idx][idx_mid])  , label = r"$J_e$")
    plt.semilogy(op.xp[idx_mid]  ,  np.abs(ne_ue[idx][idx_mid])  , label = r"$n_e u_e$")
    plt.grid(visible=True)
    plt.legend()
    plt.xlabel(r"x")
    plt.ylabel(r"flux [$m^{-2}s^{-1}$]")
    
    plt.subplot(1, 3, 3)
    plt.semilogy(op.xp[idx_right]  ,  np.abs(   Je[idx][idx_right])  , label = r"$J_e$")
    plt.semilogy(op.xp[idx_right]  ,  np.abs(ne_ue[idx][idx_right])  , label = r"$n_e u_e$")
    
    plt.grid(visible=True)
    plt.legend()
    plt.xlabel(r"x")
    plt.ylabel(r"flux [$m^{-2}s^{-1}$]")
    
    plt.suptitle("time = %2.4f T"%(io_dt * idx))
    plt.tight_layout()
    plt.savefig("%s/qoi_eFlux_%04d.png"%(folder, idx))
    plt.close()
    
# plt.figure(figsize=(6,6), dpi=200)
# plt.semilogy(op.xp  ,  np.abs(   Je[0])  , label = r"$J_e$")
# plt.semilogy(op.xp  ,  np.abs(ne_ue[0])  , label = r"$n_e u_e$")
# plt.savefig("t.png")
# plt.close()
    

# plt.legend()
# plt.grid(visible=True)
# plt.savefig("flux.png")
# plt.close()


