import numpy as np
import matplotlib.pyplot as plt
import plot_utils

u1 = np.load("../1dglow_fluid/r1_teb_fixed/1d_glow_1000_fluid.npy")
u2 = np.load("../1dglow_fluid/r1_teb_flux/1d_glow_1000.npy")

op=plot_utils.op(200)

plt.figure(figsize=(16, 6), dpi=200)
plt.subplot(1, 3, 1)
plt.plot(op.xp, u1[:, 0] * op.np0, label=r"$T_{eb} = 1.5 eV$")
plt.plot(op.xp, u2[:, 0] * op.np0, label=r"$2.5 T_{eb}\Gamma_{e,wall} = Q_{e,wall}$")
plt.xlabel(r"x")
plt.ylabel(r"$n_e (m^{-3})$")
plt.grid()
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(op.xp, u1[:, 2]/u1[:,0] , label=r"$T_{eb} = 1.5 eV$")
plt.plot(op.xp, u2[:, 2]/u2[:,0] , label=r"$2.5 T_{eb}\Gamma_{e,wall} = Q_{e,wall}$")
plt.grid()
plt.legend()
plt.xlabel(r"x")
plt.ylabel(r"$T_e$ (eV)")

E1  = np.dot(op.Dp, -op.solve_poisson(u1[:,0], u1[:,1], 0)) * (op.V0/op.L)
E2  = np.dot(op.Dp, -op.solve_poisson(u2[:,0], u2[:,1], 0)) * (op.V0/op.L)
plt.subplot(1, 3, 3)
plt.plot(op.xp, E1 , label=r"$T_{eb} = 1.5 eV$")
plt.plot(op.xp, E2 , label=r"$2.5 T_{eb}\Gamma_{e,wall} = Q_{e,wall}$")
plt.grid()
plt.legend()
plt.xlabel(r"x")
plt.ylabel(r"E (V/m)")
plt.tight_layout()
plt.savefig("fluid_teb.png")
