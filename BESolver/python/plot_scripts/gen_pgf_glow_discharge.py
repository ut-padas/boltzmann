import numpy as np
import matplotlib.pyplot as plt
import plot_utils
import scipy.constants
import sys
from matplotlib.colors import LogNorm

sys.path.append("../.")
import collisions
import spec_spherical as sp

qe_by_k  = scipy.constants.elementary_charge/ scipy.constants.Boltzmann

fname    = "../../../../papers/boltzmann1d-paper/dat/glow_discharge/1Torr300K"

def load_hybrid_cycle_data():
      data = dict()
      fprefix = "../1dglow_hybrid/1Torr300K_Ar_3sp2r_l2_cycle"
      tmp                = np.load("%s/species_densities.npy"%(fprefix))
      
      nT                 = tmp.shape[0]
      tt                 = np.linspace(0, 1, nT)
      T                  = (tt[-1]-tt[0])
      dt                 = T/(nT-1)
      
      assert abs(tt[1] -tt[0] - dt) < 1e-10
    
      tw    = np.ones_like(tt) * dt
      tw[0] = 0.5 * tw[0]; tw[-1] = 0.5 * tw[-1];
    
      assert (T-np.sum(tw)) < 1e-12
      
      op                 = plot_utils.op(tmp.shape[1])
      data["tt"]         = tt
      data["tw"]         = tw
      
      data["ne"]         = tmp[:, :, 0]
      data["ni"]         = tmp[:, :, 1] 
      data["Te"]         = np.load("%s/Te.npy"%(fprefix))
      data["ionization"] = np.load("%s/rates_ionization.npy"%(fprefix))
      data["De"]         = np.load("%s/De.npy"%(fprefix))
      data["mueE"]       = np.load("%s/mueE.npy"%(fprefix))
      
      data["op"]         = op
      
      data["phi"]            = np.array([op.solve_poisson(data["ne"][i], data["ni"][i], data["tt"][i]) for i in range(nT)]) * op.V0
      data["E"]              = -np.dot(op.Dp, data["phi"].T).T * (1 / op.L)
      data["energy_density"] = data["ne"] * data["Te"] * 1.5 * scipy.constants.electron_mass * op.np0
      data["production"]     = data["ne"] * op.np0 * op.n0 * op.np0 * data["ionization"]
      
      return data

def load_fluid_cycle_data():
      data = dict()
      uu   = np.load("../1dglow_fluid/1Torr300K_Ar_3sp2r_tab_cycle/1d_glow_cycle.npy")
      
      nT                 = uu.shape[0]
      tt                 = np.linspace(0, 1, nT)
      T                  = (tt[-1]-tt[0])
      dt                 = T/(nT-1)
      
      assert abs(tt[1] -tt[0] - dt) < 1e-10
    
      tw    = np.ones_like(tt) * dt
      tw[0] = 0.5 * tw[0]; tw[-1] = 0.5 * tw[-1];
    
      assert (T-np.sum(tw)) < 1e-12
      
      data["tt"]             = tt
      data["tw"]             = tw
      
      data["ne"]             = uu[:, :, 0]
      data["ni"]             = uu[:, :, 1] 
      data["Te"]             = uu[:, :, 2] / uu[:, :, 0]
      Te                     = data["Te"]
      
      op                     = plot_utils.op(uu.shape[1])
      
      data["ionization"]     = np.array([op.ki(Te[i],1)   * op.r_fac  for i in range(nT)])
      
      data["phi"]            = np.array([op.solve_poisson(data["ne"][i], data["ni"][i], data["tt"][i]) for i in range(nT)]) * op.V0
      data["E"]              = -np.dot(op.Dp, data["phi"].T).T * (1 / op.L)
      data["energy_density"] = data["ne"] * data["Te"] * 1.5 * scipy.constants.electron_mass * op.np0
      data["production"]     = data["ne"] * op.np0 * op.n0 * op.np0 * data["ionization"]
      
      
      data["De"]             = np.array([op.De(Te[i],1)    * op.D_fac         for i in range(nT)])
      data["mueE"]           = np.array([op.mu_e(Te[i],1)  * op.mu_fac *  data["E"][i] for i in range(nT)])
      data["op"]             = op
      
      return data
      
      
# writing the radial components

data              = plot_utils.load_data_bte("../../../../bte-docs/1dglow/Ar_3_species_100V/",range(1900, 1901, 1), None, read_cycle_avg=True)
spec_sp, col_list = plot_utils.gen_spec_sp(data[0])

args             = data[0]
op               = plot_utils.op(int(args["Np"]))
c_gamma          = np.sqrt(2 * scipy.constants.electron_volt / scipy.constants.electron_mass)
Te               = (float)(args["Te"])
vth              = collisions.electron_thermal_velocity(Te * (scipy.constants.elementary_charge / scipy.constants.Boltzmann))
ev               = np.linspace(1e-2, 30, 256)
fr               = plot_utils.compute_radial_components(data[0], data[3], spec_sp, ev, data[2])

plt.figure(figsize=(14, 4), dpi=200)
for xidx in range(np.argmin(np.abs(op.xp)) + 1 , len(op.xp), 10):
      for l in range(0, 3):
            plt.subplot(1, 3, l+1)
            plt.semilogy(ev, np.abs(fr[0][xidx][l]), label=r"x=%.2f"%(op.xp[xidx]))
            
            
for l in range(0, 3):
      plt.subplot(1, 3, l+1)
      plt.ylim((1e-8,None))
      plt.xlabel(r"energy [eV]")
      plt.ylabel(r"|f_%d| [$eV^{3/2}$]"%(l))
      plt.grid(visible=True)
      plt.legend()
                  
plt.show()
plt.close()


for xidx in range(np.argmin(np.abs(op.xp)) + 1 , len(op.xp), 10):
      np.savetxt("%s_radial_comp_x_%.3f.csv"%(fname,op.xp[xidx]),np.array([ev, fr[0][xidx][0], fr[0][xidx][1], fr[0][xidx][2]],dtype=np.float64).T, 
                 header="ev\tf0\tf1\tf2\tf3",comments="", delimiter="\t")      
      


d0 = load_hybrid_cycle_data()
d1 = load_fluid_cycle_data()




plt.figure(figsize=(16, 4), dpi=100)
plt.subplot(1, 3, 1)
plt.plot(d0["op"].xp, plot_utils.time_average(d0["ne"],d0["tt"]) * d0["op"].np0, label=r"hybrid")
plt.plot(d1["op"].xp, plot_utils.time_average(d1["ne"],d1["tt"]) * d1["op"].np0, label=r"fluid")

plt.xlabel(r"x")
plt.ylabel(r"electron number density [$m^{-3}$]")
plt.legend()
plt.grid(visible=True)

plt.subplot(1, 3, 2)
plt.plot(d0["op"].xp, plot_utils.time_average(d0["energy_density"],d0["tt"]), label=r"hybrid")
plt.plot(d1["op"].xp, plot_utils.time_average(d1["energy_density"],d1["tt"]), label=r"fluid")


plt.xlabel(r"x")
plt.ylabel(r"energy density [$eV Kg m^{-3}$]")
plt.legend()
plt.grid(visible=True)

plt.subplot(1, 3, 3)
plt.semilogy(d0["op"].xp, plot_utils.time_average(d0["production"],d0["tt"]), label=r"hybrid")
plt.semilogy(d1["op"].xp, plot_utils.time_average(d1["production"],d1["tt"]), label=r"fluid")
plt.xlabel(r"x")
plt.ylabel(r"ionization production [$s^{-1}$]")
plt.legend()
plt.grid(visible=True)



plt.tight_layout()
plt.show()
plt.close()

d    = d0
xx   = d["op"].xp
ne   = plot_utils.time_average(d["ne"] , d["tt"]) * d["op"].np0
ni   = plot_utils.time_average(d["ni"] , d["tt"]) * d["op"].np0
Te   = plot_utils.time_average(d["Te"] , d["tt"]) 
pE   = plot_utils.time_average(d["energy_density"] , d["tt"]) 
pp   = plot_utils.time_average(d["production"]     , d["tt"]) 
E    = plot_utils.time_average(d["E"]     , d["tt"]) 
phi  = plot_utils.time_average(d["phi"]   , d["tt"]) 
mueE = plot_utils.time_average(d["mueE"]  , d["tt"]) 
De   = plot_utils.time_average(d["De"]    , d["tt"]) 
np.savetxt("%s_bte_ca.csv"%(fname), np.array([xx, ne, ni, Te, pE, pp, E, phi, mueE, De] , dtype=np.float64).T, 
      header="x\tne\tni\tTe\tenergy_density\tion_production\tE\tphi\tmueE\tDe",comments="", delimiter="\t")


d    = d1
xx   = d["op"].xp
ne   = plot_utils.time_average(d["ne"] , d["tt"]) * d["op"].np0
ni   = plot_utils.time_average(d["ni"] , d["tt"]) * d["op"].np0
Te   = plot_utils.time_average(d["Te"] , d["tt"]) 
pE   = plot_utils.time_average(d["energy_density"] , d["tt"]) 
pp   = plot_utils.time_average(d["production"]     , d["tt"]) 
E    = plot_utils.time_average(d["E"]     , d["tt"]) 
phi  = plot_utils.time_average(d["phi"]   , d["tt"]) 
mueE = plot_utils.time_average(d["mueE"]  , d["tt"]) 
De   = plot_utils.time_average(d["De"]    , d["tt"]) 
np.savetxt("%s_fluid_tab_ca.csv"%(fname), np.array([xx, ne, ni, Te, pE, pp, E, phi, mueE, De] , dtype=np.float64).T, 
      header="x\tne\tni\tTe\tenergy_density\tion_production\tE\tphi\tmueE\tDe",comments="", delimiter="\t")

sys.exit(0)

u0       = np.load("../../../../bte-docs/1dglow/Ar_3_species_100V/1d_glow_1900_u.npy")
mu_eE    = np.load("../../../../bte-docs/1dglow/Ar_3_species_100V/mueE.npy")[-1,:]
u_z      = np.load("../../../../bte-docs/1dglow/Ar_3_species_100V/u_z.npy")[-1,:]
De       = np.load("../../../../bte-docs/1dglow/Ar_3_species_100V/De.npy")[-1,:]

u1       = np.load("../../../../bte-docs/1dglow/fluid_Ar_3_species_100V/1d_glow_0008.npy")
u1[:, 2] = u1[:, 2]/u1[:, 0]

u2       = np.load("../../../../bte-docs/1dglow/1Torr300K_Ar_3sp2r_liu/1d_glow_1300.npy")
u2[:, 2] = u2[:, 2]/u2[:, 0]

E1       = -np.dot(op.Dp , op.solve_poisson(u1[:, 0], u1[:, 1], 0.0)) * op.V0/op.L # V/m
E2       = -np.dot(op.Dp , op.solve_poisson(u2[:, 0], u2[:, 1], 0.0)) * op.V0/op.L # V/m


ne          = u0[:, 0] * op.np0
ni          = u0[:, 1] * op.np0 
Te          = u0[:, 2]
E_density   = ne * op.np0 * Te * (1.5 * qe_by_k * scipy.constants.Boltzmann) * scipy.constants.electron_mass
prod        = ki0 * ne * op.np0
phi         = op.solve_poisson(ne/op.np0, ni/op.np0, 0) * op.V0 # [V]
E           = np.dot(op.Dp, -op.solve_poisson(ne/op.np0, ni/op.np0, 0)) * op.V0 / op.L

plt.figure(figsize=(16, 4), dpi=100)
plt.subplot(1, 3, 1)
plt.plot(op.xp,  mu_eE                                 ,       label = r"hybrid")
plt.plot(op.xp,  op.mu_e(u1[:, 2], 1) * op.mu_fac * E1 ,       label = r"fluid (tabulated)")
plt.plot(op.xp, mu_e2 * op.mu_fac * E2                ,       label = r"fluid (A)")
plt.xlabel(r"x")
plt.ylabel(r"$\mu_e E $ [$ms^{-1}$]")
plt.grid(visible=True)
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(op.xp, De,                              label = r"hybrid")
plt.plot(op.xp, op.De(u1[:, 2], 1) * op.D_fac ,  label = r"fluid (tabulated)")
plt.plot(op.xp, De2 * np.ones_like(op.xp) * op.D_fac ,  label = r"fluid (A)")
plt.xlabel(r"x")
plt.ylabel(r"$D_e$ [$m^2s^{-1}$]")
plt.grid(visible=True)
plt.legend()

Je1 = op.mu_e(u1[:, 2], 1) * op.mu_fac  * u1[:, 0] - op.De(u1[:, 2], 1) * op.D_fac * (1/op.L) * np.dot(op.Dp, u1[:, 0])
Je2 = mu_e2 * op.mu_fac *  u2[:, 0] * E2  - De2 * op.D_fac * (1/op.L) * np.dot(op.Dp, u2[:, 0])

u_z = np.abs(u_z)
Je1 = np.abs(Je1)
Je2 = np.abs(Je2)

plt.subplot(1, 3, 3)
plt.semilogy(op.xp, u_z * u0[:, 0] * op.np0  ,   label = r"hybrid")
plt.semilogy(op.xp, Je1 * op.np0             ,   label = r"fluid(tabulated)")
plt.semilogy(op.xp, Je2 * op.np0             ,   label = r"fluid(A)")
plt.grid(visible=True)
plt.xlabel(r"x")
plt.ylabel(r"electron flux [$m^{-2}s^{-1}$]")
plt.legend()

plt.show()
plt.close()

