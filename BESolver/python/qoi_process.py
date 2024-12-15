import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
import os
import scipy.interpolate
import h5py
import sys

class op():
    def __init__(self,Np):
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
      
      self.L     = 0.5 * 2.54e-2             # m 
      self.V0    = 1e2                       # V
      self.f     = 13.56e6                   # Hz
      self.tau   = (1/self.f)                # s
      self.qe    = scipy.constants.e         # C
      self.eps0  = scipy.constants.epsilon_0 # eps_0 
      self.kB    = scipy.constants.Boltzmann # J/K
      self.ev_to_K = scipy.constants.electron_volt / scipy.constants.Boltzmann
      self.me    = scipy.constants.electron_mass
        
      self.np0   = 8e16                      #"nominal" electron density [1/m^3]
      self.n0    = 3.22e22                   #m^{-3}
      
      
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
        # ki   = np.genfromtxt("../Ar3species/Ar_n03.22e22_0K/Ionization.0K.txt" , delimiter=",", skip_header=True) # m^3/s
        # mu_e = np.genfromtxt("../Ar3species/Ar_n03.22e22_0K/Mobility.0K.txt"   , delimiter=",", skip_header=True) # mu * n0 (1/(Vms))
        # De   = np.genfromtxt("../Ar3species/Ar_n03.22e22_0K/Diffusivity.0K.txt", delimiter=",", skip_header=True) # D  * n0 (1/(ms))
        
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

def compute_E(ne, ni, tt):
    cheb = op(ne.shape[1])
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

# which folder to analyze
print("args: ", sys.argv)    
folder_name = sys.argv[1] 

if (sys.argv[2] == "bte"):
    ne          = np.load("%s/species_densities.npy"%(folder_name))[:, :, 0]    
    ni          = np.load("%s/species_densities.npy"%(folder_name))[:, :, 1]
    Te          = np.load("%s/Te.npy"%(folder_name))
    ke          = np.load("%s/rates_elastic.npy" %(folder_name))
    ki          = np.load("%s/rates_ionization.npy" %(folder_name))
    tt          = np.linspace(0, 1, ne.shape[0])
    EF          = compute_E(ne, ni, tt)
    cheb        = op(ne.shape[1])
    
elif (sys.argv[2] == "fluid"):
    U           = np.array([np.load("%s/1d_glow_%04d.npy"%(folder_name,idx)) for idx in range(0, 101)])
    ne          = U[:, :, 0]    
    ni          = U[:, :, 1]
    Te          = U[:, :, 2] / U[:, :, 0]
    tt          = np.linspace(0, 1, ne.shape[0])
    EF          = compute_E(ne, ni, tt)
    cheb        = op(ne.shape[1])
    ki          = np.array([cheb.ki(Te[i], 1) * cheb.r_fac for i in range(len(tt))])
else:
    raise NotImplementedError

# # put your xt grid parameters here. 
# xx_gird  = np.linspace(-1 , 1, 600)
# tt_grid  = np.linspace( 0 , 2, 401)
# EFtx     = Etx_interpolate(EF, tt_grid, xx_gird, tt, None)

# io_feq     = 100 # every 100 cycle
# num_cycles = 2 # number of cycles

#tt_avg      = np.linspace(0, 1, io_feq+1)

# ion_prod   = np.array([time_average(cheb.np0 * cheb.n0 * ki[0   : 101, :]   * ne[0   : 101 , :] * cheb.np0, tt_avg), 
#                        time_average(cheb.np0 * cheb.n0 * ki[100 : 201, :]   * ne[100 : 201 , :] * cheb.np0, tt_avg)])


with h5py.File("%s/macro.h5"%(folder_name), 'w') as F:
    xc = cheb.xp
    F.create_dataset("time[T]"      , data = tt)
    F.create_dataset("x[-1,1]"      , data = xc)
    F.create_dataset("E[Vm^-1]"     , data = EF)
    F.create_dataset("ne[m^-3]"     , data = cheb.np0 * ne)
    F.create_dataset("ni[m^-3]"     , data = cheb.np0 * ni)
    F.create_dataset("Te[eV]"		, data = Te)
    F.create_dataset("ki[m^3s^{-1}]", data = ki)
    F.create_dataset("ke[m^3s^{-1}]", data = ke)
    
    F.create_dataset("avg_E[Vm^-1]"     , data = time_average(EF , tt))
    F.create_dataset("avg_ne[m^-3]"     , data = time_average(cheb.np0 * ne , tt))
    F.create_dataset("avg_ni[m^-3]"     , data = time_average(cheb.np0 * ni , tt))
    F.create_dataset("avg_Te[eV]"		, data = time_average(Te , tt))
    F.create_dataset("avg_ki[m^3s^{-1}]"  , data = time_average(ki , tt))
    F.create_dataset("avg_ke[m^3s^{-1}]"  , data = time_average(ke , tt))
    
    F.create_dataset("avg_energy_density[eVkgm^{-3}]", data = 1.5 * cheb.np0 * scipy.constants.electron_mass * time_average(Te * ne, tt))
    F.create_dataset("avg_ion_prod[m^-3s^{-1}]"      , data = cheb.n0 * cheb.np0 **2 * time_average(ki * ne, tt))
    F.create_dataset("avg_elastic [m^-3s^{-1}]"      , data = cheb.n0 * cheb.np0 **2 * time_average(ke * ne, tt))
    
    F.close()

#     gf = F.create_group("forward[m^3s^-1]")
#     for i in range(rf.shape[1]):
#         gf.create_dataset(eqc_name[process[i]], data=rf[:, i])
    
#     gf = F.create_group("backward[m^6s^-1]")
#     for i in range(rf.shape[1]):
#         gf.create_dataset(eqc_name[process[i]], data=rb[:, i])
        
#     F.close()

#np.save("%s/electric_field_tx.npy"%(folder_name), EFtx)
#np.save("%s/ion_production_rate_ca.npy"%(folder_name), ion_prod)

#plt.semilogy(cheb.xp, ion_prod)
#plt.show()



# xp = op(ne.shape[1]).xp
# for i in range(0, len(tt), 25):
#     plt.plot(xp     , EF[i]    , label=r"time = %.4E T "%(tt[i]))
# plt.grid(visible=True)
# plt.legend()
# plt.show()
# plt.close()


# for i in range(0, len(tt), 25):
#     plt.semilogy(xp     , ne[i]    , label=r"time = %.4E T "%(tt[i]))
# plt.grid(visible=True)
# plt.legend()
# plt.show()
# plt.close()

ff = h5py.File("%s/macro.h5"%(folder_name))
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
plt.show()
plt.close()