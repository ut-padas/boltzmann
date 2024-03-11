import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
import os
import scipy.interpolate

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
        
        ki   = np.genfromtxt("../Ar3species/Ar_1Torr_300K/Ionization.300K.txt" , delimiter=",", skip_header=True) # m^3/s
        mu_e = np.genfromtxt("../Ar3species/Ar_1Torr_300K/Mobility.300K.txt"   , delimiter=",", skip_header=True) # mu * n0 (1/(Vms))
        De   = np.genfromtxt("../Ar3species/Ar_1Torr_300K/Diffusivity.300K.txt", delimiter=",", skip_header=True) # D  * n0 (1/(ms))
        
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
        
        self.mu_fac   = (self.n0 * self.np0) / ( (self.V0 * self.tau/(self.L**2)) )
        self.D_fac    = (self.n0 * self.np0) / (self.tau/(self.L**2))
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

def make_dir(dir_name):
    # Check whether the specified path exists or not
    isExist = os.path.exists(dir_name)
    if not isExist:
       # Create a new directory because it does not exist
       os.makedirs(dir_name)
       print("directory %s is created!"%(dir_name))
    
def load_data(dir_name, num_cycles):
    a=list()
    for i in range(num_cycles):
        a.append(np.load(dir_name+"/1d_glow_%04d.npy"%(i)))
    return np.array(a)

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
    vth      = (float) (args["Te"]) * c_gamma
    v_lm     = np.array([np.dot(bte_op["po2sh"], v[idx]) for idx in range(n_pts)])
    scale    = np.array([np.dot(mm_op / mm_fac, v_lm[idx]) * (2 * (vth/c_gamma)**3) for idx in range(n_pts)])
    v_lm_n   = np.array([v_lm[idx]/scale[idx] for idx in range(n_pts)])
    num_sh   = (int) (args["l_max"]) + 1
    return [np.dot(bte_op[col_type], v_lm_n[:, 0::num_sh, :]) for col_type in collisions]



    
    
