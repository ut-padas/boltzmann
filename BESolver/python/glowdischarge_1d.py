"""
macroscopic/microscopic modeling of the 1d glow discharge problem
1). We use Gauss-Chebyshev-Lobatto co-location method with implicit time integration. 
"""
import numpy as np
import scipy.constants 
import argparse
import matplotlib.pyplot as plt
import sys
import glow1d_utils
import os
import scipy.optimize
import scipy.interpolate
np.seterr(divide='ignore', invalid='ignore', over='ignore')
#np.seterr(all=='raise')

try:
  import cupy as cp
  #CUDA_NUM_DEVICES=cp.cuda.runtime.getDeviceCount()
except ImportError:
  print("Please install CuPy for GPU use")
  #sys.exit(0)
except:
  print("CUDA not configured properly !!!")
  sys.exit(0)

class state_idx():
  electron_idx        = 0
  ionized_Ar_idx      = 1
  electron_temp       = 2

class glow1d_fluid():
    def __init__(self, args) -> None:
      self.args  = args
      
      dir            = args.dir
      if (dir!=""):
        if (os.path.exists(dir)):
          print("run directory exists, data will be overwritten")
          #sys.exit(0)
        else:
          os.makedirs(dir)
          print("directory %s created"%(dir))
      
        args.fname=str(dir)+"/"+args.fname
      
      with open("%s_args.txt"%(args.fname), "w") as ff:
        ff.write("args: %s"%(args))
        ff.close()
      
      self.param = glow1d_utils.parameters(args)
      
      self.Ns = self.args.Ns                   # Number of species
      self.NT = self.args.NT                   # Number of temperatures
      self.Nv = self.args.Ns + self.args.NT    # Total number of 'state' variables

      self.deg = self.args.Np-1  # degree of Chebyshev polys we use
      self.Np  = self.args.Np    # Number of points used to define state in space
      self.Nc  = self.args.Np-2  # number of collocation pts (Np-2 b/c BCs)
      
      self.ele_idx = state_idx.electron_idx
      self.ion_idx = state_idx.ionized_Ar_idx
      self.Te_idx  = state_idx.electron_temp
      
      self.kB   = scipy.constants.Boltzmann
      
      # charge number
      self.Zp    = np.zeros(self.Ns)
      self.Zp[0] = -1 # electrons are always -1
      self.Zp[1] =  1 # ions are always 1
      
      # mobility
      self.mu = np.zeros((self.Np , self.Ns))
      # diffusivity
      self.D  = np.zeros((self.Np , self.Ns))
      
      self.xp = -np.cos(np.pi*np.linspace(0,self.deg,self.Np)/self.deg)
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
      
      
      Imat      = np.eye(self.Np)
      Imat[0,0] = Imat[-1,-1] = 0

      self.phi_ni =  np.linalg.solve(self.LpD, -self.param.alpha*Imat)
      self.phi_ne = -self.phi_ni
      
      self.E_ni    = -np.dot(self.Dp, self.phi_ni)
      self.E_ne    = -np.dot(self.Dp, self.phi_ne)
      
      Imat[0,0] = Imat[-1,-1] = 1.0
      self.I_Np = Imat
      
      
      self.weak_bc_Te = False
      self.weak_bc_ni = False
      self.weak_bc_ne = False
      
      
      self.I_NpNv     = np.eye(self.Nv * self.Np)
      
      self.xp_module = np
    
    def initialize_kinetic_coefficients(self, mode):
      xp = self.xp_module
      
      if mode == "tabulated":
        """
        kinetic-coefficient initialization from tabulated data
        """
        
        ki   = np.genfromtxt("Ar3species/Ar_1Torr_300K/Ionization.300K.txt" , delimiter=",", skip_header=True) # m^3/s
        mu_e = np.genfromtxt("Ar3species/Ar_1Torr_300K/Mobility.300K.txt"   , delimiter=",", skip_header=True) # mu * n0 (1/(Vms))
        De   = np.genfromtxt("Ar3species/Ar_1Torr_300K/Diffusivity.300K.txt", delimiter=",", skip_header=True) # D  * n0 (1/(ms))
        
        #ki   = np.genfromtxt("Ar3species/Ar_1Torr_300K/Ar_o_sp_Ar_P0_1.00Torr_T0_300.00K_Ionization_15.76_eV.txt" , delimiter=",", skip_header=True) # m^3/s
        #mu_e = np.genfromtxt("Ar3species/Ar_1Torr_300K/Ar_o_sp_Ar_P0_1.00Torr_T0_300.00K_mobility.txt"            , delimiter=",", skip_header=True) # mu * n0 (1/(Vms))
        #De   = np.genfromtxt("Ar3species/Ar_1Torr_300K/Ar_o_sp_Ar_P0_1.00Torr_T0_300.00K_diffusion.txt"           , delimiter=",", skip_header=True) # D  * n0 (1/(ms))
        
        
        # non-dimentionalized QoIs
        ki  [:, 0]  *= (1/self.param.ev_to_K)
        mu_e[:, 0]  *= (1/self.param.ev_to_K)
        De [:, 0]   *= (1/self.param.ev_to_K)
        
        ki  [:, 1]  *= (self.param.np0 * self.param.tau)
        mu_e[:, 1]  *= (self.param.V0 * self.param.tau / (self.param.L**2 * self.param.n0 * self.param.np0))
        De [:, 1]   *= (self.param.tau / (self.param.L**2 *self.param.n0 * self.param.np0) )
        
        ki_data   = ki
        mu_e_data = mu_e
        De_data   = De
        
        # non-dimensional QoI interpolations and their derivatives, w.r.t., ne, nTe
        ki                  = scipy.interpolate.UnivariateSpline(ki[:,0],  ki  [:,1], k=1, s=0, ext="const")
        ki_d                = ki.derivative(n=1)
        
        self.param.ki       =  lambda nTe, ne : ki(nTe/ne)
        self.param.ki_ne    =  lambda nTe, ne : ki_d(nTe/ne) * (-nTe/(ne**2))
        self.param.ki_nTe   =  lambda nTe, ne : ki_d(nTe/ne) * (1/ne)
        
        mu_e                = scipy.interpolate.UnivariateSpline(mu_e[:,0],  mu_e[:,1], k=1, s=0, ext="const")
        mu_e_d              = mu_e.derivative(n=1)
        self.param.mu_e     = lambda nTe, ne : mu_e(nTe/ne)
        self.param.mu_e_ne  = lambda nTe, ne : mu_e_d(nTe/ne) * (-nTe/(ne**2))
        self.param.mu_e_nTe = lambda nTe, ne : mu_e_d(nTe/ne) * (1/ne)
        
        De                  = scipy.interpolate.UnivariateSpline(De [:,0],   De [:,1], k=1, s=0, ext="const")
        De_d                = De.derivative(n=1)
        self.param.De       = lambda nTe, ne : De(nTe/ne)
        self.param.De_ne    = lambda nTe, ne : De_d(nTe/ne) * (-nTe/(ne**2))
        self.param.De_nTe   = lambda nTe, ne : De_d(nTe/ne) * (1/ne)
        
        # # De                  = scipy.interpolate.UnivariateSpline(De [:,0],   De [:,1], k=1, s=0, ext="const")
        # # De_d                = De.derivative(n=1)
        # a1                  = (self.param.V0 * self.param.tau / (self.param.L**2 * self.param.n0 * self.param.np0))
        # a2                  = (self.param.tau / (self.param.L**2 *self.param.n0 * self.param.np0) )
        # a3                  = a2/a1
        # self.param.De       = lambda nTe, ne : a3 * (mu_e(nTe/ne) * (nTe/ne))
        # self.param.De_ne    = lambda nTe, ne : a3 * ((nTe/ne) * self.param.mu_e_ne(nTe, ne)  + mu_e(nTe/ne) * (-nTe/(ne**2)))
        # self.param.De_nTe   = lambda nTe, ne : a3 * ((nTe/ne) * self.param.mu_e_nTe(nTe, ne) + mu_e(nTe/ne) * (1/ne))
        
        Te = ki_data[:,0]#np.linspace(ki_data[0,0], ki_data[-1,0], 100)
        plt.figure(figsize=(16, 6), dpi=300)
        
        plt.subplot(1, 3, 1)
        plt.loglog(Te, self.param.mu_e(Te, 1)/((self.param.V0 * self.param.tau/(self.param.L**2))) ,  ".-",             label ="0D-BTE")
        plt.loglog(Te, np.ones_like(Te) * self.param._mu_e/((self.param.V0 * self.param.tau/(self.param.L**2))), "--", label ="Liu")
        plt.loglog(mu_e_data[:,0], mu_e_data[:,1]/((self.param.V0 * self.param.tau/(self.param.L**2))), ".--", label ="data")
        plt.grid(visible=True)
        plt.xlabel(r"$T_e$(eV)")
        plt.ylabel(r"$\mu_e$ ($V^{-1} m^2 s^{-1} $)")
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.loglog(Te, self.param.De(Te, 1)/(self.param.tau/(self.param.L**2)) ,             ".-", label   = "0D-BTE")
        plt.loglog(Te, np.ones_like(Te) * self.param._De/((self.param.tau/(self.param.L**2))), "--", label = "Liu")
        plt.loglog(De_data[:,0], De_data[:,1]/((self.param.tau/(self.param.L**2))), ".--", label           = "data")
        plt.grid(visible=True)
        plt.xlabel(r"$T_e$(eV)")
        plt.ylabel(r"$D_e$ ($m^{2}s^{-1}$)")
        plt.legend()
        
        ki_arr       = lambda nTe,ne : self.param.np0 * self.param.tau * 1.235e-13 * np.exp(-18.687 * np.abs(ne / nTe))   
        plt.subplot(1, 3, 3)
        plt.loglog(Te, self.param.ki(Te, 1) / (self.param.np0 * self.param.tau) ,             ".-", label ="0D-BTE")
        plt.loglog(Te, ki_arr(Te, 1)/(self.param.np0 * self.param.tau),                     "--", label ="Liu")
        plt.loglog(Te, ki_data[:,1]/(self.param.np0 * self.param.tau),                     ".--", label ="data")
        plt.grid(visible=True)
        plt.legend()
        plt.xlabel(r"$T_e$(eV)")
        plt.ylabel(r"$k_i$ ($m^{3}s^{-1}$)")
        
        plt.tight_layout()
        plt.savefig(args.fname+"_kinetic_coefficients.png")
        plt.close()
      
      elif mode == "fixed-0":
        self.param.ki       = lambda nTe,ne : self.param.np0 * self.param.tau * 1.235e-13 * xp.exp(-18.687 * xp.abs(ne / nTe))   
        self.param.ki_ne    = lambda nTe,ne : self.param.np0 * self.param.tau * 1.235e-13 * xp.exp(-18.687 * xp.abs(ne / nTe))  * (-18.687 / nTe)
        self.param.ki_nTe   = lambda nTe,ne : self.param.np0 * self.param.tau * 1.235e-13 * xp.exp(-18.687 * xp.abs(ne / nTe))  * (18.687 * ne / nTe**2)  
        
        self.param.mu_e     = lambda nTe,ne : self.param._mu_e * xp.ones_like(ne)
        self.param.mu_e_ne  = lambda nTe,ne : ne * 0
        self.param.mu_e_nTe = lambda nTe,ne : ne * 0
        
        self.param.De       = lambda nTe,ne : self.param._De * xp.ones_like(ne)
        self.param.De_ne    = lambda nTe,ne : ne * 0
        self.param.De_nTe   = lambda nTe,ne : ne * 0
      
      else:
        raise NotImplementedError
    
    def initial_condition(self, type=0):
      xp      = self.xp_module
      args    = self.args
      ele_idx = self.ele_idx
      ion_idx = self.ion_idx
      Te_idx  = self.Te_idx
      Uin     = xp.ones((self.Np, self.Nv))
      
      if type==0:
        if self.args.restore==1:
          print("~~~restoring solver from ", "%s_%04d_u.npy"%(args.fname, args.rs_idx))
          Uin             =  xp.load("%s_%04d.npy"%(args.fname, args.rs_idx))
          npts            = Uin.shape[0]
          xx1             = -np.cos(np.pi*np.linspace(0,npts-1, npts)/(npts-1))
          v0pinv          = np.linalg.solve(np.polynomial.chebyshev.chebvander(xx1, npts-1), np.eye(npts))
          P1              = np.dot(np.polynomial.chebyshev.chebvander(self.xp, npts-1), v0pinv)
          Uin             = np.dot(P1, Uin)
          
        else:
          
          if (self.args.ic_file != ""):
            print("reading IC from file %s "%(self.args.ic_file))
            u1             =  xp.load(self.args.ic_file)
            
            npts            = u1.shape[0]
            xx1             = -np.cos(np.pi*np.linspace(0,npts-1, npts)/(npts-1))
            v0pinv          = np.linalg.solve(np.polynomial.chebyshev.chebvander(xx1, npts-1), np.eye(npts))
            P1              = np.dot(np.polynomial.chebyshev.chebvander(self.xp, npts-1), v0pinv)
              
            u1              = np.dot(P1, u1)
            Uin[:, ele_idx] = u1[:, ele_idx] 
            Uin[:, ion_idx] = u1[:, ion_idx] 
            Uin[:, Te_idx]  = u1[:, Te_idx]  
            
          else:
            xx = self.param.L * (self.xp + 1)
            Uin[:, ele_idx] = 1e6 * (1e7 + 1e9 * (1-0.5 * xx/self.param.L)**2 * (0.5 * xx/self.param.L)**2) / self.param.np0
            Uin[:, ion_idx] = 1e6 * (1e7 + 1e9 * (1-0.5 * xx/self.param.L)**2 * (0.5 * xx/self.param.L)**2) / self.param.np0
            Uin[:, Te_idx]  = self.param.Teb
          
            Uin[0, Te_idx]  = self.param.Teb0
            Uin[-1, Te_idx] = self.param.Teb1
            Uin[:, Te_idx] *= Uin[:, ele_idx]
          
          
      
        nTe = Uin[:, Te_idx]
        ne  = Uin[:, ele_idx]
      else:
        raise NotImplementedError
        
      return Uin
    
    def initialize(self,type=0):
      xp      = self.xp_module
      args    = self.args
      ele_idx = self.ele_idx
      ion_idx = self.ion_idx
      Te_idx  = self.Te_idx
      
      if(args.use_tab_data==1):
        self.initialize_kinetic_coefficients(mode="tabulated")
      else:
        self.initialize_kinetic_coefficients(mode="fixed-0")
      
      self.mu[:, ele_idx] = 0.0
      self.mu[:, ion_idx] = self.param.mu_i
      
      self.D[:, ele_idx] = 0.0
      self.D[:, ion_idx] = self.param.Di
    
    def copy_operators_H2D(self, dev_id):
      
      if self.args.use_gpu==0:
        return
      
      with cp.cuda.Device(dev_id):
        self.Dp             = cp.asarray(self.Dp)
        self.LpD            = cp.asarray(self.LpD)
        self.Lp             = cp.asarray(self.Lp)
        self.LpD_inv        = cp.asarray(self.LpD_inv)
        self.Zp             = cp.asarray(self.Zp)
        self.E_ne           = cp.asarray(self.E_ne)
        self.E_ni           = cp.asarray(self.E_ni)
        self.mu             = cp.asarray(self.mu)
        self.D              = cp.asarray(self.D)
        self.Zp             = cp.asarray(self.Zp)
        self.I_Np           = cp.asarray(self.I_Np)
        self.I_NpNv         = cp.asarray(self.I_NpNv)
        
      return
    
    def copy_operators_D2H(self, dev_id):

      if self.args.use_gpu==0:
        return

      with cp.cuda.Device(dev_id):
        self.Dp             = cp.asnumpy(self.Dp)
        self.LpD_inv        = cp.asnumpy(self.DpT)
        self.Zp             = cp.asnumpy(self.Zp)
        self.E_ne           = cp.asnumpy(self.E_ne)
        self.E_ni           = cp.asnumpy(self.E_ni)
        self.mu             = cp.asnumpy(self.mu)
        self.D              = cp.asnumpy(self.D)
        self.Zp             = cp.asnumpy(self.Zp)
        self.I_Np           = cp.asnumpy(self.I_Np)
        self.I_NpNv         = cp.asnumpy(self.I_NpNv)
        
        
      return  
        
    def electron_bdy_temperature(self, Uin: np.array, time, dt):
      xp  = self.xp_module
      
      ele_idx = self.ele_idx
      ion_idx = self.ion_idx
      Te_idx  = self.Te_idx
      
      ne      = Uin[: , ele_idx]
      ni      = Uin[: , ion_idx]
      nTe     = Uin[: ,  Te_idx]
      
      Te      = nTe/ne
      
      phi     = self.solve_poisson(Uin[:,ele_idx], Uin[:, ion_idx], time)
      E       = -xp.dot(self.Dp, phi)
      
      ki                  = self.param.ki(nTe, ne)
      self.mu[:, ele_idx] = self.param.mu_e(nTe, ne)
      self.D [:, ele_idx] = self.param.De(nTe, ne)
      
      
      mu_i    = self.mu[:, ion_idx]
      mu_e    = self.mu[:, ele_idx]
      
      De      = self.D[:, ele_idx]
      Di      = self.D[:, ion_idx]
      
      Je      = xp.array([0., 0.])
      Ji      = xp.array([0., 0.])
      
      Ji[0]   = self.Zp[ion_idx] * mu_i[0]  * ni[0]  * E[0]
      Ji[-1]  = self.Zp[ion_idx] * mu_i[-1] * ni[-1] * E[-1]
      
      Je[0]   = -self.param.mw_flux(Te[0])  * ne[0]  - self.param.gamma * Ji[0]
      Je[-1]  =  self.param.mw_flux(Te[-1]) * ne[-1] - self.param.gamma * Ji[-1]
      
      rhs     = xp.copy(Te)
      rhs[0]  = 2.5 * Te[0]  * Je[0]
      rhs[-1] = 2.5 * Te[-1] * Je[-1]
      
      qe_mat      = xp.eye(self.Np)
      Imat        = self.I_Np
      #qe         = -1.5 * De * xp.dot(self.Dp, nTe) - 2.5 * mu_e * E * nTe - De * Te * xp.dot(self.Dp, ne)
      
      ne_x         = xp.dot(self.Dp, ne)
      qe_mat[0,:]  = -1.5 * De[0]  * ne[0]  * self.Dp[0,:]  - 2.5 * mu_e[0]  * E[0]  * ne[0]  * Imat[0, :]  - De[0]  * ne_x * Imat[0  , :] 
      qe_mat[-1,:] = -1.5 * De[-1] * ne[-1] * self.Dp[-1,:] - 2.5 * mu_e[-1] * E[-1] * ne[-1] * Imat[-1, :] - De[-1] * ne_x * Imat[-1 , :]
      
      Te_new       = xp.linalg.solve(qe_mat, rhs) 
      #print("old \n", Te)
      #print("new \n", Te_new)
      return Te_new[0], Te_new[-1]
    
    def temperature_solve(self, Uin : np.array, time, dt):
      """
      solves non-linear system for boundary conditions. 
      """
      xp         = self.xp_module
      
      ele_idx = self.ele_idx
      ion_idx = self.ion_idx
      Te_idx  = self.Te_idx
      
      ne      = Uin[: , ele_idx]
      ni      = Uin[: , ion_idx]
      nTe     = xp.copy(Uin[: ,  Te_idx])
      
      mu_i    = self.mu[:, ion_idx]
      mu_e    = self.mu[:, ele_idx]
      
      De      = self.D[:, ele_idx]
      Di      = self.D[:, ion_idx]
      
      phi     = self.solve_poisson(Uin[:,ele_idx], Uin[:, ion_idx], time)
      E       = -xp.dot(self.Dp, phi)
      
      Je      = self.Zp[ele_idx] * mu_e * ne * E - De * xp.dot(self.Dp, ne)
      Ji      = self.Zp[ion_idx] * mu_i * ni * E - Di * xp.dot(self.Dp, ni)
      
      Ji[0]   = self.Zp[ion_idx] * mu_i[0]  * ni[0]  * E[0]
      Ji[-1]  = self.Zp[ion_idx] * mu_i[-1] * ni[-1] * E[-1]
      
      Teb0_g = nTe[0]/ne[0]
      Teb1_g = nTe[-1]/ne[-1]
      
      def res(Te_bdy, xloc):
        nTe[:]    = Uin[:, Te_idx]
        
        nTe[xloc] = ne[xloc] * Te_bdy
        Te        = nTe/ne
        
        Je[0]     = -self.param.mw_flux(Te[0])  * ne[0]  - self.param.gamma * Ji[0]
        Je[-1]    =  self.param.mw_flux(Te[-1]) * ne[-1] - self.param.gamma * Ji[-1]
        
        qe        = -1.5 * De * xp.dot(self.Dp, nTe) - 2.5 * mu_e * E * nTe - De * Te * xp.dot(self.Dp, ne)
        return qe[xloc] - 2.5 * Te[xloc] * Je[xloc]

      # sol0 = scipy.optimize.root_scalar(res, args=(0) , x0=self.param.Teb0, method='brentq',bracket = (self.param.Teb0 * 0  , 10 * self.param.Teb0), xtol=self.args.atol, rtol=self.args.rtol, maxiter=50)
      # sol1 = scipy.optimize.root_scalar(res, args=(-1), x0=self.param.Teb1, method='brentq',bracket = (self.param.Teb1 * 0  , 10 * self.param.Teb1), xtol=self.args.atol, rtol=self.args.rtol, maxiter=50)
      
      try:
        sol0 = scipy.optimize.root_scalar(res, args=(0) , x0=self.param.Teb0, method='brentq',bracket = (Teb0_g * (0.9)  , 1.1 * Teb0_g), xtol=1e-8, rtol=1e-8, maxiter=100)
        T0   = sol0.root
      except:
        T0   = Teb0_g
        print("left boundary temperature solve failed setting T0=%.8E, res(T0)=%.8E\n"%(T0, res(T0, 0)))
        
        # Te  = xp.linspace(0,100,100)
        # fTe = xp.array([res(Te[i], -1) for i in range(len(Te))])
        # plt.plot(Te, fTe)
        # plt.title("left bdy")
        # plt.show()
        # plt.close()
        
      
      try:
        sol1 = scipy.optimize.root_scalar(res, args=(-1), x0=self.param.Teb1, method='brentq',bracket = (Teb1_g * 0.9  , 1.1 * Teb1_g), xtol=1e-8, rtol=1e-8, maxiter=100)
        T1   = sol1.root
      except:
        T1   = Teb1_g
        print("right boundary temperature solve failed setting T1=%.8E, res(T1)=%.8E\n"%(T1, res(T1, -1)))
        
        
        # Te  = xp.linspace(0,100,100)
        # fTe = xp.array([res(Te[i], -1) for i in range(len(Te))])
        # plt.plot(Te, fTe)
        # plt.title("right bdy")
        # plt.show()
        # plt.close()
        
      # sol0 = scipy.optimize.root_scalar(res, args=(0) , x0=0, x1 = 100 * self.param.Teb0, method='secant', xtol=1e-4, rtol=1e-4, maxiter=50)
      # sol1 = scipy.optimize.root_scalar(res, args=(-1), x0=0, x1 = 100 * self.param.Teb1, method='secant', xtol=1e-4, rtol=1e-4, maxiter=50)
      
      
      
      
        
      # assert sol0.converged == True
      # assert sol1.converged == True
      #print("Te[0] = %.8E Te[-1]=%.8E feval[0]=%.8E, feval[-1]=%.8E "%(T0, T1, res(T0, 0), res(T1, -1)))
      #print("ks = %.8E ks0 = %.8E ks1= %.8E"%(self.param.ks, self.param.mw_flux(sol0.root), self.param.mw_flux(sol1.root)))
      return T0, T1
      
    def rhs(self, Uin : np.array, time, dt):
      """Evaluates the residual.

      Inputs:
        Uin     : Current state
        time    : Current time
        dt      : Time step
        weak_bc : enforce the boundary conditions weakly before derivative computations. 
        for implicit methods, the above  struggles to converge, not sure why ? 
        store_poisson: If true, phi, and E will be stored for future reuse. 

      Outputs:
        returns residual vector

      Notes:
        This function currently assumes that Ns=2 and NT=1
      """
      xp         = self.xp_module
      FUin       = xp.zeros_like(Uin)
      
      ele_idx = self.ele_idx
      ion_idx = self.ion_idx
      Te_idx  = self.Te_idx
      
      ne      = Uin[: , ele_idx]
      ni      = Uin[: , ion_idx]
      nTe     = Uin[: ,  Te_idx]
      
      if (self.weak_bc_Te):
        nTe[0]  = self.param.Teb0 * ne[0]
        nTe[-1] = self.param.Teb1 * ne[-1]
        
      Te      = nTe/ne

      ki                  = self.param.ki(nTe, ne)
      self.mu[:, ele_idx] = self.param.mu_e(nTe, ne)
      self.D [:, ele_idx] = self.param.De(nTe, ne)
      
      mu_i    = self.mu[:, ion_idx]
      mu_e    = self.mu[:, ele_idx]
      
      De      = self.D[:, ele_idx]
      Di      = self.D[:, ion_idx]
      
      phi     = self.solve_poisson(Uin[:,ele_idx], Uin[:, ion_idx], time)
      E       = -xp.dot(self.Dp, phi)
      
      Us_x    = xp.dot(self.Dp, Uin[: , 0:self.Ns])
      fluxJ    = xp.empty((self.Np, self.Ns))
      for sp_idx in range(self.Ns):
        fluxJ[:, sp_idx] = self.Zp[sp_idx] * self.mu[: , sp_idx] * Uin[: , sp_idx] * E - self.D[: , sp_idx] * Us_x[:, sp_idx]
      
      if (self.weak_bc_ne):        
        fluxJ[0 , ele_idx] = -self.param.ks0 * ne[0]   - self.param.gamma * self.mu[0 , ion_idx] * ni[0]  * E[0] 
        fluxJ[-1, ele_idx] = self.param.ks1  * ne[-1]  - self.param.gamma * self.mu[-1, ion_idx] * ni[-1] * E[-1] 
        
      if (self.weak_bc_ni):
        fluxJ[0 , ion_idx] = self.mu[0 , ion_idx] * ni[0]  * E[0] 
        fluxJ[-1, ion_idx] = self.mu[-1, ion_idx] * ni[-1] * E[-1] 
        
      fluxJ_x = xp.dot(self.Dp, fluxJ)
      for sp_idx in range(self.Ns):
        FUin[:,sp_idx] = ki * self.param.n0 * ne - fluxJ_x[:,sp_idx]
      
      qe                  = -1.5 * De * xp.dot(self.Dp, nTe) - 2.5 * mu_e * E * nTe - De * Te * Us_x[:,ele_idx]
      qe_x                = xp.dot(self.Dp, qe)
      
      Je                  = fluxJ[:,ele_idx]
      JeE                 = Je * E * self.param.V0
      
      FUin[:  , Te_idx]   = (-2/3) * (qe_x  + JeE + self.param.Hi * ki * self.param.n0 * ne)
      
      strong_bc = xp.zeros((2,self.Nv))  
      if self.args.ts_type=="FE":
        # if evolving nT else just set this to zero. 
        FUin[0  , Te_idx]   = (self.param.Teb0 * (ne[0]  + dt * FUin[0,ele_idx] ) - Uin[0,Te_idx])/dt
        FUin[-1 , Te_idx]   = (self.param.Teb1 * (ne[-1] + dt * FUin[-1,ele_idx]) - Uin[-1,Te_idx])/dt
        return FUin
      
      elif self.args.ts_type=="BE":
        if not self.weak_bc_ne:            
          strong_bc[0,  ele_idx] = (fluxJ[0,ele_idx]  - (-self.param.ks0 * ne[ 0]   - self.param.gamma * mu_i[0]  * E[0 ] * ni[ 0] ))
          strong_bc[-1, ele_idx] = (fluxJ[-1,ele_idx] - (self.param.ks1  * ne[-1]   - self.param.gamma * mu_i[-1] * E[-1] * ni[-1] ))
          
        if not self.weak_bc_ni:            
          strong_bc[0,  ion_idx] = (fluxJ[0 , ion_idx] - self.mu[0 , ion_idx] * ni[0]  * E[0] )
          strong_bc[-1, ion_idx] = (fluxJ[-1, ion_idx] - self.mu[-1, ion_idx] * ni[-1] * E[-1] )
          
        if not self.weak_bc_Te:
          strong_bc[0,  Te_idx]  = (nTe[0]  - self.param.Teb0 * ne[ 0])
          strong_bc[-1, Te_idx]  = (nTe[-1] - self.param.Teb1 * ne[-1])
      
      return FUin, strong_bc
    
    def rhs_jacobian(self, Uin: np.array, time, dt):
      xp  = self.xp_module
      dof = self.Nv * self.Np
      jac = xp.zeros((dof, dof))
      
      ele_idx = self.ele_idx
      ion_idx = self.ion_idx
      Te_idx  = self.Te_idx
      
      ne      = Uin[: , ele_idx]
      ni      = Uin[: , ion_idx]
      nTe     = Uin[: , Te_idx]
      
      if self.weak_bc_Te:
        nTe[0]  = self.param.Teb0 * ne[0]
        nTe[-1] = self.param.Teb1 * ne[-1]
      
      Te      = nTe/ne
      
      ki      = self.param.ki(nTe,  ne)
      ki_ne   = self.param.ki_ne(nTe, ne)  
      ki_nTe  = self.param.ki_nTe(nTe, ne)
      
      self.mu[:, ele_idx] = self.param.mu_e(nTe, ne)
      self.D [:, ele_idx] = self.param.De(nTe, ne)
      
      mu_e_ne = self.param.mu_e_ne (nTe, ne)
      mu_e_nTe= self.param.mu_e_nTe(nTe, ne)
      
      De_ne   = self.param.De_ne(nTe, ne)
      De_nTe  = self.param.De_nTe(nTe, ne)
      
      mu_i    = self.mu[:, ion_idx]
      mu_e    = self.mu[:, ele_idx]
      
      De      = self.D[:, ele_idx]
      Di      = self.D[:, ion_idx]
      
      Np      = self.Np
      Nv      = self.Nv
      
      phi_ni  = self.phi_ni
      phi_ne  = self.phi_ni
      
      E_ni    = self.E_ni
      E_ne    = self.E_ne
      
      Imat    = self.I_Np 
      
      phi     = self.solve_poisson(ne, ni, time)
      E       = -xp.dot(self.Dp, phi)
      
      ne_x     = xp.dot(self.Dp, ne)
      ni_x     = xp.dot(self.Dp, ni)
      
      Js_nk    = xp.zeros((self.Ns, self.Ns, self.Np, self.Np))
      
      for i in range(self.Ns):
        if i == ele_idx:
          Js_nk[i,i] = self.Zp[i] * ( self.mu[:,i] * (E * Imat + Uin[:,i] * E_ne) + Uin[:,i] * E * mu_e_ne * Imat ) - (self.D[:,i] * self.Dp + ne_x * De_ne * Imat)
        elif i == ion_idx:
          Js_nk[i,i] = self.Zp[i] * self.mu[:,i] * (E * Imat + Uin[:,i] * E_ni) - self.D[:,i] * self.Dp
        else:
          Js_nk[i,i] = self.Zp[i] * self.mu[:,i] * (E * Imat) - self.D[:,i] * self.Dp
          
      Js_nk[ele_idx, ion_idx] = self.Zp[ele_idx] * self.mu[:,ele_idx] * Uin[:,ele_idx] * E_ni 
      Je_nTe                  = self.Zp[ele_idx] * ne * E * mu_e_nTe * Imat - ne_x * De_nTe * Imat
      Js_nk[ion_idx, ele_idx] = self.Zp[ion_idx] * self.mu[:,ion_idx] * Uin[:,ion_idx] * E_ne
      
      Je_ne        = Js_nk[ele_idx, ele_idx]
      Je_ni        = Js_nk[ele_idx,ion_idx]
      
      if self.weak_bc_ne:
        Je_ne[0  , :] = -self.param.ks0 * Imat[0,:] - self.param.gamma * mu_i[0]   * ni[0]   * E_ne[0  , :]
        Je_ne[-1 , :] = self.param.ks1 *Imat[-1,:]  - self.param.gamma * mu_i[-1]  * ni[-1]  * E_ne[-1 , :]
        
        Je_ni[0 , :]  = - self.param.gamma * mu_i[0]   * (ni[0]   * E_ni[0 , :] + E[0]  * Imat[0,:])
        Je_ni[-1, :]  = - self.param.gamma * mu_i[-1]  * (ni[-1]  * E_ni[-1 ,:] + E[-1] * Imat[-1,:])
        
        Je_nTe[0,  :]  = 0
        Je_nTe[-1, :]  = 0
        
      
      Ji_ni        = Js_nk[ion_idx, ion_idx]
      Ji_ne        = Js_nk[ion_idx, ele_idx]
      
      if self.weak_bc_ni:
        Ji_ni[0 ,:] = mu_i[0]  * (ni[0]  * E_ni[0 ,:] + E[0]  * Imat[0 ,:])
        Ji_ni[-1,:] = mu_i[-1] * (ni[-1] * E_ni[-1,:] + E[-1] * Imat[-1,:])
        
        Ji_ne[0,:]  = mu_i[0]  * ni[0]  * E_ne[0 , :]
        Ji_ne[-1,:] = mu_i[-1] * ni[-1] * E_ne[-1, :]
        
      Je_x_ne  = xp.dot(self.Dp, Je_ne)
      Je_x_nTe = xp.dot(self.Dp, Je_nTe)
      Je_x_ni  = xp.dot(self.Dp, Je_ni)
      
      Ji_x_ni  = xp.dot(self.Dp, Ji_ni)
      Ji_x_ne  = xp.dot(self.Dp, Ji_ne)
      
      Rne_ne   = self.param.n0 * (ki + ne * ki_ne) * Imat - Je_x_ne
      Rne_ni   = -Je_x_ni
      Rne_nTe  = ki_nTe * self.param.n0 * ne * Imat - Je_x_nTe
      
      Rni_ni   = -Ji_x_ni
      Rni_ne   = self.param.n0 * (ki + ne * ki_ne) * Imat -Ji_x_ne
      Rni_nTe  = ki_nTe * self.param.n0 * ne * Imat 
      
      Je       = self.Zp[ele_idx] * mu_e * ne * E - De * ne_x
      
      if self.weak_bc_ne:
        Je[0 ]  = -self.param.ks0 * ne[0]   - self.param.gamma * mu_i[0 ] * ni[0]  * E[0]
        Je[-1]  = self.param.ks1  * ne[-1]  - self.param.gamma * mu_i[-1] * ni[-1] * E[-1]
      
      Teb0     = self.param.Teb0
      Teb1     = self.param.Teb1
      
      qe_ne          = -2.5 * (nTe * (mu_e * E_ne + E * mu_e_ne * Imat)) - De * ( Te * self.Dp - ne_x * (Te/ne) * Imat) - (Te * ne_x) * De_ne * Imat
      qe_ni          = -2.5 * (mu_e * nTe) * E_ni
      qe_nTe         = -1.5 * De * self.Dp  -2.5 * E * (nTe * mu_e_nTe + mu_e) * Imat - (ne_x/ne) * (nTe * De_nTe + De) * Imat    #- 2.5 * mu_e * E * Imat - De * (ne_x / ne) * Imat
      
      if self.weak_bc_Te:
        qe_ne[0 , :]   = -1.5 * De[ 0]  * self.Dp[ 0, :] * Teb0 * Imat[0 ,:] - 2.5 * mu_e[ 0] * (E[0]  * Teb0 * Imat[0 ,:] + ne[0]  * Teb0 * E_ne[0, :]) - De[0]  * Teb0 * self.Dp[0 ,:]
        qe_ne[-1, :]   = -1.5 * De[-1]  * self.Dp[-1, :] * Teb1 * Imat[-1,:] - 2.5 * mu_e[-1] * (E[-1] * Teb1 * Imat[-1,:] + ne[-1] * Teb1 * E_ne[-1,:]) - De[-1] * Teb1 * self.Dp[-1,:]
        
        qe_ni[0 ,:]    = 0
        qe_ni[-1,:]    = 0
      
        qe_nTe[0,:]    = 0
        qe_nTe[-1,:]   = 0
      
      qe_x_ne  = xp.dot(self.Dp, qe_ne)
      qe_x_ni  = xp.dot(self.Dp, qe_ni)
      qe_x_nTe = xp.dot(self.Dp, qe_nTe)
      
      JeE_ne        = (E_ne * Je + Je_ne * E) * self.param.V0 
      JeE_ni        = (E_ni * Je + Je_ni * E) * self.param.V0
      JeE_nTe       = (Je_nTe * E * Imat)     * self.param.V0
      
      RnTe_ne       = (-2/3) * (qe_x_ne  + JeE_ne + self.param.Hi * self.param.n0 * (ki  + ne * ki_ne) * Imat)
      RnTe_ni       = (-2/3) * (qe_x_ni  + JeE_ni)
      RnTe_nTe      = (-2/3) * (qe_x_nTe + JeE_nTe + self.param.Hi * self.param.n0 * (ne * ki_nTe * Imat))
      
      jac[ele_idx :: self.Nv , ele_idx :: self.Nv] = Rne_ne
      jac[ele_idx :: self.Nv , ion_idx :: self.Nv] = Rne_ni
      jac[ele_idx :: self.Nv , Te_idx  :: self.Nv] = Rne_nTe
      
      jac[ion_idx :: self.Nv , ele_idx :: self.Nv] = Rni_ne
      jac[ion_idx :: self.Nv , ion_idx :: self.Nv] = Rni_ni
      jac[ion_idx :: self.Nv , Te_idx  :: self.Nv] = Rni_nTe
      
      jac[Te_idx :: self.Nv , ele_idx :: self.Nv]  = RnTe_ne 
      jac[Te_idx :: self.Nv , ion_idx :: self.Nv]  = RnTe_ni
      jac[Te_idx :: self.Nv , Te_idx  :: self.Nv]  = RnTe_nTe
      
      jac_bc = xp.zeros((2, self.Nv, self.Np * self.Nv))
      
      if self.args.ts_type=="BE":
        if not self.weak_bc_ne:
          jac_bc[0, ele_idx , ele_idx::self.Nv]  = Je_ne[ 0,:] - (-self.param.ks0 * Imat[ 0 ,:] - self.param.gamma * mu_i[ 0]  * ni[ 0]  * E_ne[ 0  , :])
          jac_bc[1, ele_idx , ele_idx::self.Nv]  = Je_ne[-1,:] - (self.param.ks1  * Imat[-1 ,:] - self.param.gamma * mu_i[-1]  * ni[-1]  * E_ne[-1  , :])
          
          jac_bc[0, ele_idx , ion_idx::self.Nv]  = Je_ni[ 0,:] - ( - self.param.gamma * mu_i[ 0]  * (ni[ 0]  * E_ni[ 0  , :] + E[0]  * Imat[0  , :]))
          jac_bc[1, ele_idx , ion_idx::self.Nv]  = Je_ni[-1,:] - ( - self.param.gamma * mu_i[-1]  * (ni[-1]  * E_ni[-1  , :] + E[-1] * Imat[-1 , :]))
        
        if not self.weak_bc_ni:
          jac_bc[0, ion_idx , ion_idx::self.Nv]  = Ji_ni[0  ,:] - (mu_i[0]  * (ni[0]  * E_ni[0   , :] + E[0]  * Imat[0 ,:]))
          jac_bc[1, ion_idx , ion_idx::self.Nv]  = Ji_ni[-1 ,:] - (mu_i[-1] * (ni[-1] * E_ni[-1  , :] + E[-1] * Imat[-1,:]))
          
          jac_bc[0, ion_idx , ele_idx::self.Nv]  = Ji_ne[0  ,:] - (mu_i[0]  * ni[0]  * E_ne[0   , :])
          jac_bc[1, ion_idx , ele_idx::self.Nv]  = Ji_ne[-1 ,:] - (mu_i[-1] * ni[-1] * E_ne[-1  , :])
        
        if not self.weak_bc_Te:            
          jac_bc[0, Te_idx, 0 * self.Nv + ele_idx]  = -self.param.Teb0
          jac_bc[0, Te_idx, 0 * self.Nv + Te_idx ]  = 1
          
          jac_bc[1, Te_idx, (self.Np-1) * self.Nv + ele_idx]  = -self.param.Teb1
          jac_bc[1, Te_idx, (self.Np-1) * self.Nv + Te_idx ]  = 1
          
      else:
        raise NotImplementedError
      
      return jac, jac_bc
      
    def rhs_jacobian_FD(self, Uin, time, dt):
      """
      compute finite differences based jacobian
      """
      xp  = self.xp_module
      r0  = self.rhs(Uin, time, dt)[0].reshape((-1))
      dof = self.Nv * self.Np
      jac = xp.zeros((dof, dof))
      
      for j in range(0, self.Np):
        for i in range(0, self.Nv):
          dU = max(xp.finfo(np.float64).eps**0.125 * xp.absolute(Uin[j,i]), xp.finfo(xp.float64).eps)
          Up = np.copy(Uin)
          Up[j,i] +=dU
          if i==0:
            #print(dU, xp.absolute(Uin[j,i]))
            rp = self.rhs(Up, time, dt)[0].reshape((-1))
            #print(rp.reshape(Uin.shape))
            
          rp = self.rhs(Up, time + dt, dt)[0].reshape((-1))
          jac[:, j * self.Nv + i] = (rp - r0)/dU  
      
      #  #print(jac)
      # R_Te = xp.zeros((self.Np, self.Np))
      # for j in range(0, self.Np):
      #   dU      = max(xp.sqrt(xp.finfo(np.float64).eps)*xp.absolute(Uin[j,2]), xp.sqrt(xp.finfo(xp.float64).eps))
      #   Up      = np.copy(Uin)
      #   Up[j,2] +=dU
      #   rp = self.rhs(Up, time, dt).reshape((-1))
        
      #   w=(rp - r0)/dU  
      #   R_Te[j,:] = w[2::self.Nv]
      
      # jac[2::self.Nv, 2::self.Nv] = R_Te
      return jac
    
    def sensitivity_jac_FD(self, Uin, time, dt):
      xp  = self.xp_module
      dof = self.Nv * self.Np
      jac = xp.zeros((dof, dof))
      
      u      = np.copy(Uin)
      du     = xp.zeros_like(u)
      
      v, _   = self.solve_step(u, du, time, dt, 1e-13, 1e-13, 400)
      
      for j in range(0, self.Np):
        for i in range(0, self.Nv):
          dU      = max(xp.finfo(np.float64).eps**0.125 * xp.absolute(Uin[j,i]), xp.finfo(xp.float64).eps)
          up      = np.copy(u)
          up[j,i] +=dU 
          vp, _       = self.solve_step(up, du, time, dt, 1e-13, 1e-13, 400)
          
          # plt.figure(figsize=(8,8))
          # plt.plot(self.xp,  u[:,0], label=r"u")
          # plt.plot(self.xp, up[:,0], label=r"up")
          # plt.plot(self.xp, vp[:,0], label=r"vp")
          # plt.plot(self.xp, v[:,0], label=r"v")
          # plt.legend()
          # plt.grid()
          # plt.savefig("%s_i%02d_j%02d"%(self.args.fname, i, j))
          # plt.close()
          jac[:, j * self.Nv + i] = (vp - v).reshape((-1))/dU
      
      return jac 
      
    def solve_poisson(self, ne, ni,time):
        """Solve Gauss' law for the electric potential.

        Inputs:
          ne   : Values of electron density at xp
          ni   : Values of ion density at xp
          time : Current time

        Outputs: None (sets self.phi to computed potential)
        """
        xp    = self.xp_module
        r     = - self.param.alpha * (ni-ne)
        r[0]  = xp.sin(2 * xp.pi * time) #+ self.params.verticalShift
        r[-1] = 0.0
        return xp.dot(self.LpD_inv, r)
    
    def solve(self, Uin, ts_type):
      xp = self.xp_module
      if ts_type == "FE":
        dx              = xp.min(self.xp[1:] - self.xp[0:-1])
        dt              = self.args.cfl * dx
        tT              = 1.0 * self.args.cycles
        steps           = max(1,int(tT/dt))
        u               = xp.copy(Uin)
        
        tt              = 0
        
        print(tT, dt, tT/dt)
               
        for ts_idx in range(steps):
          if ts_idx % 1000 == 0:
            print("time = %.2E step=%d/%d"%(tt, ts_idx, steps))
          k1 = dt * self.rhs(u, tt, dt)
          u  = u + k1
          #print(u)
          tt+=dt
        return u
        
      if ts_type == "RK2":
        dx              = xp.min(self.xp[1:] - self.xp[0:-1])
        dt              = self.args.cfl * dx
        tT              = 1.0 * self.args.cycles
        steps           = max(1,int(tT/dt))
        u               = xp.copy(Uin)
        
        tt              = 0
        
        print(tT, dt, tT/dt)
               
        for ts_idx in range(steps):
          if ts_idx % 1000 == 0:
            print("time = %.2E step=%d/%d"%(tt, ts_idx, steps))
          k1 = dt * self.rhs(u, tt, dt)
          k2 = dt * self.rhs(u + 0.5 * k1, tt + 0.5 * dt, dt)
          u  = u + k2
          #print(u)
          tt+=dt

        return u
      elif ts_type == "RK4":
        dx              = xp.min(self.xp[1:] - self.xp[0:-1])
        dt              = self.args.cfl * dx
        tT              = 1.0 * self.args.cycles
        steps           = max(1,int(tT/dt))
        u               = xp.copy(Uin)
        tt              = 0
        
        print(tT, dt, steps)
        
        for ts_idx in range(steps):
          if ts_idx % 1000 == 0:
            print("time = %.2E step=%d/%d"%(tt, ts_idx, steps))
          
          k1 = dt * self.rhs(u, tt , dt)
          k2 = dt * self.rhs(u + 0.5 * k1, tt + 0.5 * dt, dt)
          k3 = dt * self.rhs(u + 0.5 * k2, tt + 0.5 * dt, dt)
          k4 = dt * self.rhs(u +  k3     , tt + dt      , dt)
          
          u = u + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)
          tt+=dt
        return u 
      elif ts_type == "BE":
        args            = self.args
        dx              = xp.min(self.xp[1:] - self.xp[0:-1])
        dt              = self.args.cfl 
        tT              = self.args.cycles
        
        io_cycle        = self.args.io_cycle_freq
        io_freq         = int(io_cycle/dt)
        cycle_freq      = int(1.0/dt)
      
        cp_cycle        = self.args.cp_cycle_freq
        cp_freq         = int(cp_cycle/dt)
        
        steps           = max(1,int(tT/dt))
        u               = xp.copy(Uin)
        Imat            = xp.eye(u.shape[0] * u.shape[1])
        rtol            = self.args.rtol
        atol            = self.args.atol
        iter_max        = self.args.max_iter
        
        print("++++ Using backward Euler ++++")
        print("T = %.4E RF cycles = %.1E dt = %.4E steps = %d atol = %.2E rtol = %.2E max_iter=%d"%(tT, self.args.cycles, dt, steps, self.args.atol, self.args.rtol, self.args.max_iter))
        
        # jac  = self.rhs_jacobian(u, 0, dt)[0]
        # jac1 = self.rhs_jacobian_FD(u, 0, dt)
        # print("jac")
        # #print(jac[2::self.Nv, 0::self.Nv])
        
        # print("ne, ne")
        # print(jac[0::self.Nv, 0::self.Nv])
        # print("ne, ni")
        # print(jac[0::self.Nv, 1::self.Nv])
        # print("ne, nTe")
        # print(jac[0::self.Nv, 2::self.Nv])
        
        # print("ni, ne")
        # print(jac[1::self.Nv, 0::self.Nv])
        # print("ni, ni")
        # print(jac[1::self.Nv, 1::self.Nv])
        # print("ni, nTe")
        # print(jac[1::self.Nv, 2::self.Nv])
        
        # print("nTe, ne")
        # print(jac[2::self.Nv, 0::self.Nv])
        # print("nTe, ni")
        # print(jac[2::self.Nv, 1::self.Nv])
        # print("nTe, nTe")
        # print(jac[2::self.Nv, 2::self.Nv])
        
        # print("\n\nFD jac")
        # # print(jac1[0::self.Nv, 0::self.Nv])
        # # print(jac1[1::self.Nv, 1::self.Nv])
        # #print(jac1[2::self.Nv, 0::self.Nv])
        # #print(jac1[2::self.Nv, 1::self.Nv])
        # print("ne, ne")
        # print(jac1[0::self.Nv, 0::self.Nv])
        # print("ne, ni")
        # print(jac1[0::self.Nv, 1::self.Nv])
        # print("ne, nTe")
        # print(jac1[0::self.Nv, 2::self.Nv])
        
        
        # print("ni, ne")
        # print(jac1[1::self.Nv, 0::self.Nv])
        # print("ni, ni")
        # print(jac1[1::self.Nv, 1::self.Nv])
        # print("ni, nTe")
        # print(jac1[1::self.Nv, 2::self.Nv])
        
        # # print(jac1[0::self.Nv, 0::self.Nv])
        # # print(jac1[1::self.Nv, 1::self.Nv])
        # #print(jac1[2::self.Nv, 0::self.Nv])
        # #print(jac1[2::self.Nv, 1::self.Nv])
        # print("nTe, ne")
        # print(jac1[2::self.Nv, 0::self.Nv])
        # print("nTe, ni")
        # print(jac1[2::self.Nv, 1::self.Nv])
        # print("nTe, nTe")
        # print(jac1[2::self.Nv, 2::self.Nv])
        # sys.exit(-1)
        
        
        tt              = 0
        Imat            = xp.eye(self.Np * self.Nv)
        u0              = xp.copy(u)
        
        #self.weak_bc_ni = True
        #self.weak_bc_ne = True
        du  = xp.zeros_like(u)
        
        ts_idx_b  = 0 
        if args.restore==1:
          ts_idx_b = int(args.rs_idx * io_freq)
          tt       = ts_idx_b * dt
          print("restoring solver from ts_idx = ", int(args.rs_idx * io_freq), "at time = ",tt)
          
        cycle_avg_u=xp.zeros_like(u)
        
        # 0  ,    1       ,  2          , 3          , 4 
        # ne , 1.5 neTe me,  mue E ne   , k_ela n0 ne, k_ion n0 ne  
        num_mc_qoi        = 5
        self.macro_qoi    = xp.zeros((num_mc_qoi, self.args.Np))
        output_macro_qoi  = bool(self.args.ca_macro_qoi)
        
        for ts_idx in range(ts_idx_b, steps):
          
          if ((ts_idx % cycle_freq) == 0):
            u1 = xp.copy(u)
            # print("time = %.2E step=%d/%d"%(tt, ts_idx, steps))
            # print("  Newton iter {0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}".format(ns_info["iter"], ns_info["atol"], ns_info["rtol"]))
            a1 = xp.max(xp.abs(u1[:,0] -u0[:, 0]))
            a2 = xp.max(xp.abs((u1[:,0]-u0[:,0]) / xp.max(xp.abs(u0[:,0]))))
            print("ts_idx = %d, Teb0= %.10E, Teb1= %.10E" %(ts_idx, self.param.Teb0, self.param.Teb1))
            print("||u(t+T) - u(t)|| = %.8E and ||u(t+T) - u(t)||/||u(t)|| = %.8E"% (a1, a2))
            u0=u1
            
            if output_macro_qoi == True:
              # macro_qoi 
              self.macro_qoi = 0.5 * dt  * self.macro_qoi / 1.0
              np.save("%s_macro_qoi_cycle_%04d.npy"%(args.fname, ts_idx//cycle_freq), self.macro_qoi)
              self.macro_qoi[:, :] = 0.0
          
          if (self.args.bc_dirichlet_e == 0):
            self.param.Teb0 , self.param.Teb1 = self.electron_bdy_temperature(u, tt, dt) #self.temperature_solve(u, tt, dt)
            self.param.ks0  , self.param.ks1  = self.param.mw_flux(self.param.Teb0), self.param.mw_flux(self.param.Teb1)
            #print("ts_idx = %d, Teb0= %.10E, Teb1= %.10E" %(ts_idx, self.param.Teb0, self.param.Teb1))
          
          cycle_avg_u[:, self.ele_idx] += u[:, self.ele_idx]
          cycle_avg_u[:, self.ion_idx] += u[:, self.ion_idx]
          cycle_avg_u[:, self.Te_idx]  += u[:, self.Te_idx] / u[:, self.ele_idx] 
          
          if output_macro_qoi == True:
            
            E  = -xp.dot(self.Dp,self.solve_poisson(u[:, self.ele_idx], u[:, self.ion_idx], tt)) * (self.param.V0/self.param.L)
            a1 = 1.0 / (self.param.V0 * self.param.tau/(self.param.L**2))  
            
            self.macro_qoi[0] += u[:, self.ele_idx]  * self.param.np0
            self.macro_qoi[1] += u[:, self.Te_idx ]  * scipy.constants.electron_mass * 1.5 * self.param.np0
            self.macro_qoi[2] += self.param.mu_e(u[:, self.Te_idx], u[:, self.ele_idx]) * (a1) * u[:, self.ele_idx] * self.param.np0 * E
            
            self.macro_qoi[3] += 0.0
            self.macro_qoi[4] += self.param.ki(u[:, self.Te_idx], u[:, self.ele_idx]) * (1/self.param.tau/self.param.np0) * self.param.n0 * self.param.np0 *  u[:, self.ele_idx] * self.param.np0
          
          def residual(du):
            du       = du.reshape((self.Np, self.Nv))
            u1       = u + du
            rhs, bc  = self.rhs(u1, tt + dt, dt) 
            res      = du - dt * rhs
            
            if not self.weak_bc_ne:
              res[0  , self.ele_idx] = bc[0  , self.ele_idx]
              res[-1 , self.ele_idx] = bc[-1 , self.ele_idx]
            
            if not self.weak_bc_ni:              
              res[0  , self.ion_idx] = bc[0  , self.ion_idx]
              res[-1 , self.ion_idx] = bc[-1 , self.ion_idx]
            
            if not self.weak_bc_Te:              
              res[0  , self.Te_idx]  = bc[0  , self.Te_idx]
              res[-1 , self.Te_idx]  = bc[-1 , self.Te_idx]
              
            return res.reshape(-1)
        
          def jacobian(du):
            du          = du.reshape((self.Np, self.Nv))
            rhs_j, j_bc = self.rhs_jacobian(u, tt, dt)
            jac         = Imat - dt * rhs_j
            
            if not self.weak_bc_ne:
              jac[0 * self.Nv           + self.ele_idx, :] = j_bc[0, self.ele_idx,:]
              jac[(self.Np-1) * self.Nv + self.ele_idx, :] = j_bc[1, self.ele_idx,:]
            
            if not self.weak_bc_ni:              
              jac[0 * self.Nv           + self.ion_idx, :] = j_bc[0, self.ion_idx, :]
              jac[(self.Np-1) * self.Nv + self.ion_idx, :] = j_bc[1, self.ion_idx, :]
            
            if not self.weak_bc_Te:
              jac[0 * self.Nv           + self.Te_idx, :]  = j_bc[0, self.Te_idx, :]
              jac[(self.Np-1) * self.Nv + self.Te_idx, :]  = j_bc[1, self.Te_idx, :]
            
            return jac
          
          if(self.args.checkpoint==1 and (ts_idx % io_freq)==0):
            
            print("time = %.6E step=%d/%d"%(tt, ts_idx, steps))
            
            cycle_avg_u                 = cycle_avg_u * 0.5 * dt / io_cycle
            cycle_avg_u[:, self.Te_idx] = cycle_avg_u[:, self.Te_idx] * cycle_avg_u[:, self.ele_idx]
            
            self.plot(cycle_avg_u, tt, "%s_avg_%04d.png"     %(args.fname, ts_idx//io_freq))
            self.plot(u,           tt, "%s_%04d.png" %(args.fname, ts_idx//io_freq))
            
            if (ts_idx % cp_freq ==0):
              xp.save("%s_%04d_avg.npy"%(self.args.fname, ts_idx//io_freq), cycle_avg_u)
              xp.save("%s_%04d.npy"%(self.args.fname, ts_idx//io_freq), u)

            cycle_avg_u[:,:] = 0
              
          if ((ts_idx % cycle_freq) ==0):  
            cycle_avg_u[:,:] = 0
              
          ns_info = glow1d_utils.newton_solver(du, residual, jacobian, atol, rtol, iter_max ,xp)
          
          # jac_u = jacobian(0*du)
          # if(ts_idx % 10 ==0):
          #   jac_u_inv = xp.linalg.inv(jac_u)

          # ns_info = glow1d_utils.newton_solver_matfree(du, residual, lambda x: xp.dot(jac_u, x.reshape((-1))), lambda x: xp.dot(jac_u_inv, x), atol, rtol, 0.0, 1e-2, 20, iter_max ,xp)
          
          if ns_info["status"]==False:
            print("time = %.2E step=%d/%d"%(tt, ts_idx, steps))
            print("non-linear solver step FAILED!!! try with smaller time step size or increase max iterations")
            print("  Newton iter {0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}".format(ns_info["iter"], ns_info["atol"], ns_info["rtol"]))
            self.plot(u,           tt, "%s_%04d_failed.png" %(args.fname, ts_idx//io_freq))
            return u0
          
          du = ns_info["x"]
          u  = u + du
          
          cycle_avg_u[:, self.ele_idx] += u[:, self.ele_idx]
          cycle_avg_u[:, self.ion_idx] += u[:, self.ion_idx]
          cycle_avg_u[:, self.Te_idx]  += u[:, self.Te_idx] / u[:, self.ele_idx] 
          
          if output_macro_qoi == True:
            
            E  = -xp.dot(self.Dp,self.solve_poisson(u[:, self.ele_idx], u[:, self.ion_idx], tt + dt)) * (self.param.V0/self.param.L)
            a1 = 1.0 / (self.param.V0 * self.param.tau/(self.param.L**2))  
            
            self.macro_qoi[0] += u[:, self.ele_idx]  * self.param.np0
            self.macro_qoi[1] += u[:, self.Te_idx ]  * scipy.constants.electron_mass * 1.5 * self.param.np0
            self.macro_qoi[2] += self.param.mu_e(u[:, self.Te_idx], u[:, self.ele_idx]) * (a1) * u[:, self.ele_idx] * self.param.np0 * E
            
            self.macro_qoi[3] += 0.0
            self.macro_qoi[4] += self.param.ki(u[:, self.Te_idx], u[:, self.ele_idx]) * (1/self.param.tau/self.param.np0) * self.param.n0 * self.param.np0 *  u[:, self.ele_idx] * self.param.np0
          
          if(ts_idx % 100 == 0 ):
            print("time = %.6E step=%d/%d"%(tt, ts_idx, steps) + "--Newton iter {0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}".format(ns_info["iter"], ns_info["atol"], ns_info["rtol"]))
            
          tt+=dt
          
        print("time = %.10E step=%d/%d"%(tt, ts_idx, steps))
        return u  
      
      else:
        raise NotImplementedError
      
    def plot(self, Uin, time, fname):
      fig       = plt.figure(figsize=(18,8), dpi=300)
      
      def asnumpy(x):
        if self.args.use_gpu==1:
          if self.xp_module==cp:
            return self.xp_module.asnumpy(x)
          else:
            return x
        else:
          return x
      
      Uin = asnumpy(Uin)
      ne  = Uin[:, self.ele_idx]
      ni  = Uin[:, self.ion_idx]
      Te  = Uin[:, self.Te_idx]/ne
      
      if self.args.use_gpu==1:
        self.LpD_inv = asnumpy(self.LpD_inv)
        self.Dp      = asnumpy(self.Dp)
      
      phi = self.solve_poisson(Uin[:,0], Uin[:,1], time)
      E   = -np.dot(self.Dp, phi)
      
      if self.args.use_gpu==1:
        self.LpD_inv = cp.asarray(self.LpD_inv)
        self.Dp      = cp.asarray(self.Dp)
      
      label_str = "T=%.4f cycles"%(time)
      plt.subplot(2, 3, 1)
      plt.semilogy(self.xp, self.param.np0 * ne, label=label_str)
      plt.xlabel(r"x/L")
      plt.ylabel(r"$n_e (m^{-3})$")
      plt.legend()
      plt.grid(visible=True)
      
      plt.subplot(2, 3, 2)
      plt.semilogy(self.xp, self.param.np0 * ni, label=label_str)
      plt.xlabel(r"x/L")
      plt.ylabel(r"$n_i (m^{-3})$")
      plt.grid(visible=True)
      
      plt.subplot(2, 3, 3)
      plt.plot(self.xp, Te, label=label_str)
      plt.xlabel(r"x/L")
      plt.ylabel(r"$T_e (eV)$")
      plt.legend()
      plt.grid(visible=True)
      
      
      plt.subplot(2, 3, 4)
      plt.plot(self.xp, E * ((self.param.V0 / self.param.L)), label=label_str)
      plt.xlabel(r"x/L")
      plt.ylabel(r"$E (V/m)$")
      plt.legend()
      plt.grid(visible=True)
      
      plt.subplot(2, 3, 5)
      plt.plot(self.xp, phi * (((self.param.V0))), label=label_str)
      plt.xlabel(r"x/L")
      plt.ylabel(r"$\phi (V)$")
      plt.legend()
      plt.grid(visible=True)
      
      plt.suptitle("T=%.4f cycles"%(time))
      
      plt.tight_layout()
      fig.savefig("%s"%(fname))
      plt.close()
    
    def solve_step(self, u, du, time, dt, atol, rtol, iter_max):
      xp              = self.xp_module 
      Imat            = self.I_NpNv
      tt              = time
      status          = True
      
      if (self.args.bc_dirichlet_e == 0):
        self.param.Teb0 , self.param.Teb1 = self.electron_bdy_temperature(u, tt, dt) 
        self.param.ks0  , self.param.ks1  = self.param.mw_flux(self.param.Teb0), self.param.mw_flux(self.param.Teb1)
      
      def residual(du):
        u1       = u + du
        rhs, bc  = self.rhs(u1, tt + dt, dt) 
        res      = du - dt * rhs
        
        if not self.weak_bc_ne:
          res[0  , self.ele_idx] = bc[0  , self.ele_idx]
          res[-1 , self.ele_idx] = bc[-1 , self.ele_idx]
        
        if not self.weak_bc_ni:              
          res[0  , self.ion_idx] = bc[0  , self.ion_idx]
          res[-1 , self.ion_idx] = bc[-1 , self.ion_idx]
        
        if not self.weak_bc_Te:              
          res[0  , self.Te_idx]  = bc[0  , self.Te_idx]
          res[-1 , self.Te_idx]  = bc[-1 , self.Te_idx]
          
        return res.reshape(-1)
        
      def jacobian(du):
        rhs_j, j_bc = self.rhs_jacobian(u, tt, dt)
        jac         = Imat - dt * rhs_j
        
        if not self.weak_bc_ne:
          jac[0 * self.Nv           + self.ele_idx, :] = j_bc[0, self.ele_idx,:]
          jac[(self.Np-1) * self.Nv + self.ele_idx, :] = j_bc[1, self.ele_idx,:]
        
        if not self.weak_bc_ni:              
          jac[0 * self.Nv           + self.ion_idx, :] = j_bc[0, self.ion_idx, :]
          jac[(self.Np-1) * self.Nv + self.ion_idx, :] = j_bc[1, self.ion_idx, :]
        
        if not self.weak_bc_Te:
          jac[0 * self.Nv           + self.Te_idx, :]  = j_bc[0, self.Te_idx, :]
          jac[(self.Np-1) * self.Nv + self.Te_idx, :]  = j_bc[1, self.Te_idx, :]
        
        return jac
          
      ns_info = glow1d_utils.newton_solver(du, residual, jacobian, atol, rtol, iter_max ,xp)
      if ns_info["status"]==False:
        print("time %.2E non-linear solver step FAILED!!! try with smaller time step size or increase max iterations"%(tt))
        print("\tNewton iter {0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}".format(ns_info["iter"], ns_info["atol"], ns_info["rtol"]))
        status=False
        return u, status
      
      du = ns_info["x"]
      v  = u + du
      return v, status
    
    def evolve(self, u, time, dt, sensitivity_mat=False):
      xp              = self.xp_module 
      steps           = (int)(time/dt)
      cycle_freq      = (int)(1/dt)
      
      assert steps * dt == time
      
      tb              = 0
      
      dv_by_du        = np.copy(self.I_NpNv)
      J0              = np.copy(self.I_NpNv)
      u0              = np.copy(u)
      
      J1              = np.copy(self.I_NpNv)
      du              = np.zeros_like(u)
      
      assert self.weak_bc_ne==False
      assert self.weak_bc_ni==False
      assert self.weak_bc_Te==False
      
      J1[0 * self.Nv           + self.ele_idx, :]   = 0 
      J1[(self.Np-1) * self.Nv + self.ele_idx, :]   = 0 
      J1[0 * self.Nv           + self.ion_idx, :]   = 0 
      J1[(self.Np-1) * self.Nv + self.ion_idx, :]   = 0 
      J1[0 * self.Nv           + self.Te_idx,  :]   = 0 
      J1[(self.Np-1) * self.Nv + self.Te_idx,  :]   = 0 
      
      um = np.copy(u)
      
      for ts_idx in range(steps + 1):
        tn          = tb + ts_idx* dt
        
        if (ts_idx > 0 and ts_idx % cycle_freq == 0):
          a1 = np.linalg.norm(u-um)
          r1 = a1/np.linalg.norm(u)
          print("time = %.4E ||u1-u0|| = %.4E ||u1-u0||/||u0|| = %.4E"%(tn, a1, r1))
          um = np.copy(u)
        
        if ts_idx==steps:
          break
        
        v, status   = self.solve_step(u, du, tn, dt, self.args.atol, self.args.rtol, self.args.max_iter)
        du          = v-u
        
        if status == False:
          print("!!! solver failed.... :(")
          sys.exit(0)
        
        if sensitivity_mat == True:
          rhs_j, j_bc  = self.rhs_jacobian(v, tn + dt, dt)
          jac          = self.I_NpNv - dt * rhs_j
          
          if not self.weak_bc_ne:
            jac[0 * self.Nv           + self.ele_idx, :]     = j_bc[0, self.ele_idx,:]
            jac[(self.Np-1) * self.Nv + self.ele_idx, :]     = j_bc[1, self.ele_idx,:]
            
          if not self.weak_bc_ni:              
            jac[0 * self.Nv           + self.ion_idx, :]     = j_bc[0, self.ion_idx, :]
            jac[(self.Np-1) * self.Nv + self.ion_idx, :]     = j_bc[1, self.ion_idx, :]
            
          if not self.weak_bc_Te:
            jac[0 * self.Nv           + self.Te_idx, :]     = j_bc[0, self.Te_idx, :]
            jac[(self.Np-1) * self.Nv + self.Te_idx, :]     = j_bc[1, self.Te_idx, :]
          
          dv_by_du = np.linalg.solve(jac, np.dot(J1, J0))
          J0       = dv_by_du
          
          # dv_by_du_fd = self.sensitivity_jac_FD(u, tn, dt)
          # np.savetxt("dv_by_du.txt", dv_by_du, fmt='%.4E')
          # np.savetxt("dv_by_du_fd.txt", dv_by_du_fd, fmt='%.4E')
          # sys.exit(0)
          
        u = v
        
        
      u1 = u
      
      return u0, u1, dv_by_du
      
    def time_periodic_shooting(self, u, atol, rtol, max_iter):
      xp         = self.xp_module
      dt         = self.args.cfl
      Imat       = self.I_NpNv
      
      tt         = 0
      du         = xp.zeros_like(u)
      
      alpha0     = 1.0
      alpha_min  = 1e-1
      alpha_max  = 1.0
      alpha      = alpha_min
      
      def residual(u):
        u0, u1, Js = self.evolve(u, 1, dt, sensitivity_mat=False)
        return (u1-u0)
      
      u0, u1, Js = self.evolve(u, 1.0, dt, sensitivity_mat=False)
      print(xp.linalg.norm(u1-u0), xp.linalg.norm(u1-u0)/xp.linalg.norm(u0))
      res        = (u1 - u0)
      
      count      = 0
      r0         = res
      
      norm_rr    = norm_r0 = xp.linalg.norm(r0)
      converged  = ((norm_rr/norm_r0 < rtol) or (norm_rr < atol))
      u          = u1
      
      while( not converged and (count < max_iter) ):
        u0, u1, Js = self.evolve(u, 1, dt, sensitivity_mat=True)
        jac        = (Js - self.I_NpNv)
        jac_inv    = xp.linalg.inv(jac)
        
        rr         = (u1-u0)
        norm_rr    = xp.linalg.norm(rr)
        jinv_rr    = xp.dot(jac_inv, rr.reshape((-1))).reshape(u.shape)
        alpha      = min(0.01, 0.01 * xp.linalg.norm(u)/xp.linalg.norm(jinv_rr))
        u          = u  - alpha * jinv_rr
        # while(1):
        #   ug         = u  + alpha * xp.dot(jac_inv, -rr).reshape(u.shape)
        #   rr_g       = residual(ug)
        #   norm_rrg   = np.linalg.norm(rr_g)
          
        #   if norm_rrg > norm_rr:
        #     alpha *=1e-1
        #     alpha  = max(alpha_min, alpha)
        #     if alpha == alpha_min:
        #       #u = u  + alpha * xp.dot(jac_inv, -rr).reshape(u.shape)
        #       print("shooting method fallback to u1")
        #       u = u1
        #       break
        #   else:
        #     u      = ug
        #     alpha *=1e1
        #     alpha  = min(alpha_max, alpha)
        #     break
          
          
            
        
        count   += 1
        #if count%1000==0:
        print("{0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}".format(count, norm_rr, norm_rr/norm_r0))
        converged = ((norm_rr/norm_r0 < rtol) or (norm_rr < atol))
        
        plt.figure(figsize=(16,8), dpi=200)
        plt.subplot(1, 2, 1)
        
        plt.plot(self.xp, u0[:, 0]         , label=r"u0(ne)")
        plt.plot(self.xp, u1[:, 0]         , label=r"u1(ne)")
        plt.plot(self.xp,  u[:, 0]         , label=r"u(ne)")
        plt.legend()
        plt.grid(visible=True)
        plt.xlabel(r"x")
        plt.ylabel(r"density x %.2E"%(self.param.np0))
        
        plt.subplot(1, 2, 2)
        plt.plot(self.xp, u0[:, 2]/u0[:, 0]         , label=r"u0(Te)")
        plt.plot(self.xp, u1[:, 2]/u1[:, 0]         , label=r"u1(Te)")
        plt.plot(self.xp,  u[:, 2]/ u[:, 0]         , label=r"u(Te)")
        plt.legend()
        plt.grid(visible=True)
        plt.xlabel(r"x")
        plt.ylabel(r"temperature [eV]")
        
        plt.savefig("%s_newton_iter_%04d.png"%(self.args.fname, count))
        plt.close()
      
      # while(alpha > 1e-8):
        
        
      #   if (not converged):
      #     alpha *= 0.25
      #     #print(alpha)
          
      #   else:
      #     #print("  Newton iter {0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}".format(count, norm_rr, norm_rr/norm_r0))
      #     break
      
      return u
    
    def fft_analysis(self, u, dt, atol, rtol, max_iter):
      """
      perform steady state solution fft analysis. 
      """
      
      xp              = self.xp_module 
      time            = 1.0
      steps           = (int)(time/dt)
      cycle_freq      = (int)(1/dt)
      
      assert steps * dt == time
      
      tb              = 0
      u0              = xp.copy(u)
      du              = xp.zeros_like(u)
      
      assert self.weak_bc_ne==False
      assert self.weak_bc_ni==False
      assert self.weak_bc_Te==False
      
      um              = xp.copy(u)
      
      ut              = xp.zeros((self.Np * self.Nv, steps + 1))

      for ts_idx in range(steps + 1):
        tn            = tb + ts_idx* dt
        ut[:, ts_idx] = u.reshape((-1))
        
        if (ts_idx % cycle_freq == 0):
          
          self.plot(u, tn, "%s_%04d.png" %(args.fname, ts_idx//cycle_freq))
          if (ts_idx > 0):
            a1 = np.linalg.norm(u-um)
            r1 = a1/np.linalg.norm(u)
            print("time = %.4E ||u1-u0|| = %.4E ||u1-u0||/||u0|| = %.4E"%(tn, a1, r1))
            um = np.copy(u)
        
        if ts_idx==steps:
          break
        
        v, status     = self.solve_step(u, du, tn, dt, atol, rtol, max_iter)
        du            = v-u
        u             = v
        
        if status == False:
          print("!!! solver failed.... :(")
          sys.exit(0)
      
      return ut
      
      
        

if (__name__ == "__main__"):
  parser = argparse.ArgumentParser()
  parser.add_argument("-Ns", "--Ns"                         , help="number of species"      , type=int, default=2)
  parser.add_argument("-NT", "--NT"                         , help="number of temperatures" , type=int, default=1)
  parser.add_argument("-Np", "--Np"                         , help="number of collocation points" , type=int, default=100)
  parser.add_argument("-cfl", "--cfl"                       , help="CFL factor (only used in explicit integrations)" , type=float, default=1e-1)
  parser.add_argument("-cycles", "--cycles"                 , help="number of cycles to run" , type=float, default=10)
  parser.add_argument("-ts_type", "--ts_type"               , help="ts mode" , type=str, default="BE")
  parser.add_argument("-atol", "--atol"                     , help="abs. tolerance" , type=float, default=1e-6)
  parser.add_argument("-rtol", "--rtol"                     , help="rel. tolerance" , type=float, default=1e-6)
  parser.add_argument("-fname", "--fname"                   , help="file name to store the solution" , type=str, default="1d_glow")
  parser.add_argument("-restore", "--restore"               , help="restore the solver from previous solution" , type=int, default=0)
  parser.add_argument("-rs_idx",  "--rs_idx"                , help="restore file_idx"   , type=int, default=0)
  parser.add_argument("-checkpoint", "--checkpoint"         , help="store the checkpoints every 250 cycles" , type=int, default=1)
  parser.add_argument("-max_iter", "--max_iter"             , help="max iterations for Newton solver" , type=int, default=1000)
  parser.add_argument("-dir"  , "--dir"                     , help="file name to store the solution" , type=str, default="glow1d_dir")
  parser.add_argument("-use_tab_data"  , "--use_tab_data"   , help="use Te based tabulated electron kinetic coefficients" , type=int, default=1)
  parser.add_argument("-bc_dirichlet_e", "--bc_dirichlet_e" , help="use fixed Dirichlet BC for the electron energy equation" , type=int, default=1)
  parser.add_argument("-use_gpu",         "--use_gpu"       , help="enable GPU accerleration (not compatible with tabulated kinetics)" , type=int, default=0)
  parser.add_argument("-gpu_device_id", "--gpu_device_id"   , help="GPU device id to use", type=int, default=0)
  parser.add_argument("-ca_macro_qoi"  , "--ca_macro_qoi"   , help="use Te based tabulated electron kinetic coefficients" , type=int, default=1)
  
  parser.add_argument("-ic_file"       , "--ic_file"        , help="initial condition file"                  , type=str, default="")
  parser.add_argument("-io_cycle_freq" , "--io_cycle_freq"  , help="io output every k-th cycle"              , type=float, default=1e0)
  parser.add_argument("-cp_cycle_freq" , "--cp_cycle_freq"  , help="checkpoint output every k-th cycle"      , type=float, default=1e1)
  parser.add_argument("-par_file"     , "--par_file"        , help="toml par file to specify run parameters" , type=str, default="")
  
  args      = parser.parse_args()
  
  if args.par_file != "":
    import toml
    tp  = toml.load(args.par_file)
    
    tp0                 = tp["solver"]
    args.atol           = tp0["atol"]
    args.rtol           = tp0["rtol"]
    args.max_iter       = tp0["max_iter"]
    args.use_gpu        = tp0["use_gpu"]
    args.gpu_device_id  = tp0["gpu_device_id"]
    args.restore        = tp0["restore"]
    args.rs_idx         = tp0["rs_idx"]
    args.fname          = tp0["fname"]
    args.dir            = tp0["dir"]
    args.ic_file        = tp0["ic_file"]
    args.io_cycle_freq  = tp0["io_cycle"]
    args.cp_cycle_freq  = tp0["cp_cycle"]
    args.cfl            = tp0["dt"]
    args.cycles         = tp0["cycles"]
    args.bc_dirichlet_e = tp0["dirichlet_Te"]
    args.Np             = tp0["Np"]
    
    
    tp0                 = tp["chemistry"]
    args.Ns             = tp0["Ns"]
    args.NT             = tp0["NT"]
    args.use_tab_data   = tp0["use_tab_data"]
    
    
    
  
  
  glow_1d   = glow1d_fluid(args)
  glow_1d.initialize()
  

  u         = glow_1d.initial_condition()
  if args.use_gpu==1:
    assert args.use_tab_data==0
    dev_id = args.gpu_device_id
    d1 = cp.cuda.Device(dev_id)
    d1.use()
    
    glow_1d.copy_operators_H2D(dev_id)
    
    d1 = cp.cuda.Device(dev_id)
    d1.use()
    
    u  = cp.asarray(u)
    glow_1d.xp_module = cp
    
    glow_1d.initialize_kinetic_coefficients(mode="fixed-0")
    print(type(glow_1d.param.ki(u[:,2], u[:, 0])))
    
  
  v      = glow_1d.solve(u, ts_type=args.ts_type)
  #ut     = glow_1d.fft_analysis(u, args.cfl, args.atol, args.rtol, args.max_iter)
  #np.save("ut.npy", ut)
  
  #v         = glow_1d.time_periodic_shooting(u, args.atol, args.rtol, args.max_iter)
  #python3 glowdischarge_1d.py -Np 240 -cycles 11 -ts_type BE -atol 1e-14 -rtol 1e-14 -dir glow1d_liu_N240_dt5e-4 -cfl 5e-4
  
  


