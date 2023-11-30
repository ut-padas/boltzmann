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
class glow1d_fluid():
    def __init__(self, args) -> None:
      self.args  = args
      
      dir            = args.dir
      if os.path.exists(dir):
        print("run directory exists, data will be overwritten")
        #sys.exit(0)
      else:
        os.makedirs(dir)
        print("directory %s created"%(dir))
      
      args.fname=str(dir)+"/"+args.fname
      
      self.param = glow1d_utils.parameters()
      
      self.Ns = self.args.Ns                   # Number of species
      self.NT = self.args.NT                   # Number of temperatures
      self.Nv = self.args.Ns + self.args.NT    # Total number of 'state' variables

      self.deg = self.args.Np-1  # degree of Chebyshev polys we use
      self.Np  = self.args.Np    # Number of points used to define state in space
      self.Nc  = self.args.Np-2  # number of collocation pts (Np-2 b/c BCs)
      
      self.ele_idx = 0
      self.ion_idx = 1
      self.Te_idx  = self.Ns
      
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
      
      self.xp_module = np
    
    def initialize(self,type=0):
      xp      = self.xp_module
      ele_idx = self.ele_idx
      ion_idx = self.ion_idx
      Te_idx  = self.Te_idx
      Uin     = xp.ones((self.Np, self.Nv))
      
      if type==0:
        if args.restore==1:
          print("~~~restoring solver from %s.npy"%(args.fname))
          Uin = xp.load("%s.npy"%(args.fname))
          # D = np.load('newton_3spec_CN_Np100.npy')
          # Np=  self.Np
          # ne  = D[0:Np]
          # ni  = D[Np:2*Np]
          # nb  = D[2*Np:3*Np]
          # neE = D[3*Np:]
          # Te  = (2./3) * neE / ne
          
          # Uin[:, ele_idx] = ne[:,0]
          # Uin[:, ion_idx] = ni[:,0]
          # Uin[:, Te_idx]  = Te[:,0]
          
          # Uin[0, Te_idx]  = self.param.Teb0
          # Uin[-1, Te_idx] = self.param.Teb1
          # Uin[:, Te_idx] *= Uin[:, ele_idx]
          
        else:
          xx = self.param.L * (self.xp + 1)
          Uin[:, ele_idx] = 1e6 * (1e7 + 1e9 * (1-0.5 * xx/self.param.L)**2 * (0.5 * xx/self.param.L)**2) / self.param.np0
          Uin[:, ion_idx] = 1e6 * (1e7 + 1e9 * (1-0.5 * xx/self.param.L)**2 * (0.5 * xx/self.param.L)**2) / self.param.np0
          Uin[:, Te_idx]  = 0.5 
        
          Uin[0, Te_idx]  = self.param.Teb0
          Uin[-1, Te_idx] = self.param.Teb1
          Uin[:, Te_idx] *= Uin[:, ele_idx]
          
          #print(Uin[:, Te_idx]/Uin[:, ele_idx])
        
        
                
        self.mu[:, ele_idx] = self.param.mu_e
        self.mu[:, ion_idx] = self.param.mu_i
        
        self.D[:, ele_idx] = self.param.De
        self.D[:, ion_idx] = self.param.Di
        
      else:
        raise NotImplementedError
      
      return Uin
    
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
      
      
      def res(Te_bdy, xloc):
        nTe[xloc] = ne[xloc] * Te_bdy
        Te        = nTe/ne
        
        Je[0]     = -self.param.mw_flux(Te[0])  * ne[0]  - self.param.gamma * Ji[0]
        Je[-1]    =  self.param.mw_flux(Te[-1]) * ne[-1] - self.param.gamma * Ji[-1]
        
        qe        = -1.5 * De * xp.dot(self.Dp, nTe) - 2.5 * mu_e * E * nTe - De * Te * xp.dot(self.Dp, ne)
        return qe[xloc] - 2.5 * Te[xloc] * Je[xloc]

      sol0 = scipy.optimize.root_scalar(res, args=(0) , x0=self.param.Teb0, method='brentq',bracket = (0,20), xtol=self.args.atol, rtol=self.args.rtol, maxiter=50)
      sol1 = scipy.optimize.root_scalar(res, args=(-1), x0=self.param.Teb1, method='brentq',bracket = (0,20), xtol=self.args.atol, rtol=self.args.rtol, maxiter=50)
      
      assert sol0.converged == True
      assert sol1.converged == True
      
      print("Te[0] = %.8E Te[-1]=%.8E feval[0]=%.8E, feval[-1]=%.8E, iter0=%d, iter1=%d"%(sol0.root, sol1.root, res(sol0.root, 0), res(sol1.root, -1), sol0.function_calls, sol1.function_calls))
      #print("ks = %.8E ks0 = %.8E ks1= %.8E"%(self.param.ks, self.param.mw_flux(sol0.root), self.param.mw_flux(sol1.root)))
      return sol0.root, sol1.root
      
            
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
        
      if (self.weak_bc_Te):
        nTe[0]  = self.param.Teb0 * ne[0]
        nTe[-1] = self.param.Teb1 * ne[-1]
        
      Te      = nTe/ne
      fluxJ_x = xp.dot(self.Dp, fluxJ)
      ki      = self.param.ki(nTe, ne)
      
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
          Js_nk[i,i] = self.Zp[i] * self.mu[:,i] * (E * Imat + Uin[:,i] * E_ne) - self.D[:,i] * self.Dp
        elif i == ion_idx:
          Js_nk[i,i] = self.Zp[i] * self.mu[:,i] * (E * Imat + Uin[:,i] * E_ni) - self.D[:,i] * self.Dp
        else:
          Js_nk[i,i] = self.Zp[i] * self.mu[:,i] * (E * Imat) - self.D[:,i] * self.Dp
          
      Js_nk[ele_idx, ion_idx] = self.Zp[ele_idx] * self.mu[:,ele_idx] * Uin[:,ele_idx] * E_ni 
      Js_nk[ion_idx, ele_idx] = self.Zp[ion_idx] * self.mu[:,ion_idx] * Uin[:,ion_idx] * E_ne
      
      if self.weak_bc_Te:
        nTe[0]  = self.param.Teb0 * ne[0]
        nTe[-1] = self.param.Teb1 * ne[-1]
      
      Te      = nTe/ne
      
      ki      = self.param.ki(nTe,  ne)
      ki_ne   = self.param.ki_ne(nTe, ne)  
      ki_nTe  = self.param.ki_nTe(nTe, ne)
      
      
      Je_ne        = Js_nk[ele_idx,ele_idx]
      Je_ni        = Js_nk[ele_idx,ion_idx]
      
      if self.weak_bc_ne:
        Je_ne[0  , :] = -self.param.ks0 * Imat[0,:] - self.param.gamma * mu_i[0]   * ni[0]   * E_ne[0  , :]
        Je_ne[-1 , :] = self.param.ks1 *Imat[-1,:]  - self.param.gamma * mu_i[-1]  * ni[-1]  * E_ne[-1 , :]
        
        Je_ni[0 , :]  = - self.param.gamma * mu_i[0]   * (ni[0]   * E_ni[0 , :] + E[0]  * Imat[0,:])
        Je_ni[-1, :]  = - self.param.gamma * mu_i[-1]  * (ni[-1]  * E_ni[-1 ,:] + E[-1] * Imat[-1,:])
        
      
      Ji_ni        = Js_nk[ion_idx, ion_idx]
      Ji_ne        = Js_nk[ion_idx, ele_idx]
      
      if self.weak_bc_ni:
        Ji_ni[0 ,:] = mu_i[0]  * (ni[0]  * E_ni[0 ,:] + E[0]  * Imat[0 ,:])
        Ji_ni[-1,:] = mu_i[-1] * (ni[-1] * E_ni[-1,:] + E[-1] * Imat[-1,:])
        
        Ji_ne[0,:]   = mu_i[0]  * ni[0]  * E_ne[0 , :]
        Ji_ne[-1,:]  = mu_i[-1] * ni[-1] * E_ne[-1, :]
        
      Je_x_ne = xp.dot(self.Dp, Je_ne)
      Je_x_ni = xp.dot(self.Dp, Je_ni)
      
      Ji_x_ni = xp.dot(self.Dp, Ji_ni)
      Ji_x_ne = xp.dot(self.Dp, Ji_ne)
      
      Rne_ne  = self.param.n0 * (ki + ne * ki_ne) * Imat - Je_x_ne
      Rne_ni  = -Je_x_ni
      Rne_nTe = ki_nTe * self.param.n0 * ne * Imat 
      
      Rni_ni  = -Ji_x_ni
      Rni_ne  = self.param.n0 * (ki + ne * ki_ne) * Imat -Ji_x_ne
      Rni_nTe = ki_nTe * self.param.n0 * ne * Imat 
      
      Je      = self.Zp[ele_idx] * mu_e * ne * E - De * ne_x
      
      if self.weak_bc_ne:
        Je[0 ]  = -self.param.ks0 * ne[0]   - self.param.gamma * mu_i[0 ] * ni[0]  * E[0]
        Je[-1]  = self.param.ks1  * ne[-1]  - self.param.gamma * mu_i[-1] * ni[-1] * E[-1]
      
      Teb0     = self.param.Teb0
      Teb1     = self.param.Teb1
      
      qe_ne          = -2.5 * (mu_e * nTe)[:,xp.newaxis] * E_ne - (De * Te)[:,xp.newaxis] * self.Dp + De * (ne_x / ne**2) * nTe * Imat
      qe_ni          = -2.5 * (mu_e * nTe)[:,xp.newaxis] * E_ni
      qe_nTe         = -1.5 * De[:,xp.newaxis] * self.Dp - 2.5 * mu_e * E * Imat - De * (ne_x / ne) * Imat
      
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
      
      JeE_ne        = (E_ne * Je[:,xp.newaxis] + Je_ne * E[:,xp.newaxis]) * self.param.V0 
      JeE_ni        = (E_ni * Je[:,xp.newaxis] + Je_ni * E[:,xp.newaxis]) * self.param.V0
      
      RnTe_ne       = (-2/3) * (qe_x_ne  + JeE_ne + self.param.Hi * self.param.n0 * (ki  + ne * ki_ne) * Imat)
      RnTe_ni       = (-2/3) * (qe_x_ni  + JeE_ni)
      RnTe_nTe      = (-2/3) * (qe_x_nTe + self.param.Hi * self.param.n0 * (ne * ki_nTe * Imat))
      
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
          jac_bc[0, ion_idx , ion_idx::self.Nv]  = Ji_ni[0  ,:] - (mu_i[0]  * (ni[0]  * E_ni[0   , :] + E[0]  * ni[0]  * Imat[0 ,:]))
          jac_bc[1, ion_idx , ion_idx::self.Nv]  = Ji_ni[-1 ,:] - (mu_i[-1] * (ni[-1] * E_ni[-1  , :] + E[-1] * ni[-1] * Imat[-1,:]))
          
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
      r0  = self.rhs(Uin, time, dt).reshape((-1))
      dof = self.Nv * self.Np
      jac = xp.zeros((dof, dof))
      
      for j in range(0, self.Np):
        for i in range(0, self.Nv):
          dU = max(xp.finfo(np.float64).eps**0.125 * xp.absolute(Uin[j,i]), xp.finfo(xp.float64).eps)
          Up = np.copy(Uin)
          Up[j,i] +=dU
          if i==0:
            print(dU, xp.absolute(Uin[j,i]))
            rp = self.rhs(Up, time, dt).reshape((-1))
            print(rp.reshape(Uin.shape))
            
          rp = self.rhs(Up, time + dt, dt).reshape((-1))
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
        dx              = xp.min(self.xp[1:] - self.xp[0:-1])
        dt              = self.args.cfl 
        tT              = self.args.cycles
        
        io_freq         = int(1/dt)
        
        steps           = max(1,int(tT/dt))
        u               = xp.copy(Uin)
        Imat            = xp.eye(u.shape[0] * u.shape[1])
        rtol            = self.args.rtol
        atol            = self.args.atol
        iter_max        = self.args.max_iter
        
        print("++++ Using backward Euler ++++")
        print("T = %.4E RF cycles = %.1E dt = %.4E steps = %d atol = %.2E rtol = %.2E max_iter=%d"%(tT, self.args.cycles, dt, steps, self.args.atol, self.args.rtol, self.args.max_iter))
        
        # jac  = self.rhs_jacobian(u, 0, dt)
        # jac1 = self.rhs_jacobian_FD(u, 0, dt)
        # print("jac")
        # #print(jac[2::self.Nv, 0::self.Nv])
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
        # print("nTe, ne")
        # print(jac1[2::self.Nv, 0::self.Nv])
        # print("nTe, ni")
        # print(jac[2::self.Nv, 1::self.Nv])
        # print("nTe, nTe")
        # print(jac1[2::self.Nv, 2::self.Nv])
        # sys.exit(-1)
        
        
        tt              = 0
        Imat            = xp.eye(self.Np * self.Nv)
        u0              = xp.copy(u)
        
        self.weak_bc_ni = True
        #self.weak_bc_ne = True
        du  = xp.zeros_like(u)
        for ts_idx in range(steps):
          
          # self.param.Teb0 , self.param.Teb1 = self.temperature_solve(u, tt, dt)
          # self.param.ks0  , self.param.ks1  = self.param.mw_flux(self.param.Teb0), self.param.mw_flux(self.param.Teb1)
          # print("ts_idx = %d, Teb0= %.10E, Teb1= %.10E" %(ts_idx, self.param.Teb0, self.param.Teb1))
          
          
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
          
          if(self.args.checkpoint==1 and (ts_idx % io_freq)==0):
              print("time = %.2E step=%d/%d"%(tt, ts_idx, steps))
              self.plot(u, tt, "%s_%04d.png"%(args.fname, ts_idx//io_freq))
              xp.save("%s_%04d.npy"%(self.args.fname, ts_idx//io_freq), u)
          
          ns_info = glow1d_utils.newton_solver(du, residual, jacobian, atol, rtol, iter_max ,xp)
          
          if ns_info["status"]==False:
            print("time = %.2E step=%d/%d"%(tt, ts_idx, steps))
            print("non-linear solver step FAILED!!! try with smaller time step size or increase max iterations")
            print("  Newton iter {0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}".format(ns_info["iter"], ns_info["atol"], ns_info["rtol"]))
            return u0
          
          du = ns_info["x"]
          u  = u + du
          
          if ts_idx % io_freq == 0:
            u1 = xp.copy(u)
            print("time = %.2E step=%d/%d"%(tt, ts_idx, steps))
            print("  Newton iter {0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}".format(ns_info["iter"], ns_info["atol"], ns_info["rtol"]))
            a1 = xp.linalg.norm(u1-u0)
            a2 = a1/ xp.linalg.norm(u0)
            print("||u(t+T) - u(t)|| = %.8E and ||u(t+T) - u(t)||/||u(t)|| = %.8E"% (a1, a2))
            u0=u1
            
          tt+=dt
        print("time = %.10E step=%d/%d"%(tt, ts_idx, steps))
        return u  
      
      else:
        raise NotImplementedError
      
    def plot(self, Uin, time, fname):
      fig       = plt.figure(figsize=(18,8), dpi=300)
      
      ne = Uin[:, self.ele_idx]
      ni = Uin[:, self.ion_idx]
      Te = Uin[:, self.Te_idx]/ne
      
      label_str = "T=%.4f cycles"%(time)
      
      plt.subplot(2, 3, 1)
      plt.plot(self.xp, self.param.np0 * ne, label=label_str)
      plt.xlabel(r"x/L")
      plt.ylabel(r"$n_e (m^{-3})$")
      plt.legend()
      plt.grid(visible=True)
      
      plt.subplot(2, 3, 2)
      plt.plot(self.xp, self.param.np0 * ni, label=label_str)
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
      phi = self.solve_poisson(Uin[:,0], Uin[:,1], time)
      E   = -np.dot(self.Dp, phi)
      
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
      
      
    
    
parser = argparse.ArgumentParser()
parser.add_argument("-Ns", "--Ns"                       , help="number of species"      , type=int, default=2)
parser.add_argument("-NT", "--NT"                       , help="number of temperatures" , type=int, default=1)
parser.add_argument("-Np", "--Np"                       , help="number of collocation points" , type=int, default=100)
parser.add_argument("-cfl", "--cfl"                     , help="CFL factor (only used in explicit integrations)" , type=float, default=1e-1)
parser.add_argument("-cycles", "--cycles"               , help="number of cycles to run" , type=float, default=10)
parser.add_argument("-ts_type", "--ts_type"             , help="ts mode" , type=str, default="BE")
parser.add_argument("-atol", "--atol"                   , help="abs. tolerance" , type=float, default=1e-6)
parser.add_argument("-rtol", "--rtol"                   , help="rel. tolerance" , type=float, default=1e-6)
parser.add_argument("-fname", "--fname"                 , help="file name to store the solution" , type=str, default="1d_glow")
parser.add_argument("-restore", "--restore"             , help="restore the solver" , type=int, default=0)
parser.add_argument("-checkpoint", "--checkpoint"       , help="store the checkpoints every 250 cycles" , type=int, default=1)
parser.add_argument("-max_iter", "--max_iter"           , help="max iterations for Newton solver" , type=int, default=1000)
parser.add_argument("-dir"  , "--dir"                   , help="file name to store the solution" , type=str, default="glow1d_dir")

args      = parser.parse_args()
glow_1d   = glow1d_fluid(args)

u         = glow_1d.initialize()
v         = glow_1d.solve(u, ts_type=args.ts_type)

#python3 glowdischarge_1d.py -Np 240 -cycles 11 -ts_type BE -atol 1e-14 -rtol 1e-14 -dir glow1d_liu_N240_dt5e-4 -cfl 5e-4

