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

class glow1d_fluid():
    def __init__(self, args) -> None:
      self.args  = args
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
        else:
          xx = self.param.L * (self.xp + 1)
          Uin[:, ele_idx] = 1e6 * (1e7 + 1e9 * (1-0.5 * xx/self.param.L)**2 * (0.5 * xx/self.param.L)**2) / self.param.np0
          Uin[:, ion_idx] = 1e6 * (1e7 + 1e9 * (1-0.5 * xx/self.param.L)**2 * (0.5 * xx/self.param.L)**2) / self.param.np0
          Uin[:, Te_idx]  = 0.5
        
        Uin[0, Te_idx]  = self.param.Teb
        Uin[-1, Te_idx] = self.param.Teb
                
        self.mu[:, ele_idx] = self.param.mu_e
        self.mu[:, ion_idx] = self.param.mu_i
        
        self.D[:, ele_idx] = self.param.De
        self.D[:, ion_idx] = self.param.Di
        
      else:
        raise NotImplementedError
      
      return Uin
      
    def rhs(self, Uin : np.array, time, dt):
      """Evaluates the residual.

      Inputs:
        Uin  : Current state
        time : Current time
        dt   : Time step

      Outputs:
        returns residual vector

      Notes:
        This function currently assumes that Ns=2 and NT=1
      """
      xp      = self.xp_module
      FUin    = xp.zeros_like(Uin)
      
      ele_idx = self.ele_idx
      ion_idx = self.ion_idx
      Te_idx  = self.Te_idx
      
      Te      = Uin[: ,  Te_idx]
      ne      = Uin[: , ele_idx]
      ni      = Uin[: , ion_idx]
      
      mu_i    = self.mu[:, ion_idx]
      mu_e    = self.mu[:, ele_idx]
      
      De      = self.D[:, ele_idx]
      Di      = self.D[:, ion_idx]
      
      Te[0]   = self.param.Teb
      Te[-1]  = self.param.Teb
      
      phi     = self.solve_poisson(Uin[:,ele_idx], Uin[:, ion_idx], time)
      E       = -xp.dot(self.Dp, phi)
      
      
      Us_x    = xp.dot(self.Dp, Uin[: , 0:self.Ns])
      fluxJ    = xp.empty((self.Np, self.Ns))
      for sp_idx in range(self.Ns):
        fluxJ[:, sp_idx] = self.Zp[sp_idx] * self.mu[: , sp_idx] * Uin[: , sp_idx] * E - self.D[: , sp_idx] * Us_x[:, sp_idx]
      
      # fluxJ[:, ele_idx]  = - mu_e * ne * E - De * xp.dot(self.Dp, ne)
      # fluxJ[:, ion_idx]  =   mu_i * ni * E - Di * xp.dot(self.Dp, ni)
      
      fluxJ[0 , ion_idx] = self.mu[0 , ion_idx] * ni[0]  * E[0] 
      fluxJ[-1, ion_idx] = self.mu[-1, ion_idx] * ni[-1] * E[-1] 
        
      fluxJ[0 , ele_idx] = -self.param.ks * ne[0]   - self.param.gamma * fluxJ[0 , ion_idx]
      fluxJ[-1, ele_idx] = self.param.ks  * ne[-1]  - self.param.gamma * fluxJ[-1, ion_idx]
      
      fluxJ_x = xp.dot(self.Dp, fluxJ)
      Te_x    = xp.dot(self.Dp, Te)
      ki      = self.param.ki(Te)  
      
      for sp_idx in range(self.Ns):
        FUin[:,sp_idx] = ki * self.param.n0 * ne - fluxJ_x[:,sp_idx]
      
      Je      = fluxJ[:,ele_idx]
      qe      = -(1.5) * De * ne * Te_x + (2.5) * Te * Je
      qe_x    = xp.dot(self.Dp, qe)
      
      FUin[:  , Te_idx]   = -(2./(3 * ne)) * qe_x - (2 / (3 * ne)) * (Je * E) - (2/3) * self.param.Hi * ki * self.param.n0 - (Te/ne) * (FUin[:,ele_idx])
      FUin[0  , Te_idx]   = 0
      FUin[-1 , Te_idx]   = 0
      
      return FUin  
    
    def rhs_jacobian(self, Uin: np.array, time, dt):
      xp  = self.xp_module
      dof = self.Nv * self.Np
      jac = xp.zeros((dof, dof))
      
      ele_idx = self.ele_idx
      ion_idx = self.ion_idx
      Te_idx  = self.Te_idx
      
      ne      = Uin[: , ele_idx]
      ni      = Uin[: , ion_idx]
      Te      = Uin[: , Te_idx]
      
      mu_i    = self.mu[:, ion_idx]
      mu_e    = self.mu[:, ele_idx]
      
      De      = self.D[:, ele_idx]
      Di      = self.D[:, ion_idx]
      
      Te[0]   = self.param.Teb
      Te[-1]  = self.param.Teb
      
      Np      = self.Np
      Nv      = self.Nv
      
      Imat    = xp.eye(Np)
      Imat[0,0] = Imat[-1,-1] = 0

      phi_ni =  np.linalg.solve(self.LpD, -self.param.alpha*Imat)
      phi_ne = -phi_ni
      
      Imat[0,0] = Imat[-1,-1] = 1.0
      
      E_ni    = -xp.dot(self.Dp, phi_ni)
      E_ne    = -xp.dot(self.Dp, phi_ne)
      
      ki      = self.param.ki(Te)
      ki_Te   = self.param.ki_Te(Te)  
      
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
      
      
      Je_ne        = Js_nk[ele_idx,ele_idx]
      Je_ne[0,0]   = -self.param.ks - self.param.gamma * mu_i[0]   * ni[0]   * E_ne[0 , 0]
      Je_ne[0,1:]  = - self.param.gamma * mu_i[0]   * ni[0]   * E_ne[0 , 1:]
      
      Je_ne[-1,-1] = self.param.ks  -self.param.gamma * mu_i[-1]  * ni[-1]  * E_ne[-1 , -1]  
      Je_ne[-1,1:] = -self.param.gamma * mu_i[-1]  * ni[-1]  * E_ne[-1 , 1:]
      
      Je_ni        = Js_nk[ele_idx,ion_idx]
      Je_ni[0,0]   = - self.param.gamma * mu_i[0]   * (ni[0]   * E_ni[0 , 0] + E[0])
      Je_ni[0,1:]  = - self.param.gamma * mu_i[0]   * (ni[0]   * E_ni[0 , 1:])
          
      Je_ni[-1,-1] = - self.param.gamma * mu_i[-1]  * (ni[-1]  * E_ni[-1 ,-1] + E[-1])
      Je_ni[-1,1:] = - self.param.gamma * mu_i[-1]  * (ni[-1]  * E_ni[-1 , 1:]) 
      
      
      Ji_ni        = Js_nk[ion_idx, ion_idx]
      Ji_ni[0,0]   = mu_i[0]  * (ni[0]  * E_ni[0,0]   + E[0])
      Ji_ni[0,1:]  = mu_i[0]  * (ni[0]  * E_ni[0,1:])
      
      Ji_ni[-1,-1] = mu_i[-1] * (ni[-1] * E_ni[-1,-1] + E[-1])
      Ji_ni[-1,1:] = mu_i[-1] * (ni[-1] * E_ni[-1,1:])
      
      Ji_ne        = Js_nk[ion_idx, ele_idx]
      Ji_ne[0,:]   = mu_i[0]  * ni[0]  * E_ne[0 , :]
      Ji_ne[-1,:]  = mu_i[-1] * ni[-1] * E_ne[-1, :]
      
      Je_x_ne = xp.dot(self.Dp, Je_ne)
      Je_x_ni = xp.dot(self.Dp, Je_ni)
      
      Ji_x_ni = xp.dot(self.Dp, Ji_ni)
      Ji_x_ne = xp.dot(self.Dp, Ji_ne)
      
      Rne_ne  = ki * self.param.n0 * Imat - Je_x_ne
      Rne_ni  = -Je_x_ni
      Rne_Te  = ki_Te * self.param.n0 * ne * Imat 
      
      Rni_ni  = -Ji_x_ni
      Rni_ne  = ki * self.param.n0 * Imat -Ji_x_ne
      Rni_Te  = ki_Te * self.param.n0 * ne * Imat 
      
      Je      = self.Zp[ele_idx] * mu_e * ne * E - De * ne_x
      Je[0 ]  = -self.param.ks * ne[0]   - self.param.gamma * mu_i[0 ] * ni[0]  * E[0]
      Je[-1]  = self.param.ks  * ne[-1]  - self.param.gamma * mu_i[-1] * ni[-1] * E[-1]
      
      Je_x     = xp.dot(self.Dp, Je)
      Te_x     = xp.dot(self.Dp, Te)
      ki       = self.param.ki(Te)  
      
      Rne      = ki * self.param.n0 * ne - Je_x
      
      qe      = -(1.5) * De * ne * Te_x + (2.5) * Te * Je
      qe_ne   = -(1.5) * De * Te_x * Imat + 2.5 * Te * Je_ne
      qe_ni   = 2.5 * Te * Je_ni
      qe_Te   = -(1.5) * De * ne * self.Dp + 2.5 * Je * Imat
      
      qe_x    = xp.dot(self.Dp, qe)
      qe_x_ne = xp.dot(self.Dp, qe_ne)
      qe_x_ni = xp.dot(self.Dp, qe_ni)
      qe_x_Te = xp.dot(self.Dp, qe_Te)
      
      # qe_x_ne = -1.5 * xp.dot(self.Lp, Te) * De * Imat -1.5 * xp.dot(self.Dp, Te) * (De * self.Dp + Imat * xp.dot(self.Dp, De)) + 2.5 * xp.dot(self.Dp, Te) * Je_ne + 2.5 * Te * Je_x_ne
      # qe_x_Te = -1.5 * De * ne * self.Lp - 1.5 * xp.dot(self.Dp, De * ne) * self.Dp + 2.5 * Je * self.Dp + 2.5 * Je_x * Imat
      #JeE_ne       = self.Zp[ele_idx] * mu_e * E**2 * Imat - De * E * self.Dp
      #JeE_ne[0,0]  = -self.param.ks * E[0]  * self.Dp[0,0]
      #JeE_ne[-1,1] = self.param.ks  * E[-1] * self.Dp[-1,-1]
      JeE_ne        = Je * E_ne + E * Je_ne
      JeE_ni        = Je * E_ni + E * Je_ni
      
      RTe_Te  = (-2/3/ne) * qe_x_Te  -(2 /3) * self.param.Hi * self.param.n0 * ki_Te * Imat - (Rne/ne) * Imat - (Te/ne) * Rne_Te
      RTe_ne  = (-2/3/ne) * qe_x_ne + qe_x * (2/3/ne**2) * Imat -(2/3) * (1/ne) * JeE_ne + (2/3)* (Je * E) * (1/ne**2) - (Te/ne) * Rne_ne + (Te * Rne/ne**2) * Imat
      RTe_ni  = (-2/3/ne) * qe_x_ni  -(2/3/ne) * JeE_ni - (Te/ne) * Rne_ni
      
      
      RTe_Te[0  , :] = 0
      RTe_Te[-1 , :] = 0
      
      RTe_ne[0  , :] = 0
      RTe_ne[-1 , :] = 0
      
      RTe_ni[0  , :] = 0
      RTe_ni[-1 , :] = 0
      
      jac[ele_idx :: self.Nv , ele_idx :: self.Nv] = Rne_ne
      jac[ele_idx :: self.Nv , ion_idx :: self.Nv] = Rne_ni
      jac[ele_idx :: self.Nv , Te_idx  :: self.Nv] = Rne_Te
      
      jac[ion_idx :: self.Nv , ele_idx :: self.Nv] = Rni_ne
      jac[ion_idx :: self.Nv , ion_idx :: self.Nv] = Rni_ni
      jac[ion_idx :: self.Nv , Te_idx  :: self.Nv] = Rni_Te
      
      jac[Te_idx :: self.Nv , ele_idx :: self.Nv]  = RTe_ne 
      jac[Te_idx :: self.Nv , ion_idx :: self.Nv]  = RTe_ni
      jac[Te_idx :: self.Nv , Te_idx  :: self.Nv]  = RTe_Te
      
      return jac
      
    def rhs_jacobian_FD(self, Uin, time, dt):
      """
      compute finite differences based jacobian
      """
      xp  = self.xp_module
      r0  = self.rhs(Uin, time, dt).reshape((-1))
      dof = self.Nv * self.Np
      jac = xp.zeros((dof, dof))      

      # perturb each component of Uin to form finite differenc approx
      for j in range(0, self.Np):
        for i in range(0, self.Nv):
          dU = max(xp.finfo(np.float64).eps*xp.absolute(Uin[j,i]), xp.finfo(xp.float64).eps)
          Up = np.copy(Uin)
          Up[j,i] +=dU
          rp = self.rhs(Up, time, dt).reshape((-1))
          jac[j * self.Nv + i , :] = (rp - r0)/dU  
      
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
        r[0]  = 0.0
        r[-1] = xp.sin(2 * xp.pi * time) #+ self.params.verticalShift
        return xp.linalg.solve(self.LpD, r)
    
    def solve(self, Uin, ts_type):
      xp = self.xp_module
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
        dt              = 1.0/xp.ceil(1.0 / (self.args.cfl * dx))
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
        
        # jac= self.rhs_jacobian(u, 0, dt)
        # jac1= self.rhs_jacobian_FD(u, 0, dt)
        # print("jac")
        # print(jac[0::self.Nv, 0::self.Nv])
        # print("FD jac")
        # print(jac1[0::self.Nv, 0::self.Nv])
        # sys.exit(-1)
        
        tt              = 0
        Imat            = xp.eye(self.Np * self.Nv)
        u0              = xp.copy(u)
        for ts_idx in range(steps+1):
          du  = xp.zeros_like(u)
          def residual(du):
            return du - dt * self.rhs(u + du, tt + dt, dt) 
        
          def jacobian(du):
            return Imat - dt * self.rhs_jacobian(u, tt, dt)
          
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
            
          
          # if ts_idx % 1 == 0:
          #   print("ne : ", u[0,0], u[-1,0], " ni, ", u[-1,1], u[-1,1])
          tt+=dt
        return u  
      
      else:
        raise NotImplementedError
      
    def plot(self, Uin):
      fig= plt.figure(figsize=(18,8), dpi=300)
      
      ne = np.abs(Uin[:, self.ele_idx])
      ni = np.abs(Uin[:, self.ion_idx])
      Te = np.abs(Uin[:, self.Te_idx])
      
      plt.subplot(2, 3, 1)
      plt.plot(self.xp, self.param.np0 * ne, 'b')
      plt.xlabel(r"x/L")
      plt.ylabel(r"$n_e (m^{-3})$")
      plt.grid(visible=True)
      
      plt.subplot(2, 3, 2)
      plt.plot(self.xp, self.param.np0 * ni, 'b')
      plt.xlabel(r"x/L")
      plt.ylabel(r"$n_i (m^{-3})$")
      plt.grid(visible=True)
      
      plt.subplot(2, 3, 3)
      plt.plot(self.xp, Te, 'b')
      plt.xlabel(r"x/L")
      plt.ylabel(r"$T_e (eV)$")
      plt.grid(visible=True)
      
      
      plt.subplot(2, 3, 4)
      phi = self.solve_poisson(Uin[:,0], Uin[:,1], 0)
      E   = -np.dot(self.Dp, phi)
      plt.plot(self.xp, E * ((self.param.V0 / self.param.L)), 'b')
      plt.xlabel(r"x/L")
      plt.ylabel(r"$E (V/m)$")
      plt.grid(visible=True)
      
      plt.subplot(2, 3, 5)
      plt.plot(self.xp, phi * (((self.param.V0))), 'b')
      plt.xlabel(r"x/L")
      plt.ylabel(r"$\phi (V)$")
      plt.grid(visible=True)
      
      plt.tight_layout()
      
      fig.savefig("1d_glow.png")
    
    
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
parser.add_argument("-max_iter", "--max_iter"           , help="max iterations for Newton solver" , type=int, default=1000)

args      = parser.parse_args()
glow_1d   = glow1d_fluid(args)

u         = glow_1d.initialize()
v         = glow_1d.solve(u, ts_type=args.ts_type)
glow_1d.plot(v)
np.save("%s.npy"%(args.fname), v)



