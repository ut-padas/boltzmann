"""
macroscopic/microscopic modeling of the 1d glow discharge problem
1). We use Gauss-Chebyshev-Lobatto co-location method with implicit time integration. 
"""
import numpy as np
import scipy.constants 
import argparse
import matplotlib.pyplot as plt

class params():
  """
  simple class to hold 1d glow discharge parameters
  """
  def __init__(self) -> None:
    xp         = np
    
    self.L     = 0.5 * 2.54e-2             # m 
    self.V0    = 100                       # V
    self.f     = 13.56e6                   # Hz
    self.tau   = (1/self.f)                # s
    self.qe    = scipy.constants.e         # C
    self.eps0  = scipy.constants.epsilon_0 # eps_0 
    
    self.n0    = 3.22e22                   #m^{-3}
    self.np0   = 8e16                      #"nominal" electron density [1/m^3]
    
    # raw transport coefficients 
    self.De    = (3.86e22) * 1e2 / self.n0 #m^{2}s^{-1}
    self.mu_e  = (9.66e21) * 1e2 / self.n0 #V^{-1} m^{2} s^{-1} 
    self.Di    = (2.07e18) * 1e2 / self.n0 #m^{2} s^{-1}
    self.mu_i  = (4.65e19) * 1e2 / self.n0 #V^{-1} m^{2} s^{-1}
    self.ks    = 1.19e5                    # m s^{-1}
    
    # non-dimensionalized transport coefficients
    
    self.De    *= self.tau/self.L**2 
    self.mu_e  *= self.V0 * self.tau/self.L**2 
    self.Di    *= self.tau/self.L**2 
    self.mu_i  *= self.V0 * self.tau/self.L**2 
    self.ks    *= self.tau/self.L
    
    self.Teb   = 0.5                       # eV
    self.Hi    = 15.76                     # eV
    self.gamma = 0.01
    self.alpha = self.np0 * self.L**2 * self.qe / self.eps0 / self.V0
    
    self.ki    = lambda Te : 1.235e-12 * np.exp(-18.687 / Te) # m^{-3} s^{-1}
    self.ki_Te = lambda Te : 1.235e-12 * np.exp(-18.687 / Te) * (18.687/Te**2)
    
class Glow1D():
    def __init__(self, args) -> None:
      self.args  = args
      self.param = params()
      
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
        Uin[:, ele_idx] = 1e-3
        Uin[:, ion_idx] = 1e-3 
        Uin[:, Te_idx]  = self.param.Teb
        
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
      
      
      FUin[1:-1,Te_idx]   = -(2./(3 * ne[1:-1])) * qe_x[1:-1] - (2 * scipy.constants.e / (3 * ne[1:-1])) * (Je[1:-1] * E[1:-1]) - (2/3) * self.param.Hi * ki[1:-1] * self.param.n0 - (Te[1:-1]/ne[1:-1]) * (FUin[1:-1,ele_idx])
      
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
      ki      = self.param.ki(Te)  
      
      phi     = self.solve_poisson(ne, ni, time)
      E       = -xp.dot(self.Dp, phi)
      
      Rne_ne  = ki * self.param.n0 * Imat - self.Zp[ele_idx] * xp.dot(self.Dp, mu_e * E) * Imat -self.Zp[ele_idx] * mu_e * E * self.Dp + De * self.Lp + xp.dot(self.Dp, De) * self.Dp
      Rne_ni  = xp.zeros((self.Np, self.Np))
      Rne_Te  = self.param.ki_Te(Te) * self.param.n0 * ne * Imat 
      
      Rne_ne[0,0]   = -self.param.ks
      Rne_ne[-1,-1] = self.param.ks
      
      Rne_ni[0  , 0]   = -self.param.gamma * self.mu[0,ion_idx] * E[0]
      Rne_ni[-1 , -1]  = -self.param.gamma * self.mu[-1,ion_idx] * E[-1]
      
      Rne_Te[0,0]      = 0
      Rne_Te[-1,-1]    = 0
      
      Rni_ni         = - self.Zp[ion_idx] * xp.dot(self.Dp, mu_i * E) * Imat -self.Zp[ion_idx] * mu_i * E * self.Dp + Di * self.Lp + xp.dot(self.Dp, Di) * self.Dp
      Rni_ne         = ki * self.param.n0 * Imat
      Rni_Te         = self.param.ki_Te(Te) * self.param.n0 * ne * Imat 
      
      Rni_ni[0 , 0]  = self.Zp[ion_idx] * mu_i[0] * E[0]
      Rni_ni[-1,-1]  = self.Zp[ion_idx] * mu_i[-1] * E[-1]
      
      Rni_ne[0 , 0]  = 0
      Rni_ne[-1,-1]  = 0
      
      Rni_Te[0,0]    = 0
      Rni_Te[-1,-1]  = 0
      
      ne_x       = xp.dot(self.Dp, ne)
      fluxJe     = self.Zp[ele_idx] * mu_e * ne * E - De * ne_x
      fluxJe[0 ] = -self.param.ks * ne[0]   - self.param.gamma * mu_i[0 ] * ni[0]  * E[0]
      fluxJe[-1] = self.param.ks  * ne[-1]  - self.param.gamma * mu_i[-1] * ni[-1] * E[-1]
      
      fluxJe_x = xp.dot(self.Dp, fluxJe)
      Te_x     = xp.dot(self.Dp, Te)
      ki       = self.param.ki(Te)  
      
      Rne      = ki * self.param.n0 * ne - fluxJe_x
      Je       = fluxJe
      
      Je_x_ne  = self.Zp[ele_idx] * xp.dot(self.Dp, mu_e * E) * Imat + self.Zp[ele_idx] * mu_e * E * self.Dp - De * self.Lp -xp.dot(self.Dp, De) * self.Dp
      Je_x_ne[0,0]   = -self.param.ks
      Je_x_ne[-1,-1] = self.param.ks
      
      qe      = -(1.5) * De * ne * Te_x + (2.5) * Te * Je
      qe_x    = xp.dot(self.Dp, qe)
      
      qe_x_Te = (-1.5 * De * ne * self.Lp -1.5 * xp.dot(self.Dp, De * ne) * self.Dp + 2.5 * Je * self.Dp + 2.5 * xp.dot(self.Dp, Je) *Imat )
      
      qe_x_ne = -1.5 * xp.dot(self.Lp, Te) * De * Imat - 1.5 * xp.dot(self.Dp, Te) * (De * self.Dp + Imat * xp.dot(self.Dp, De)) + 2.5 * xp.dot(self.Dp, Te) * (self.Zp[ele_idx] * mu_e * E * Imat - De * self.Dp) + 2.5 * Te * Je_x_ne
      
      
      JeE_ne       = self.Zp[ele_idx] * mu_e * E**2 * Imat - De * E * self.Dp
      JeE_ne[0,0]  = -self.param.ks * E[0]
      JeE_ne[-1,1] = self.param.ks  * E[-1]
      
      RTe_Te  = (-2/3/ne) * qe_x_Te  -(2/3) * self.param.Hi * self.param.n0 * self.param.ki_Te(Te) * Imat - (Rne/ne) * Imat - (Te/ne) * Rne_Te
      RTe_ne  = (-2/3/ne) * qe_x_ne + qe_x * (2/3/ne**2) * Imat -(2/3/ne) * scipy.constants.e * JeE_ne + Je * E * (2 * scipy.constants.e/3/ne**2) - (Te/ne) * Rne_ne + (Te * Rne/ne**2) * Imat
      
      RTe_Te[0  , :] = 0
      RTe_Te[-1 , :] = 0
      
      RTe_ne[0  , :] = 0
      RTe_ne[-1 , :] = 0
      
      jac[ele_idx :: self.Nv , ele_idx :: self.Nv] = Rne_ne
      jac[ele_idx :: self.Nv , ion_idx :: self.Nv] = Rne_ni
      jac[ele_idx :: self.Nv , Te_idx  :: self.Nv] = Rne_Te
      
      jac[ion_idx :: self.Nv , ion_idx :: self.Nv] = Rni_ni
      jac[ion_idx :: self.Nv , ele_idx :: self.Nv] = Rni_ne
      jac[ion_idx :: self.Nv , Te_idx  :: self.Nv] = Rni_Te
      
      jac[Te_idx :: self.Nv , Te_idx  :: self.Nv]  = RTe_Te
      jac[Te_idx :: self.Nv , ele_idx :: self.Nv]  = RTe_ne
      
      
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
          dU = max(xp.sqrt(xp.finfo(np.float64).eps)*xp.absolute(Uin[j,i]), xp.sqrt(xp.finfo(xp.float64).eps))
          Up = np.copy(Uin)
          Up[j,i] +=dU
          rp = self.rhs(Up, time, dt).reshape((-1))
          jac[j * self.Nv + i , :] = (rp - r0)/dU  
      
       #print(jac)
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
        dt              = self.args.cfl * xp.min(self.xp[1:] - self.xp[0:-1])
        tT              = (1/self.param.f) * self.args.cycles
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
        dt              = self.args.cfl * xp.min(self.xp[1:] - self.xp[0:-1])
        tT              = (1/self.param.f) * self.args.cycles
        steps           = int(tT/dt)
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
        dt              = self.args.cfl * xp.min(self.xp[1:] - self.xp[0:-1])
        tT              = (1/self.param.f) * self.args.cycles
        steps           = max(int(tT/dt),1)
        u               = xp.copy(Uin)
        
        print(tT, dt, steps)
        
        tt              = 0
        Imat            = xp.eye(self.Np * self.Nv)
        for ts_idx in range(steps):
          if ts_idx % 1000 == 0:
            print("time = %.2E step=%d/%d"%(tt, ts_idx, steps))
          
          r0  = dt * self.rhs(u,tt,dt).reshape(-1)  
          #print(self.rhs_jacobian(u, tt, dt))
          jac = Imat - dt * self.rhs_jacobian(u, tt, dt)
          du  = np.linalg.solve(jac,r0).reshape((self.Np, self.Nv))
          u   = u + du
          tt+=dt
        return u  
      
      else:
        raise NotImplementedError
      
    def plot(self, Uin):
      fig= plt.figure(figsize=(18,4), dpi=300)
      
      ne = np.abs(Uin[:, self.ele_idx])
      ni = np.abs(Uin[:, self.ion_idx])
      Te = np.abs(Uin[:, self.Te_idx])
      
      plt.subplot(1, 4, 1)
      plt.plot(self.xp, self.param.np0 * ne, 'b')
      plt.xlabel(r"x/L")
      plt.ylabel(r"$n_e (m^{-3})$")
      plt.grid(visible=True)
      
      plt.subplot(1, 4, 2)
      plt.plot(self.xp, self.param.np0 * ni, 'b')
      plt.xlabel(r"x/L")
      plt.ylabel(r"$n_i (m^{-3})$")
      plt.grid(visible=True)
      
      plt.subplot(1, 4, 3)
      plt.plot(self.xp, Te, 'b')
      plt.xlabel(r"x/L")
      plt.ylabel(r"$T_e (eV)$")
      plt.grid(visible=True)
      
      
      plt.subplot(1, 4, 4)
      phi = self.param.V0 * self.solve_poisson(Uin[:,0], Uin[:,1], 0) / self.param.L**2
      E = -np.dot(self.Dp, phi)
      plt.plot(self.xp, E, 'b')
      plt.xlabel(r"x/L")
      plt.ylabel(r"$E (V/m)$")
      plt.grid(visible=True)
      
      plt.tight_layout()
      
      fig.savefig("1d_glow.png")
            
parser = argparse.ArgumentParser()
parser.add_argument("-Ns", "--Ns"                       , help="number of species"      , type=int, default=2)
parser.add_argument("-NT", "--NT"                       , help="number of temperatures" , type=int, default=1)
parser.add_argument("-Np", "--Np"                       , help="number of collocation points" , type=int, default=100)
parser.add_argument("-cfl", "--cfl"                     , help="CFL factor (only used in explicit integrations)" , type=float, default=1e-1)
parser.add_argument("-cycles", "--cycles"               , help="number of cycles to run" , type=int, default=10)
parser.add_argument("-ts_mode", "--ts_mode"             , help="ts mode" , type=str, default="RK4")

args = parser.parse_args()
glow_1d = Glow1D(args)

# x = glow_1d.xp
# f = x**3 + np.sin(x)
# f_x  = np.dot(glow_1d.Dp, f)
# f_xx = np.dot(glow_1d.Dp, np.dot(glow_1d.Dp, f))
 
# plt.plot(x, f_x   , label="Dx f")
# plt.plot(x, f_xx   , label="Dxx f")
# plt.plot(x, 3 * x**2 + np.cos(x) , label="f'")
# plt.plot(x, 6 * x**1 - np.sin(x) , label="f''")
# plt.grid(visible=True)
# plt.legend()
# plt.show()



u       = glow_1d.initialize()
v       = glow_1d.solve(u, ts_type=args.ts_mode)
print(v)
glow_1d.plot(v)



