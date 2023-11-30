"""
simple class to hold glow discharge parameters
"""
import numpy as np
import scipy.constants
#from multiprocess import Pool as WorkerPool
from multiprocessing.pool import ThreadPool as WorkerPool
import scipy.sparse.linalg as spla
try:
  import cupy as cp
except ImportError:
  print("Please install CuPy for GPU use")

class parameters():
  def __init__(self) -> None:
    xp         = np
    
    self.L     = 0.5 * 2.54e-2             # m 
    self.V0    = 1e2                       # V
    self.f     = 13.56e6                   # Hz
    self.tau   = (1/self.f)                # s
    self.qe    = scipy.constants.e         # C
    self.eps0  = scipy.constants.epsilon_0 # eps_0 
    self.kB    = scipy.constants.Boltzmann # J/K
    self.ev_to_K = scipy.constants.electron_volt / scipy.constants.Boltzmann
    self.me    = scipy.constants.electron_mass
    
    self.n0    = 3.22e22                   #m^{-3}
    self.np0   = 8e16                      #"nominal" electron density [1/m^3]
    
    # raw transport coefficients 
    self.De    = (3.86e22) * 1e2 / self.n0 #m^{2}s^{-1}
    self.mu_e  = (9.66e21) * 1e2 / self.n0 #V^{-1} m^{2} s^{-1} 
    self.Di    = (2.07e18) * 1e2 / self.n0 #m^{2} s^{-1}
    self.mu_i  = (4.65e19) * 1e2 / self.n0 #V^{-1} m^{2} s^{-1}
    self.ks    = (1.19e7)  * 1e-2 #(1.366109824889323e7) * 1e-2          # m s^{-1}
    
    # non-dimensionalized transport coefficients
    
    self.n0    /=self.np0
    self.De    *= (self.tau/(self.L**2))
    self.Di    *= (self.tau/(self.L**2))
    
    self.mu_e  *= (self.V0 * self.tau/(self.L**2)) 
    self.mu_i  *= (self.V0 * self.tau/(self.L**2))
    
    self.Teb   = 0.5                                # eV
    self.Hi    = 15.76                              # eV
    self.ks    *= (self.tau/self.L)
    
    self.ks0   = self.ks
    self.ks1   = self.ks
    
    self.Teb0  = self.Teb
    self.Teb1  = self.Teb
    
    self.Tg    = 300.0
    self.gamma = 0.0
    self.alpha = self.np0 * self.L**2 * self.qe / (self.eps0 * self.V0)
    
    # self.ki    = lambda Te : self.np0 * self.tau * 1.235e-13 * np.exp(-18.687 / np.abs(Te))   
    # self.ki_Te = lambda Te : self.np0 * self.tau * 1.235e-13 * np.exp(-18.687 / np.abs(Te))  * (18.687/np.abs(Te)**2)
    
    self.ki     = lambda nTe,ne : self.np0 * self.tau * 1.235e-13 * np.exp(-18.687 * np.abs(ne / nTe))   
    self.ki_nTe = lambda nTe,ne : self.np0 * self.tau * 1.235e-13 * np.exp(-18.687 * np.abs(ne / nTe))  * (18.687 * ne / nTe**2)  
    self.ki_ne  = lambda nTe,ne : self.np0 * self.tau * 1.235e-13 * np.exp(-18.687 * np.abs(ne / nTe))  * (-18.687 / nTe)
    
    # non-dimentionalized maxwellian flux. 
    self.mw_flux    = lambda Te : (self.tau/self.L) * (self.kB * Te * self.ev_to_K / (2 * np.pi * self.me)) ** 0.5
    #self.mw_flux_Te = lambda Te : 0.5 * (self.kB * self.ev_to_K / (2 * np.pi * self.me)) * (self.kB * Te * self.ev_to_K / (2 * np.pi * self.me)) ** (-0.5)      
    
def newton_solver(x, residual, jacobian, atol, rtol, iter_max, xp=np):
  x0       = xp.copy(x)
  jac      = jacobian(x0)
  jac_inv  = xp.linalg.inv(jac)
  
  ns_info  = dict()
  alpha    = 1.0e0
  while(alpha > 1e-8):
    x        = x0
    count    = 0
    r0       = residual(x)
    
    norm_rr  = norm_r0 = xp.linalg.norm(r0)
    converged = ((norm_rr/norm_r0 < rtol) or (norm_rr < atol))
    
    while( not converged and (count < iter_max) ):
      rr       = residual(x)
      norm_rr  = xp.linalg.norm(rr)
      x        = x  + alpha * xp.dot(jac_inv, -rr).reshape(x.shape)
      
      count   += 1
      #if count%1000==0:
      #print("{0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}".format(count, norm_rr, norm_rr/norm_r0), "alpha ", alpha)
      converged = ((norm_rr/norm_r0 < rtol) or (norm_rr < atol))
    
    if (not converged):
      alpha *= 0.25
      #print(alpha)
      
    else:
      #print("  Newton iter {0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}".format(count, norm_rr, norm_rr/norm_r0))
      break
  
  if (not converged):
    # solver failed !!!
    print("  {0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}".format(count, norm_rr, norm_rr/norm_r0))
    print("non-linear solver step FAILED!!! try with smaller time step size or increase max iterations")
    print(rr.reshape(x.shape))
    ns_info["status"] = converged
    ns_info["x"]      = x
    ns_info["atol"]   = norm_rr
    ns_info["rtol"]   = norm_rr/norm_r0
    ns_info["alpha"]  = alpha
    ns_info["iter"]   = count
    return ns_info
  
  ns_info["status"] = converged
  ns_info["x"]      = x
  ns_info["atol"]   = norm_rr
  ns_info["rtol"]   = norm_rr/norm_r0
  ns_info["alpha"]  = alpha
  ns_info["iter"]   = count
  return ns_info

def newton_solver_batched(x, n_pts, residual, jacobian, atol, rtol, iter_max, num_processes=4, xp=np):
  jac      = jacobian(x)
  assert jac.shape[0] == n_pts
  jac_inv  = xp.linalg.inv(jac)
  
  # def t1(i):
  #   jac_inv[i] = xp.linalg.inv(jac[i])
  #   return
  
  # pool = WorkerPool(num_processes)    
  # pool.map(t1,[i for i in range(n_pts)])
  # pool.close()
  # pool.join()
  
  ns_info  = dict()
  alpha    = xp.ones(n_pts)
  while((alpha > 1e-10).any()):
      count     = 0
      r0        = residual(x)
      norm_rr   = norm_r0 = xp.linalg.norm(r0, axis=0)
      converged = ((norm_rr/norm_r0 < rtol).all() or (norm_rr < atol).all())
      
      while( not converged and (count < iter_max) ):
          rr        = residual(x)
          norm_rr   = xp.linalg.norm(rr, axis=0)
          converged = ((norm_rr/norm_r0 < rtol).all() or (norm_rr < atol).all())
          
          x         = x + alpha * xp.einsum("ijk,ki->ji", jac_inv, -rr)
          #   for i in range(n_pts):
          #     x[:,i] = x[:,i] + alpha[i] * xp.dot(jac_inv[i], -rr[:,i])
          
          count   += 1
          #if count%1000==0:
          #print("{0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}".format(count, norm_rr, norm_rr/norm_r0), "alpha ", alpha)
          
      if (not converged):
          alpha *= 0.25
          #print(alpha)
      else:
          #print("  Newton iter {0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}".format(count, norm_rr, norm_rr/norm_r0))
          break
  
  if (not converged):
      # solver failed !!!
      print("  {0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}".format(count, xp.max(norm_rr), xp.max(norm_rr/norm_r0)))
      print("non-linear solver step FAILED!!! try with smaller time step size or increase max iterations")
      #print(rr)
      print(alpha)
      print(norm_r0)
      print(norm_rr)
      print(norm_rr/norm_r0)
      ns_info["status"] = converged
      ns_info["x"]      = x
      ns_info["atol"]   = xp.max(norm_rr)
      ns_info["rtol"]   = xp.max(norm_rr/norm_r0)
      ns_info["alpha"]  = alpha
      ns_info["iter"]   = count
      return ns_info
  
  ns_info["status"] = converged
  ns_info["x"]      = x
  ns_info["atol"]   = xp.max(norm_rr)
  ns_info["rtol"]   = xp.max(norm_rr/norm_r0)
  ns_info["alpha"]  = alpha
  ns_info["iter"]   = count
  return ns_info
