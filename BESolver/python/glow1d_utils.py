"""
simple class to hold glow discharge parameters
"""
import numpy as np
import scipy.constants

class parameters():
  def __init__(self) -> None:
    xp         = np
    
    self.L     = 0.5 * 2.54e-2             # m 
    self.V0    = 100                       # V
    self.f     = 13.6e6                    # Hz
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
    
    self.n0    /=self.np0
    self.De    *= self.tau/self.L**2 
    self.mu_e  *= self.V0 * self.tau/self.L**2 
    self.Di    *= self.tau/self.L**2 
    self.mu_i  *= self.V0 * self.tau/self.L**2 
    self.ks    *= self.tau/self.L
    
    self.Teb   = 0.5                       # eV
    self.Hi    = 15.76                     # eV
    #self.qe    = 0.0
    
    self.Tg    = 0.0
    self.gamma = 0.01
    self.alpha = self.np0 * self.L**2 * self.qe / self.eps0 / self.V0
    
    self.ki    = lambda Te : self.np0 * self.tau * 1.235e-13 * np.exp(-18.687 / np.abs(Te))   
    self.ki_Te = lambda Te : self.np0 * self.tau * 1.235e-13 * np.exp(-18.687 / np.abs(Te))  * (18.687/np.abs(Te)**2)  
    

def newton_solver(x, residual, jacobian, atol, rtol, iter_max, xp=np):
  r0       = residual(x).reshape(-1)  
  jac      = jacobian(x)
  
  norm_r0  = xp.linalg.norm(r0)
  
  norm_rr  = norm_r0 = np.linalg.norm(r0)
  count    = 0
  alpha    = 1
  
  ns_info  = dict()
  
  converged = ((norm_rr/norm_r0 < rtol) or (norm_rr < atol))
  while( not converged and (count < iter_max) ):
    rr       = residual(x).reshape(-1)
    norm_rr  = xp.linalg.norm(rr)
    
    alpha    = 1
    x1       = x + alpha * xp.linalg.solve(jac, -rr).reshape(x.shape)
    # norm_rr1 = xp.linalg.norm(residual(x1).reshape(-1))
    # while(norm_rr1 > norm_rr):
    #   alpha   *= 2
    #   rr1      = residual(x1).reshape(-1)
    #   x1       = x + alpha * xp.linalg.solve(jac, -rr1).reshape(x.shape)
      
    #   norm_rr1 = xp.linalg.norm(residual(x1).reshape(-1))
      
    
    #   if (alpha < 1e-6):
    #     # print(alpha, norm_rr1, norm_rr)
    #     print("  {0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e} ".format(count, norm_rr, norm_rr/norm_r0))
    #     print("non-linear solver line search FAILED!!! try with smaller time step size or increase max iterations")
        
    #     ns_info["status"] = converged
    #     ns_info["x"]      = x
    #     ns_info["atol"]   = norm_rr
    #     ns_info["rtol"]   = norm_rr/norm_r0
    #     ns_info["alpha"]  = alpha
    #     ns_info["iter"]   = count
    #     return ns_info
    x = x1
    count   += 1
    
    #if count%1000==0:
    #print("{0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}".format(count, norm_rr, norm_rr/norm_r0), "alpha ", alpha)
    
    converged = ((norm_rr/norm_r0 < rtol) or (norm_rr < atol))
  
  #print("  Newton iter {0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}".format(count, norm_rr, norm_rr/norm_r0))
    
  if (not converged):
    # if non-convergence encountered, save state and die
    print("  {0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}".format(count, norm_rr, norm_rr/norm_r0))
    print("non-linear solver step FAILED!!! try with smaller time step size or increase max iterations")
    
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