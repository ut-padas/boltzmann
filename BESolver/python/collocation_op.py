"""
Chebyshev collocation basis in 1D
"""
import numpy as np
from numpy.polynomial import chebyshev as cheb
xp = np

class cheb_collocation_1d():
    
    def __init__(self, Np) -> None:
        self.Np  = Np
        self.deg = self.Np-1  # degree of Chebyshev polys we use
        self.xp  = -np.cos(xp.pi*np.linspace(0,self.deg,self.Np)/self.deg)
        
        # Operators
        ident = np.identity(self.Np)

        # V0p: Coefficients to values at xp
        self.V0p = np.polynomial.chebyshev.chebvander(self.xp, self.deg)

        # V0pinv: xp values to coefficients
        self.V0pinv = np.linalg.solve(self.V0p, ident)

        # V1p: coefficients to derivatives at xp
        # self.V1p = np.zeros((self.Np,self.Np))
        # for i in range(0,self.Np):
        #     self.V1p[:,i] = cheb.chebval(self.xp, cheb.chebder(ident[i,:], m=1))
        self.V1p  = np.array([np.polynomial.Chebyshev.basis(i).deriv(m=1)(self.xp) for i in range(self.Np)]).reshape((-1, self.Np)).T  

        # Dp: values at xp to derivatives at xp
        self.Dp = self.V1p @ self.V0pinv
        
        # V2p: coefficients to 2nd derivatives at xp
        # self.V2p = np.zeros((self.Np,self.Np))
        # for i in range(0,self.Np):
        #     self.V2p[:,i] = cheb.chebval(self.xp, cheb.chebder(ident[i,:], m=2))
        self.V2p  = np.array([np.polynomial.Chebyshev.basis(i).deriv(m=2)(self.xp) for i in range(self.Np)]).reshape((-1, self.Np)).T  
        self.V3p  = np.array([np.polynomial.Chebyshev.basis(i).deriv(m=3)(self.xp) for i in range(self.Np)]).reshape((-1, self.Np)).T  
        self.V4p  = np.array([np.polynomial.Chebyshev.basis(i).deriv(m=4)(self.xp) for i in range(self.Np)]).reshape((-1, self.Np)).T  
        

        # Lp: values at xp to 2nd derivatives at xp
        self.Lp  = self.V2p @ self.V0pinv
        self.Lp3 = self.V3p @ self.V0pinv
        self.Lp4 = self.V4p @ self.V0pinv
        
        # LpD: values at xp to 2nd derivatives at xc, with identity
        # for top and bottom row (for Dirichlet BCs)
        self.LpD = np.identity(self.Np)
        self.LpD[1:-1,:] = self.Lp[1:-1,:]
        self.LpD_inv     = np.linalg.solve(self.LpD, np.eye(self.Np)) 
        
        

    

