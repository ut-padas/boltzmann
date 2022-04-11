"""
@package : Explicit ode integrator with specified coefficients. 
"""

import enum
import numpy as np

class TSType(enum.Enum):
    FORWARD_EULER = 0
    RK2           = 1
    RK3           = 2 
    RK4           = 3

class ExplicitODEIntegrator():
    
    def __init__(self, type):
        """
        initialize the time integrator object. 
        """
        self._rhs_func = None
        self._t_coord  = 0
        self._t_step   = 0 
        self._dt       = 0 
        self._type     = type  
        
        if self._type   == TSType.FORWARD_EULER:
            self._k_stages = 1
            self._k_c      = 1.0
            self._k_aij    = None
            self._k_bi     = None
        elif self._type == TSType.RK2:
            pass
        elif self._type == TSType.RK3:
            pass
        elif self._type == TSType.RK4:
            self._k_stages = 4
            self._k_c      = np.array([1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0])
            self._k_aij    = np.array([[0.0     , 0.0  , 0.0   , 0.0],
                                       [1.0/2.0 , 0.0  , 0.0   , 0.0],
                                       [0.0 , 1.0/2.0  , 0.0   , 0.0],
                                       [0.0 , 0.0      , 1.0   , 0.0]])
            self._k_bi     = np.array([0,1.0/2.0,1.0/2.0,1.0])
            
    
    def set_rhs_func(self,func):
        """
        set the RHS function. 
        """
        self._rhs_func = func

    def set_ts_size(self,dt):
        """
        set the dt value. 
        """
        self._dt = dt
    
    def init(self):
        """
        initialize the timestepper
        """
        self._t_coord  = 0
        self._t_step   = 0 
        
    def current_ts(self):
        """
        get the current timestep info
        [current time, step number]
        """
        return [float(self._t_coord), int(self._t_step)]
    
    def evolve(self,u):
        """
        Note : we can do this in place, but left this to be 
        more generic, ts. 
        """
        v = np.zeros(u.shape)
        if self._type == TSType.FORWARD_EULER:
            v =  u + self._dt * self._rhs_func(u,self._t_coord)
            self._t_coord += self._dt
            self._t_step  += 1

        elif self._type ==  TSType.RK4:
            k1 = self._rhs_func(u,self._t_coord)
            k2 = self._rhs_func(u + 0.5 * self._dt * k1 , self._t_coord + 0.5*self._dt)
            k3 = self._rhs_func(u + 0.5 * self._dt * k2 , self._t_coord + 0.5*self._dt)
            k4 = self._rhs_func(u + 1.0 * self._dt * k3 , self._t_coord + self._dt)

            v = u + (self._dt/6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            self._t_coord += self._dt
            self._t_step  += 1

        return v 

        

        




    

        

        