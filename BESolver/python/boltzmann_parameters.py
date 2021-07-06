"""
@package: Velocity space Petrov-Galerkin spectral discretization collission
operator in Boltzmann equation. (0d space)
"""
import scipy.constants
import numpy as np
import basis
import spec as sp
import pint

class BESolverParameters:
    """
    @todo : need to enable initialization from a proper parameter file
    Stores the application parameters for the 
    Boltzmann Solver
    """
    def __init__(self):
        # numerical solver parameters
        self._order= 3
        self._vdim = 3
        self._xdim = 0

        # physical parameters. 
        self._ureg = pint.UnitRegistry()
        self._T = 100 * self._ureg.kelvin
        self._Kb = scipy.constants.Boltzmann  * self._ureg.meter**2 * self._ureg.kilogram / (self._ureg.second**2 * self._ureg.kelvin)
        self._Me = scipy.constants.electron_mass * self._ureg.kilogram
        
      

    




    