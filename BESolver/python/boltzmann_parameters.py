"""
@package: Velocity space Petrov-Galerkin spectral discretization collission
operator in Boltzmann equation. (0d space)
"""
import scipy.constants
import numpy as np
import basis
import spec as sp
import pint

class BoltzmannEquationParameters:
    """
    @todo : need to enable initialization from a proper parameter file
    Stores the application parameters for the 
    Boltzmann Solver
    """
    def __init__(self):
        self.POLY_ORDER         = 3
        self.VELOCITY_SPACE_DIM = 3
        self.POSITION_SPACE_DIM = 0
        
        # physical parameters. 
        self.UNITS                      =  pint.UnitRegistry()
        self.STEADY_STATE_TEMPERATURE   =  100 * self.UNITS.kelvin
        self.BOLTSMANN_CONSTANT         =  scipy.constants.Boltzmann  * self.UNITS.meter**2 * self.UNITS.kilogram / (self.UNITS.second**2 * self.UNITS.kelvin)
        self.ELECTRON_MASS              =  scipy.constants.electron_mass * self.UNITS.kilogram

        # note this is distribution dependent (need to integrate initial distribution over velocity space)
        self.ELECTRON_NUMBER_DENSITY    =  1E6
        #self.ELECTRON_CHARGE            =  scipy.constants.elementary_charge * self.UNITS.columb


#PARS = BoltzmannEquationParameters()
# def maxwellian3d_electron(v):
#     me = PARS.ELECTRON_MASS.magnitude()
#     T  = PARS.STEADY_STATE_TEMPERATURE.magnitude()
#     K  = PARS.BOLTSMANN_CONSTANT.magnitude()
#     N  = PARS.ELECTRON_NUMBER_DENSITY

#     vth = np.sqrt(2*K*T/me)
#     v_l2 = np.linalg.norm(v,2)
#     return N/((np.sqrt(np.pi)*vth)**3) * np.exp(-(v_l2/vth)**2)

def maxwellian_normalized(v):
    """
    Normalized Maxwellian without 
    any properties of the application parameters. 
    """
    v_l2 = np.linalg.norm(v,2)
    return np.exp(-0.5*(v_l2**2))





        
      

    




    