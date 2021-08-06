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
        self.VEL_SPACE_POLY_ORDER  = 2
        self.VELOCITY_SPACE_DIM    = 3
        self.TIME_STEP_SIZE        = 0.01

        # Position space parameters. 
        self.X_SPACE_DIM           = 3
        self.X_GRID_MIN            = np.array([-1,-1,-1])
        self.X_GRID_MAX            = np.array([1,1,1])
        self.X_GRID_RES            = np.array([0.1,0.1,0.1])
        self.X_DERIV_ORDER         = 2

        self.TOTAL_TIME            = 10
        self.IO_STEP_FREQ          = 10
        self.IO_FILE_NAME_PREFIX   = f'bplots/u_sol_%04d.png'
        
        
        # physical parameters. 
        self.UNITS                      =  pint.UnitRegistry()
        self.STEADY_STATE_TEMPERATURE   =  1000 * self.UNITS.kelvin
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

def maxwellian_normalized(v_abs):
    """
    Normalized Maxwellian without 
    any properties of the application parameters. 
    v_abs - norm(v), scalar

    The steady state maxwellian depends on, 
    mass of the particle, k_B, steady state temperature, and the number density
    M(vp) = A exp(-m * vp**2/ 2 * k_B * T)

    For the computations we can use the normalized maxwellian and use 
    v =  \sqrt(m/(k_B * T)) vp go to application specific maxwellian. 

    """
    #v_l2 = np.linalg.norm(v,2)
    return np.exp(-0.5*(v_abs**2))


def gaussian(v,mu=None,sigma=1.0):
    """
    Gaussian distribution function for d dim
    v  = np.array (dim,)
    mu = None, then mu=0 will be used. 
    sigma =1.0 standard deviation
    """
    if mu is None:
        mu = np.zeros(v.shape)
    return ( 1.0/(sigma * np.sqrt(2 * np.pi)) ) * np.exp(-0.5 * np.linalg.norm((v-mu),2)**2/(sigma**2))
