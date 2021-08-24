"""
@package: Simple structure to maintain simulation parameters
"""
import scipy.constants
import numpy as np
import basis
import spec as sp
import pint
import collisions
import scipy.constants 

#http://farside.ph.utexas.edu/teaching/plasma/Plasmahtml/node6.html
PLASMA_FREQUENCY  = np.sqrt(collisions.MAXWELLIAN_N * (scipy.constants.elementary_charge**2) / (scipy.constants.epsilon_0  * scipy.constants.electron_mass))

class BEVelocitySpace():

    VELOCITY_SPACE_POLY_ORDER  = 1
    VELOCITY_SPACE_DIM         = 3
    VELOCITY_SPACE_DT          = 1e-15#0.01/PLASMA_FREQUENCY  
    IO_FILE_NAME_PREFIX        = f'plots/f_sol_%08d.png'
    IO_STEP_FREQ               = 1e5
    #SPH_HARM_LM               = [[0,0], [1,-1], [1,0], [1,1], [2,-2],[2,-1],[2,0],[2,1],[2,2]] # , [3,0], [4,0]]
    SPH_HARM_LM                = [[0,0], [1,0]]
    #SPH_HARM_LM               = [[0,0]]
    NUM_Q_VR                   = 10
    NUM_Q_VT                   = 10
    NUM_Q_VP                   = 5
    NUM_Q_CHI                  = 512
    NUM_Q_PHI                  = 5
    NUM_Q_PTS_ON_V             = 20
    NUM_Q_PTS_ON_SPHERE        = 10 #np.max(NUM_Q_CHI,NUM_Q_VT)

    


    



class BEPositionSpace():
    X_SPACE_DIM           = 3
    X_GRID_MIN            = np.array([-1,-1,-1])
    X_GRID_MAX            = np.array([1,1,1])
    X_GRID_RES            = np.array([0.1,0.1,0.1])
    X_DERIV_ORDER         = 2




