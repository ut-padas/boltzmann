"""
@package: Simple structure to maintain simulation parameters
"""
import scipy.constants
import numpy as np
import basis
import spec as sp
import pint


class BEVelocitySpace():

    VELOCITY_SPACE_POLY_ORDER  = 2
    VELOCITY_SPACE_DIM         = 3
    VELOCITY_SPACE_DT          = 1e-3  
    IO_FILE_NAME_PREFIX        = f'plots/f_sol_%04d.png'
    IO_STEP_FREQ               = 500
    #SPH_HARM_LM                = [[0,0], [1,-1], [1,0], [1,1], [2,-2],[2,-1],[2,0],[2,1],[2,2]] # , [3,0], [4,0]]
    SPH_HARM_LM                = [[0,0], [1,0]]
    NUM_Q_PTS_ON_V             = 5
    NUM_Q_PTS_ON_SPHERE        = 5

    


    



class BEPositionSpace():
    X_SPACE_DIM           = 3
    X_GRID_MIN            = np.array([-1,-1,-1])
    X_GRID_MAX            = np.array([1,1,1])
    X_GRID_RES            = np.array([0.1,0.1,0.1])
    X_DERIV_ORDER         = 2




