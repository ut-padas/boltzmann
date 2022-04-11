"""
@package: Simple structure to maintain simulation parameters
"""
import numpy as np


class BEVelocitySpace():
    VELOCITY_SPACE_POLY_ORDER  = 1
    VELOCITY_SPACE_DIM         = 3
    VELOCITY_SPACE_DT          = 1e-15#0.01/PLASMA_FREQUENCY  
    IO_FILE_NAME_PREFIX        = f'plots/f_sol_%08d.png'
    IO_STEP_FREQ               = 1e5
    SPH_HARM_LM                = [[0,0], [1,0]]
    NUM_Q_VR                   = 51
    NUM_Q_VT                   = 10
    NUM_Q_VP                   = 10
    NUM_Q_CHI                  = 64
    NUM_Q_PHI                  = 10
    NUM_Q_PTS_ON_V             = 20
    NUM_Q_PTS_ON_SPHERE        = 10 #np.max(NUM_Q_CHI,NUM_Q_VT)

    
class BEPositionSpace():
    X_SPACE_DIM           = 3
    X_GRID_MIN            = np.array([-1,-1,-1])
    X_GRID_MAX            = np.array([1,1,1])
    X_GRID_RES            = np.array([0.1,0.1,0.1])
    X_DERIV_ORDER         = 2




def print_parameters():
    print("NUM_POLY_IN_R: ", BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER)
    print("SPH_LM_MODES: ", BEVelocitySpace.SPH_HARM_LM)
    print("NUM_Q_VR: ", BEVelocitySpace.NUM_Q_VR)
    print("NUM_Q_VT: ", BEVelocitySpace.NUM_Q_VT)
    print("NUM_Q_VP: ", BEVelocitySpace.NUM_Q_VP)
    print("NUM_Q_CHI: ", BEVelocitySpace.NUM_Q_CHI)
    print("NUM_Q_PHI: ", BEVelocitySpace.NUM_Q_PHI)
    