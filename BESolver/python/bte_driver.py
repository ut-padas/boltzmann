"""
@package Basic boltzmann equation driver without external force field. 
"""
import numpy as np
import boltzmann_operator as BoltzOp
import boltzmann_parameters
import matplotlib.pyplot as plt
import numba as nb
from time import perf_counter as time


@nb.njit
def bte_maxwellian(x_abs):
    return np.exp(-0.5*(x_abs**2))

@nb.njit
def f_init(x,v):
    #return ( 1.0/(1.0 * np.sqrt(2 * np.pi)) ) * bte_maxwellian(np.linalg.norm(v,2))
    return bte_maxwellian(np.linalg.norm(v,2)) # * bte_maxwellian(np.linalg.norm(x,2))
    #return np.linalg.norm(v,2)**2 * bte_maxwellian(np.linalg.norm(v-1,2))/ (1+ np.linalg.norm(x,2))
   


bOp   = BoltzOp.BoltzmannOperator_3V_3X()
fv    = bOp.init_xv_vec(f_init)
PARS  = bOp._PARS

import os
if not os.path.exists('bplots'):
    print("creating folder `plots`, output will be written into it")
    os.makedirs('bplots')

#print(fv[0,0,0,:])
#print(fv[0,0,1,:])

TOTAL_TIME          = PARS.TOTAL_TIME
IO_STEP_FREQ        = PARS.IO_STEP_FREQ
Z_SLICE_VAL         = 0.0
IO_FILE_NAME_PREFIX = PARS.IO_FILE_NAME_PREFIX
GRID_MIN            = PARS.X_GRID_MIN
GRID_RES            = PARS.X_GRID_RES

z_slice_index = int ((Z_SLICE_VAL - GRID_MIN[2])/GRID_RES[2])
while bOp.current_time() < TOTAL_TIME:
    if bOp.current_step() % IO_STEP_FREQ == 0:
        fv_m0 = bOp.zeroth_velocity_moment(fv,maxwellian=bte_maxwellian)
        print("time stepper current time ",bOp.current_time())
        plt.title("T=%.2f"%bOp.current_time())
        plt.imshow(fv_m0[:,:,z_slice_index,0])
        plt.colorbar()
        fname = IO_FILE_NAME_PREFIX %bOp.current_step()
        plt.savefig(fname)
        plt.close()
    
    fv = bOp.evolve(fv,maxwellian=bte_maxwellian,splitting_method=BoltzOp.OpSplittingType.FIRST_ORDER)
    



#v = bOp.evolve(fv,maxwellian=boltzmann_parameters.maxwellian_normalized,splitting_method=BoltzOp.OpSplittingType.FIRST_ORDER)
#print(v)
#print(fv[0,0,0,:])
#print(fv[0,0,1,:])



# fv_m0 = bOp.zeroth_velocity_moment(fv,maxwellian=bte_maxwellian)
# fv_m1 = bOp.first_velocity_moment(fv,maxwellian=bte_maxwellian,m0=fv_m0)

# Z_SLICE_VAL = 0
# GRID_MIN = bOp._PARS.X_GRID_MIN
# GRID_RES = bOp._PARS.X_GRID_RES
# z_slice_index = int ((Z_SLICE_VAL - GRID_MIN[2])/GRID_RES[2])

# plt.imshow(fv[:,:,z_slice_index,0])
# plt.colorbar()
# plt.savefig("fv_z=%.2f_.png"%Z_SLICE_VAL)
# plt.close()


# plt.imshow(fv_m0[:,:,z_slice_index,0])
# plt.colorbar()
# plt.savefig("fv_m0_z=%.2f_.png"%Z_SLICE_VAL)
# plt.close()

# plt.imshow(fv_m1[:,:,z_slice_index,0])
# plt.colorbar()
# plt.savefig("fv_m1_vx_z=%.2f_.png"%Z_SLICE_VAL)
# plt.close()

# plt.imshow(fv_m1[:,:,z_slice_index,1])
# plt.colorbar()
# plt.savefig("fv_m1_vy_z=%.2f_.png"%Z_SLICE_VAL)
# plt.close()

# plt.imshow(fv_m1[:,:,z_slice_index,2])
# plt.colorbar()
# plt.savefig("fv_m1_vz_z=%.2f_.png"%Z_SLICE_VAL)
# plt.close()