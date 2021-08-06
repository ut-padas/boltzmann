"""
@package simple linear advection on uniform grid, MOL, FD (traditional)
"""
import numpy as np
import unigrid 
import fd_derivs
import ets
import matplotlib.pyplot as plt
import numba as nb
import numpy as np

# set the threading layer before any parallel target compilation
nb.config.THREADING_LAYER = 'threadsafe'

GRID_DIM            = 3
GRID_MIN            = np.array([-0.5,-0.5,-0.5])
GRID_MAX            = np.array([ 0.5, 0.5, 0.5])
GRID_RES            = np.array([ 0.01, 0.01, 0.01])
TOTAL_TIME          = 10.0 
IO_STEP_FREQ        = 10
Z_SLICE_VAL         = 0.0
IO_FILE_NAME_PREFIX = f'plots/u_sol_%04d.png'

ADV_VEC  = np.array([1.0,1.0,0.0])
#ADV_VEC  = np.zeros(3)
CFL_FAC  = 0.4

#F_IC = lambda x : np.sin(2*np.pi*x[0]) * np.sin(2*np.pi*x[1]) * np.sin(2*np.pi*x[2])
F_IC = lambda x :  np.exp (-10.0 * (x[0]**2 + x[1]**2 + x[2]**2) )
UGRID = unigrid.UCartiesianGrid(GRID_DIM,GRID_MIN,GRID_MAX,GRID_RES)
u = unigrid.init_vec(UGRID,F_IC)

@nb.njit(parallel=True)
def f_rhs(u,t):
    v = np.zeros(u.shape)

    dx_f = fd_derivs.deriv42(u,GRID_RES[0],0)
    dy_f = fd_derivs.deriv42(u,GRID_RES[1],1)
    dz_f = fd_derivs.deriv42(u,GRID_RES[2],2)

    v = -ADV_VEC[0] * dx_f - ADV_VEC[1] * dy_f - ADV_VEC[2] * dz_f

    # Robin boundary conditions. 
    # f(r) = f0 + k / r^n 
    # the solution decays radially, bodunary conditions dervied based on that. 
    if True: 
        f_falloff = 1
        f_asym    = 0

        for j in nb.prange(u.shape[1]):
            for k in nb.prange(u.shape[2]):
                
                xx = GRID_MIN[0] + 0 * GRID_RES[0]
                yy = GRID_MIN[1] + j * GRID_RES[1]
                zz = GRID_MIN[2] + k * GRID_RES[2]
                
                inv_r = 1/np.sqrt(xx**2  + yy**2 + zz**2)
                v[0,j,k,:] = - inv_r * (  xx * ADV_VEC[0] * dx_f[0,j,k,:]
                                        + yy * ADV_VEC[1] * dy_f[0,j,k,:]
                                        + zz * ADV_VEC[2] * dz_f[0,j,k,:]
                                        + f_falloff * (u[0,j,k,:] - f_asym))

                xx = GRID_MIN[0] + (u.shape[0]-1) * GRID_RES[0]
                yy = GRID_MIN[1] + j * GRID_RES[1]
                zz = GRID_MIN[2] + k * GRID_RES[2]

                inv_r = 1/np.sqrt(xx**2  + yy**2 + zz**2)
                v[-1,j,k,:] = - inv_r * ( xx * ADV_VEC[0] * dx_f[-1,j,k,:]
                                        + yy * ADV_VEC[1] * dy_f[-1,j,k,:]
                                        + zz * ADV_VEC[2] * dz_f[-1,j,k,:]
                                        + f_falloff * (u[-1,j,k,:] - f_asym))
                
        
        for i in nb.prange(u.shape[0]):
            for k in nb.prange(u.shape[2]):
                
                xx = GRID_MIN[0] + i * GRID_RES[0]
                yy = GRID_MIN[1] + 0 * GRID_RES[1]
                zz = GRID_MIN[2] + k * GRID_RES[2]

                inv_r = 1/np.sqrt(xx**2  + yy**2 + zz**2)
                v[i,0,k,:] = - inv_r * (  xx * ADV_VEC[0] * dx_f[i,0,k,:]
                                        + yy * ADV_VEC[1] * dy_f[i,0,k,:]
                                        + zz * ADV_VEC[2] * dz_f[i,0,k,:]
                                        + f_falloff * (u[i,0,k,:] - f_asym))

                xx = GRID_MIN[0] + i * GRID_RES[0]
                yy = GRID_MIN[1] + (u.shape[1]-1) * GRID_RES[1]
                zz = GRID_MIN[2] + k * GRID_RES[2]

                inv_r = 1/np.sqrt(xx**2  + yy**2 + zz**2)
                v[i,-1,k,:] = - inv_r * ( xx * ADV_VEC[0] * dx_f[i,-1,k,:]
                                        + yy * ADV_VEC[1] * dy_f[i,-1,k,:]
                                        + zz * ADV_VEC[2] * dz_f[i,-1,k,:]
                                        + f_falloff * (u[i,-1,k,:] - f_asym))

        for i in nb.prange(u.shape[0]):
            for j in nb.prange(u.shape[1]):
                
                xx = GRID_MIN[0] + i * GRID_RES[0]
                yy = GRID_MIN[1] + j * GRID_RES[1]
                zz = GRID_MIN[2] + 0 * GRID_RES[2]

                inv_r = 1/np.sqrt(xx**2  + yy**2 + zz**2)
                v[i,j,0,:] = - inv_r * (  xx * ADV_VEC[0] * dx_f[i,j,0,:]
                                        + yy * ADV_VEC[1] * dy_f[i,j,0,:]
                                        + zz * ADV_VEC[2] * dz_f[i,j,0,:]
                                        + f_falloff * (u[i,j,0,:] - f_asym))

                xx = GRID_MIN[0] + i * GRID_RES[0]
                yy = GRID_MIN[1] + j * GRID_RES[1]
                zz = GRID_MIN[2] + (u.shape[2]-1) * GRID_RES[2]

                inv_r = 1/np.sqrt(xx**2  + yy**2 + zz**2)
                v[i,j,-1,:] = - inv_r * ( xx * ADV_VEC[0] * dx_f[i,j,-1,:]
                                        + yy * ADV_VEC[1] * dy_f[i,j,-1,:]
                                        + zz * ADV_VEC[2] * dz_f[i,j,-1,:]
                                        + f_falloff * (u[i,j,-1,:] - f_asym))


    return v

ts_integrator = ets.ExplicitODEIntegrator(ets.TSType.RK4)
ts_integrator.set_ts_size(0.3 * np.min(GRID_RES))
ts_integrator.set_rhs_func(f_rhs)

ts_integrator.init()
z_slice_index = int ((Z_SLICE_VAL - GRID_MIN[2])/GRID_RES[2])

import os
if not os.path.exists('plots'):
    print("creating folder `plots`, output will be written into it")
    os.makedirs('plots')

NUMBA_PARALLEL_DIAGNOSTIC = True
while ts_integrator.current_ts()[0] < TOTAL_TIME:
    ts_info = ts_integrator.current_ts()
    if ts_info[1] % IO_STEP_FREQ == 0:
        print("time stepper current time ",ts_info[0])
        plt.title("T=%.2f"%ts_info[0])
        plt.imshow(u[:,:,z_slice_index,0])
        plt.colorbar()
        fname = IO_FILE_NAME_PREFIX %ts_info[1]
        #print(fname)
        plt.savefig(fname)
        plt.close()
        
    
    #v= np.zeros(u.shape)
    u = ts_integrator.evolve(u)
    if(NUMBA_PARALLEL_DIAGNOSTIC):
        print("Threading layer chosen: %s" % nb.threading_layer())
        f_rhs.parallel_diagnostics(level=4)
        NUMBA_PARALLEL_DIAGNOSTIC=False
    




  


