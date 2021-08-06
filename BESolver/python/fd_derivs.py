"""
@package traditional finite difference stencil application on numpy array. 
"""
import numpy as np
import numba as nb

# set the threading layer before any parallel target compilation
nb.config.THREADING_LAYER = 'threadsafe'

@nb.njit(parallel=True)
def deriv42(u,dx,axis):
    """
    u is shape (dim 0, dim 1, dim 2, dof) shape. 
    compute the first deriv using 4th order stencil interior 2nd order at the boundary for all dofs
    on the specified axis, axis i stencil applied to dim i, 
    """
    idx = 1.0/dx
    idx_by_2 = 0.5 * idx
    idx_by_12 = idx / 12.0

    Du = np.zeros(u.shape)

    if(axis==0):
        Du[2:-2,:,:,:] = (u[0:-4,:,:,:] -8.0 * u[1:-3,:,:,:] + 8.0 * u[3:-1,:,:,:] - u[4:,:,:,:])*idx_by_12
        
        #min bdy
        Du[0,:,:,:] = (-3.0 * u[0,:,:,:] + 4.0 * u[1,:,:,:] - u[2,:,:,:] ) * idx_by_2
        Du[1,:,:,:] = (-u[0,:,:,:] +  u[2,:,:,:]) * idx_by_2
        #max bdy
        Du[-2,:,:,:] = (-u[-3,:,:,:] +  u[-1,:,:,:]) * idx_by_2
        Du[-1,:,:,:] = (3.0 * u[-1,:,:,:] - 4.0 * u[-2,:,:,:] + u[-3,:,:,:] ) * idx_by_2
        
    elif (axis==1):
        Du[:,2:-2,:,:] = (u[:,0:-4,:,:] -8.0 * u[:,1:-3,:,:] + 8.0 * u[:,3:-1,:,:] - u[:,4:,:,:])*idx_by_12
        
        #min bdy
        Du[:,0,:,:] = (-3.0 * u[:,0,:,:] + 4.0 * u[:,1,:,:] - u[:,2,:,:] ) * idx_by_2
        Du[:,1,:,:] = (-u[:,0,:,:] +  u[:,2,:,:]) * idx_by_2
        #max bdy
        Du[:,-2,:,:] = (-u[-3,:,:,:] +  u[-1,:,:,:]) * idx_by_2
        Du[:,-1,:,:] = (3.0 * u[-1,:,:,:] - 4.0 * u[-2,:,:,:] + u[-3,:,:,:] ) * idx_by_2

    elif (axis==2):
        Du[:,:,2:-2,:] = (u[:,:,0:-4,:] -8.0 * u[:,:,1:-3,:] + 8.0 * u[:,:,3:-1,:] - u[:,:,4:,:])*idx_by_12

        # min bdy
        Du[:,:,0,:] = (-3.0 * u[:,:,0,:] + 4.0 * u[:,:,1,:] - u[:,:,2,:] ) * idx_by_2
        Du[:,:,1,:] = (-u[0,:,:,:] +  u[2,:,:,:]) * idx_by_2
        # max bdy
        Du[:,:,-2,:] = (-u[:,:,-3,:] +  u[:,:,-1,:]) * idx_by_2
        Du[:,:,-1,:] = (3.0 * u[:,:,-1,:] - 4.0 * u[:,:,-2,:] + u[:,:,-3,:]) * idx_by_2
    
    return Du

@nb.njit(parallel=True)
def deriv21(u,dx,axis):
    """
    u is shape (dim 0, dim 1, dim 2, dof) shape. 
    compute the first deriv using 4th order stencil interior 2nd order at the boundary for all dofs
    on the specified axis, axis i stencil applied to dim i, 
    """
    idx = 1.0/dx
    idx_by_2 = 0.5 * idx
    Du = np.zeros(u.shape)

    if(axis==0):
        Du[1:-1,:,:,:] = (u[2:,:,:,:] -u[0:-2,:,:,:]) * idx_by_2
        
        # min bdy
        Du[0,:,:,:] = (-u[0,:,:,:] + u[1,:,:,:]) * idx
        # max bdy
        Du[-1,:,:,:] = (u[-1,:,:,:] - u[-2,:,:,:]) * idx
    elif (axis==1):
        Du[:,1:-1,:,:] = (u[:,2:,:,:] -u[:,0:-2,:,:]) * idx_by_2
        
        # min bdy
        Du[:,0,:,:] = (-u[:,0,:,:] + u[:,1,:,:]) *idx
        # max bdy
        Du[:,-1,:,:] = (u[:,-1,:,:] - u[:,-2,:,:]) * idx

    elif (axis==2):
        Du[:,:,1:-1,:] = (u[:,:,2:,:] -u[:,:,0:-2,:]) * idx_by_2

        # min bdy
        Du[:,:,0,:] = (-u[:,:,0,:] + u[:,:,1,:]) * idx
        # max bdy
        Du[:,:,-1,:] = (u[:,:,-1,:] - u[:,:,-2,:]) * idx

    return Du