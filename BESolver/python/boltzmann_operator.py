"""
@package : Class to handle full boltzmann equation, and to manage different
discretization with operator splitting. 
"""

import numpy as np
import basis
import spec as sp
import math
import collision_operator
import binary_collisions 
import boltzmann_parameters
import visualize_utils
import unigrid
import ets
import fd_derivs
import abc
import enum
import numba as nb


class OpSplittingType(enum.Enum):
    """
    Currently supported operator splitting methods.
    """
    FIRST_ORDER      = 0
    STRANG_SPLITTING = 1 

@nb.njit(parallel=True)
def __x3D_rhs_adv__(u,DERV_ORDER,ADV_VEC,GRID_MIN,GRID_RES):
    
    v = np.zeros(u.shape)

    if DERV_ORDER == 4 :
        dx_f = fd_derivs.deriv42(u,GRID_RES[0],0)
        dy_f = fd_derivs.deriv42(u,GRID_RES[1],1)
        dz_f = fd_derivs.deriv42(u,GRID_RES[2],2)
    elif DERV_ORDER == 2:
        dx_f = fd_derivs.deriv21(u,GRID_RES[0],0)
        dy_f = fd_derivs.deriv21(u,GRID_RES[1],1)
        dz_f = fd_derivs.deriv21(u,GRID_RES[2],2)
    

    for k in nb.prange(u.shape[2]):
        for j in nb.prange(u.shape[1]):
            for i in nb.prange(u.shape[0]):
                v[i,j,k,:] = -ADV_VEC[i,j,k,0] * dx_f[i,j,k,:] - ADV_VEC[i,j,k,1] * dy_f[i,j,k,:] - ADV_VEC[i,j,k,2] * dz_f[i,j,k,:]

    
    if (True):
        f_falloff = 1
        f_asym    = 0
        
        for j in nb.prange(u.shape[1]):
            for k in nb.prange(u.shape[2]):

                xx = GRID_MIN[0] + 0 * GRID_RES[0]
                yy = GRID_MIN[1] + j * GRID_RES[1]
                zz = GRID_MIN[2] + k * GRID_RES[2]
        
                inv_r = 1/np.sqrt(xx**2  + yy**2 + zz**2)
                v[0,j,k,:] = - inv_r * (  xx * ADV_VEC[0, j, k, 0] * dx_f[0,j,k,:]
                                        + yy * ADV_VEC[0, j, k, 1] * dy_f[0,j,k,:]
                                        + zz * ADV_VEC[0, j, k, 2] * dz_f[0,j,k,:]
                                        + f_falloff * (u[0,j,k,:] - f_asym))

                xx = GRID_MIN[0] + (u.shape[0]-1) * GRID_RES[0]
                yy = GRID_MIN[1] + j * GRID_RES[1]
                zz = GRID_MIN[2] + k * GRID_RES[2]
                inv_r = 1/np.sqrt(xx**2  + yy**2 + zz**2)

                v[-1,j,k,:] = - inv_r * ( xx * ADV_VEC[u.shape[0]-1, j, k, 0] * dx_f[-1,j,k,:]
                                        + yy * ADV_VEC[u.shape[0]-1, j, k, 1] * dy_f[-1,j,k,:]
                                        + zz * ADV_VEC[u.shape[0]-1, j, k, 2] * dz_f[-1,j,k,:]
                                        + f_falloff * (u[-1,j,k,:] - f_asym))
                
        
        for i in nb.prange(u.shape[0]):
            for k in nb.prange(u.shape[2]):
                
                xx = GRID_MIN[0] + i * GRID_RES[0]
                yy = GRID_MIN[1] + 0 * GRID_RES[1]
                zz = GRID_MIN[2] + k * GRID_RES[2]
                inv_r = 1/np.sqrt(xx**2  + yy**2 + zz**2)
                v[i,0,k,:] = - inv_r * (  xx * ADV_VEC[i, 0, k, 0] * dx_f[i,0,k,:]
                                        + yy * ADV_VEC[i, 0, k, 1] * dy_f[i,0,k,:]
                                        + zz * ADV_VEC[i, 0, k, 2] * dz_f[i,0,k,:]
                                        + f_falloff * (u[i,0,k,:] - f_asym))

                xx = GRID_MIN[0] + i * GRID_RES[0]
                yy = GRID_MIN[1] + (u.shape[1]-1) * GRID_RES[1]
                zz = GRID_MIN[2] + k * GRID_RES[2]

                inv_r = 1/np.sqrt(xx**2  + yy**2 + zz**2)
                v[i,-1,k,:] = - inv_r * ( xx * ADV_VEC[i, u.shape[1]-1, k, 0] * dx_f[i,-1,k,:]
                                        + yy * ADV_VEC[i, u.shape[1]-1, k, 1] * dy_f[i,-1,k,:]
                                        + zz * ADV_VEC[i, u.shape[1]-1, k, 2] * dz_f[i,-1,k,:]
                                        + f_falloff * (u[i,-1,k,:] - f_asym))

        for i in nb.prange(u.shape[0]):
            for j in nb.prange(u.shape[1]):
                
                xx = GRID_MIN[0] + i * GRID_RES[0]
                yy = GRID_MIN[1] + j * GRID_RES[1]
                zz = GRID_MIN[2] + 0 * GRID_RES[2]
                inv_r = 1/np.sqrt(xx**2  + yy**2 + zz**2)

                v[i,j,0,:] = - inv_r * (  xx * ADV_VEC[i, j, 0, 0] * dx_f[i,j,0,:]
                                        + yy * ADV_VEC[i, j, 0, 1] * dy_f[i,j,0,:]
                                        + zz * ADV_VEC[i, j, 0, 2] * dz_f[i,j,0,:]
                                        + f_falloff * (u[i,j,0,:] - f_asym))

                xx = GRID_MIN[0] + i * GRID_RES[0]
                yy = GRID_MIN[1] + j * GRID_RES[1]
                zz = GRID_MIN[2] + (u.shape[2]-1) * GRID_RES[2]
                inv_r = 1/np.sqrt(xx**2  + yy**2 + zz**2)

                v[i,j,-1,:] = - inv_r * ( xx * ADV_VEC[i, j, u.shape[2]-1, 0] * dx_f[i,j,-1,:]
                                        + yy * ADV_VEC[i, j, u.shape[2]-1, 1] * dy_f[i,j,-1,:]
                                        + zz * ADV_VEC[i, j, u.shape[2]-1, 2] * dz_f[i,j,-1,:]
                                        + f_falloff * (u[i,j,-1,:] - f_asym))
    
    return v


@nb.njit(parallel=True)
def __x3D_v3D_xv_init_vec__(func_v, p, Q_normalized, x_num_pts, x_grid_min, x_grid_res, qx_1d, qw_1d, invWq):
    
    num_q = len(qx_1d)
    num_p = p+1
    fv  = np.zeros((x_num_pts[0],x_num_pts[1],x_num_pts[2],num_p**3))
    fvq = np.zeros((x_num_pts[0],x_num_pts[1],x_num_pts[2],num_q**3))
    
    assert num_q**3 == Q_normalized.shape[0] , "Invalid sizes between Vandomnne Q rows and num. quadrature points."
    assert num_p**3 == Q_normalized.shape[1] , "Invalid sizes between Vandomnne Q cols and v space polynomial order."

    Q_normalized =  np.transpose(Q_normalized)

    for k in nb.prange(fv.shape[2]):
        for j in nb.prange(fv.shape[1]):
            for i in nb.prange(fv.shape[0]):
                x_pt = np.zeros(3)
                v_pt = np.zeros(3)
                x_pt[0] = x_grid_min[0] + i * x_grid_res[0]
                x_pt[1] = x_grid_min[1] + j * x_grid_res[1]
                x_pt[2] = x_grid_min[2] + k * x_grid_res[2]
                for qk,qz in enumerate(qx_1d):
                    for qj,qy in enumerate(qx_1d):
                        for qi,qx in enumerate(qx_1d):
                            v_pt[0] = qx
                            v_pt[1] = qy
                            v_pt[2] = qz
                            v_abs = np.linalg.norm(v_pt,2)
                            r_id  = qk * num_q * num_q + qj * num_q + qi
                            fvq[i,j,k,r_id] = (qw_1d[qk] * qw_1d[qj] * qw_1d[qi] * func_v(x_pt,v_pt) * invWq[r_id] )
                
                
                for c in range(Q_normalized.shape[1]):
                    fv[i,j,k,c]=np.dot(fvq[i,j,k,:], Q_normalized[c,:])
    return fv


@nb.njit(parallel=True)
def __x3D_v3D_zero_velocity_moment__(u,Q,maxwellian,qx_1d,qw_1d,invWq):
    v  = np.zeros((u.shape[0],u.shape[1],u.shape[2],1))
    num_q = len(qw_1d)
    Qu = np.zeros((u.shape[0],u.shape[1],u.shape[2],num_q**3))
    
    assert num_q**3 == Q.shape[0] , "Invalid sizes between Vandomnne Q rows and num. quadrature points. "

    for k in nb.prange(v.shape[2]):
        for j in nb.prange(v.shape[1]):
            for i in nb.prange(v.shape[0]):

                for r in range(Q.shape[0]):
                    Qu[i,j,k,r]=np.dot(Q[r,:],u[i,j,k,:])
                
                q_tmp=0
                for qk,qz in enumerate(qx_1d):
                    for qj,qy in enumerate(qx_1d):
                        for qi,qx in enumerate(qx_1d):
                            q_abs = np.sqrt(qx**2 + qy**2 + qz**2)
                            r_id  = qk * num_q **2 + qj * num_q + qi 
                            q_tmp += (qw_1d[qk] * qw_1d[qj] * qw_1d[qi] * maxwellian(q_abs) * Qu[i,j,k,r_id] *invWq[r_id])
                
                v[i,j,k,0]=q_tmp
    return v
    
@nb.njit(parallel=True)
def __x3D_v3D_first_velocity_moment__(u,Q,maxwellian,qx_1d,qw_1d,invWq,m0):
    v  = np.zeros((u.shape[0],u.shape[1],u.shape[2],3))
    num_q = len(qw_1d)
    Qu = np.zeros((u.shape[0],u.shape[1],u.shape[2],num_q**3))

    assert num_q**3 == Q.shape[0] , "Invalid sizes between Vandomnne Q rows and num. quadrature points. "

    for k in nb.prange(v.shape[2]):
        for j in nb.prange(v.shape[1]):
            for i in nb.prange(v.shape[0]):
                
                for r in range(Q.shape[0]):
                    Qu[i,j,k,r]=np.dot(Q[r,:],u[i,j,k,:])
                
                q_tmp_x=0
                q_tmp_y=0
                q_tmp_z=0
                for qk,qz in enumerate(qx_1d):
                    for qj,qy in enumerate(qx_1d):
                        for qi,qx in enumerate(qx_1d):
                            q_abs = np.sqrt(qx**2 + qy**2 + qz**2)
                            r_id  = qk * num_q **2 + qj * num_q + qi 
                            q_tmp_x += (qx * qw_1d[qk] * qw_1d[qj] * qw_1d[qi] * maxwellian(q_abs) * Qu[i,j,k,r_id] *invWq[r_id])
                            q_tmp_y += (qy * qw_1d[qk] * qw_1d[qj] * qw_1d[qi] * maxwellian(q_abs) * Qu[i,j,k,r_id] *invWq[r_id])
                            q_tmp_z += (qz * qw_1d[qk] * qw_1d[qj] * qw_1d[qi] * maxwellian(q_abs) * Qu[i,j,k,r_id] *invWq[r_id])

                
                v[i,j,k,0]=q_tmp_x/m0[i,j,k,0]
                v[i,j,k,1]=q_tmp_y/m0[i,j,k,0]
                v[i,j,k,2]=q_tmp_z/m0[i,j,k,0]

    return v



    



class BoltzmannOperator_3V_3X():

    def __init__(self):

        self._PARS = boltzmann_parameters.BoltzmannEquationParameters()
        
        # create a basis for the computations for collision operator computations (spectral PG approx.) 
        self._hermite_e        = basis.HermiteE()

        # instance of the collision operator
        self._col_op           = collision_operator.CollisionOpElectronNeutral3D(self._PARS.VEL_SPACE_POLY_ORDER,self._hermite_e)
        self._col_op_spec      = self._col_op.get_spectral_structure()
        self._x_mesh           = unigrid.UCartiesianGrid(self._PARS.X_SPACE_DIM,self._PARS.X_GRID_MIN,self._PARS.X_GRID_MAX,self._PARS.X_GRID_RES)
        self._explicit_ts      = ets.ExplicitODEIntegrator(ets.TSType.RK4)
        self._col_op_spec_vq   = self._col_op_spec.compute_vandermonde_at_quadrature()
        self._col_op_spec_mm_diag = self._col_op_spec.compute_mass_matrix(is_diagonal=True)
        self._current_dt       = None
        self._current_step     = 0
        self._current_time     = 0   
        self._col_op_inv_setup = True

    def current_dt(self):
        return self._current_dt
    
    def current_step(self):
        return self._current_step

    def current_time(self):
        return self._current_time
        
    def create_x_vec(self,dof=1,dtype=float):
        """
        create a position space vector. 
        """
        return self._x_mesh.create_vec(dof,dtype)

    def create_v_vec(self,dof=1,dtype=float):
        """
        create a velocity space vector. 
        """
        num_c = self._col_op_spec.get_num_coefficients()
        return np.zeros((num_c,dof),dtype=dtype)
    
    def init_xv_vec(self,func):
        """
        set the initial conditions for the evolution. 
        func : x , v -- > R+
        """
        num_x_pts  = np.array(self._x_mesh._num_pts)
        #print(num_x_pts)
        x_grid_min = self._x_mesh._min_pt
        x_grid_res = self._x_mesh._res
        mm_diag    = self._col_op_spec_mm_diag 
        vp_order   = self._col_op_spec._p 
        Q_normalized = np.array(self._col_op_spec_vq)
        num_q      = vp_order + 1

        assert Q_normalized.shape is not (num_q**3,num_q**3) , "Vq and the polynomial order does not match"

        # normalize Vq with the correct diagonal factor. 
        for i in range(Q_normalized.shape[0]):
            for j in range(Q_normalized.shape[1]):
                Q_normalized[i,j]/= mm_diag[j]
        
        gauss_q    = self._col_op_spec._basis_p.Gauss_Pn(num_q)
        qx_1d      = gauss_q[0]
        qw_1d      = gauss_q[1]
        weight_f   = self._col_op_spec._basis_p.Wx()

        inv_weight_f = np.zeros(num_q**3)
        # compute the poly weights. 
        for qk,qz in enumerate(qx_1d):
            for qj,qy in enumerate(qx_1d):
                for qi,qx in enumerate(qx_1d):
                    v_abs = np.sqrt(qx**2 + qy**2 + qz**2)
                    r_id  = qk * num_q * num_q + qj * num_q + qi
                    inv_weight_f[r_id] = 1.0/weight_f(v_abs)
                    


        fv = __x3D_v3D_xv_init_vec__(func,vp_order,Q_normalized,num_x_pts,x_grid_min,x_grid_res,gauss_q[0],gauss_q[1],inv_weight_f)
        return fv

    def create_xv_vec(self):
        fv = self.create_x_vec(dof=self._col_op_spec.get_num_coefficients())
        return fv
    
    def zeroth_velocity_moment(self,fv,maxwellian):
        """
        computes the zeroth order velocity moment. 
        """
        num_p = self._col_op_spec._p + 1
        [qx_1d,qw_1d] = self._col_op_spec._basis_p.Gauss_Pn(num_p)
        weight_f   = self._col_op_spec._basis_p.Wx()
        
        num_q = len(qx_1d)
        inv_weight_f = np.zeros(num_q**3)
        # compute the poly weights. 
        for qk,qz in enumerate(qx_1d):
            for qj,qy in enumerate(qx_1d):
                for qi,qx in enumerate(qx_1d):
                    v_abs = np.sqrt(qx**2 + qy**2 + qz**2)
                    r_id  = qk * num_q * num_q + qj * num_q + qi
                    inv_weight_f[r_id] = 1.0/weight_f(v_abs)

        Q = self._col_op_spec_vq
        v = __x3D_v3D_zero_velocity_moment__(fv,Q,maxwellian,qx_1d,qw_1d,inv_weight_f)
        return v

    def first_velocity_moment(self,fv,maxwellian,m0):
        
        """
        computes first velocity moment (avg velocity (x,t)) 
        """
        num_p = self._col_op_spec._p + 1
        [qx_1d,qw_1d] = self._col_op_spec._basis_p.Gauss_Pn(num_p)
        weight_f   = self._col_op_spec._basis_p.Wx()
        
        num_q = len(qx_1d)
        inv_weight_f = np.zeros(num_q**3)
        # compute the poly weights. 
        for qk,qz in enumerate(qx_1d):
            for qj,qy in enumerate(qx_1d):
                for qi,qx in enumerate(qx_1d):
                    v_abs = np.sqrt(qx**2 + qy**2 + qz**2)
                    r_id  = qk * num_q * num_q + qj * num_q + qi
                    inv_weight_f[r_id] = 1.0/weight_f(v_abs)

        Q = self._col_op_spec_vq
        v = __x3D_v3D_first_velocity_moment__(fv,Q,maxwellian,qx_1d,qw_1d,inv_weight_f,m0)
        return v

    def adv_xspace_rhs(self,u,adv_v,bdy_type):
        GRID_MAX = self._x_mesh._max_pt
        GRID_MIN = self._x_mesh._min_pt
        GRID_RES = self._x_mesh._res

        ADV_VEC  = adv_v
        v  = __x3D_rhs_adv__(u,self._PARS.X_DERIV_ORDER,ADV_VEC,GRID_MIN,GRID_RES)
        return v
        
    def assemble_col_op(self):
        self._col_op_Lij       = self._col_op.assemble_collision_mat(binary_collisions.ElectronNeutralCollisionElastic_X0D_V3D(),maxwellian=self._hermite_e.Wx())
    
    def evolve(self, u, maxwellian ,splitting_method=None):
        
        v = np.zeros(u.shape,dtype=u.dtype)
        self.assemble_col_op()

        if splitting_method == OpSplittingType.FIRST_ORDER:
            
            # set up the advection part, 
            GRID_RES = self._PARS.X_GRID_RES
            m0 = self.zeroth_velocity_moment(u,maxwellian)
            adv_v = self.first_velocity_moment(u,maxwellian,m0)
            print(adv_v.shape)

            rhs_func = lambda uh,th : self.adv_xspace_rhs(uh,adv_v,True)

            min_vx = np.min(adv_v[:,:,:,0])
            min_vy = np.min(adv_v[:,:,:,1])
            min_vz = np.min(adv_v[:,:,:,2])

            min_v  = min(min_vx,min(min_vy,min_vz))
            cfl    = 0.8 * min_v

            dt     = max(1e-5,cfl * np.min(GRID_RES))
            if( dt <= 1e-5):
                dt = self._PARS.TIME_STEP_SIZE
                print("min (mean velocity) on the grid :  ", min_v)
            
            if self._current_dt == None:
                self._current_dt = dt
                self._col_op_inv_setup = True
            
            if(self._current_dt is not dt):
                self._current_dt = dt
                self._col_op_inv_setup = True

            self._explicit_ts.set_ts_size(self._current_dt)
            self._explicit_ts.set_rhs_func(rhs_func)

            up = self._explicit_ts.evolve(u)

            if self._col_op_inv_setup :
                # implicit operator for the collision term
                [rows , cols] = self._col_op_Lij.shape
                self._col_op_invLij = self._current_dt * self._col_op_Lij

                for r in range(rows):
                    self._col_op_invLij[r,r] = self._col_op_invLij[r,r]/self._col_op_spec_mm_diag[r]

                self._col_op_invLij = np.eye(rows) - self._col_op_invLij
                self._col_op_invLij = np.linalg.inv(self._col_op_invLij)

                self._col_op_inv_setup = False

            for k in range(u.shape[2]):
                for j in range(u.shape[1]):
                    for i in range(u.shape[0]):
                        v[i,j,k,:] = np.matmul(self._col_op_invLij,up[i,j,k,:])

            self._current_time += 2*self._current_dt
            self._current_step +=1
            return v





        











        

