"""
@brief : Boltzmann solver in 1d-space glow discharge problem
"""
from ast import arg
from cProfile import label
import sys
import numpy as np
import numba as nb
import spec_spherical as sp
import collision_operator_spherical as colOpSp
import collisions 
import parameters as params
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.constants
import scipy.integrate
import utils as BEUtils
import argparse
import enum
import basis
import dill as pickle
import scipy.interpolate
#import pickle
import time
# set the threading layer before any parallel target compilation
nb.config.THREADING_LAYER = 'threadsafe'
import cupy as cp
import cupyx.scipy.sparse as cp_sparse
import cupyx.profiler
import matplotlib as mpl
import pathlib
plt.rcParams.update({
    "text.usetex": False,
    "font.size": 14,
    #"font.family": "Helvetica",
    #"lines.linewidth":2.0
})

class CollissionMode(enum.Enum):
    ELASTIC_ONLY=0
    ELASTIC_W_EXCITATION=1
    ELASTIC_W_IONIZATION=2
    ELASTIC_W_EXCITATION_W_IONIZATION=3

collisions.MAXWELLIAN_N=1e0
collisions.AR_NEUTRAL_N=3.22e22
collisions.AR_IONIZED_N=collisions.AR_NEUTRAL_N 
parser = argparse.ArgumentParser()

parser.add_argument("-Nr", "--NUM_P_RADIAL"                   , help="Number of polynomials in radial direction", type=int, default=64)
parser.add_argument("-T", "--T_END"                           , help="Simulation time", type=float, default=1e-4)
parser.add_argument("-dt", "--T_DT"                           , help="Simulation time step size ", type=float, default=1e-11)
parser.add_argument("-ts_atol", "--ts_atol"                   , help="absolute ts tol", type=float, default=1e-15)
parser.add_argument("-ts_rtol", "--ts_rtol"                   , help="relative ts tol", type=float, default=1e-6)
parser.add_argument("-l_max",   "--l_max"                     , help="max polar modes in SH expansion", type=int, default=1)

parser.add_argument("-c", "--collisions"     , help="collisions included (g0, g0Const, g0NoLoss, g2, g2Const)",nargs='+', type=str, default=["g0","g2"])
parser.add_argument("-ev", "--electron_volt" , help="initial electron volt", type=float, default=2.0)
parser.add_argument("-bscale", "--basis_scale"                , help="basis electron volt", type=float, default=1.0)
parser.add_argument("-E", "--E_field"                         , help="Electric field in V/m", type=float, default=100)

parser.add_argument("-q_vr", "--quad_radial"                  , help="quadrature in r"        , type=int, default=200)
parser.add_argument("-q_vt", "--quad_theta"                   , help="quadrature in polar"    , type=int, default=4)
parser.add_argument("-q_vp", "--quad_phi"                     , help="quadrature in azimuthal", type=int, default=4)
parser.add_argument("-q_st", "--quad_s_theta"                 , help="quadrature in scattering polar"    , type=int, default=4)
parser.add_argument("-q_sp", "--quad_s_phi"                   , help="quadrature in scattering azimuthal", type=int, default=4)
parser.add_argument("-radial_poly", "--radial_poly"           , help="radial basis", type=str, default="bspline")
parser.add_argument("-sp_order", "--spline_order"             , help="b-spline order", type=int, default=2)
parser.add_argument("-spline_qpts", "--spline_q_pts_per_knot" , help="q points per knots", type=int, default=4)

parser.add_argument("-Lz", "--Lz"               , help="1d length of the torch in (m)", type=float, default=0.0254)
parser.add_argument("-num_z", "--num_z"         , help="number of points in z-direction", type=int, default=50)
parser.add_argument("-v0_z0", "--v0_z0"         , help="voltage at z=0 (v)", type=float, default=100.0)
parser.add_argument("-rf_freq", "--rf_freq"     , help="voltage oscillation freq (Hz)", type=float, default=13.56e6)
parser.add_argument("-ne", "--ne"               , help="initial electron density 1/m^3", type=float, default=1e0)
parser.add_argument("-n0", "--n0"               , help="initial neutral density 1/m^3", type=float, default=3.22e22)
parser.add_argument("-ne_fac", "--ne_fac"       , help="normalization factor for ne, ni 1/m^3", type=float, default=8e16)
parser.add_argument("-rs", "--restore"          , help="restore solver", type=int, default=0)
parser.add_argument("-gpu", "--gpu"             , help="enable gpu", type=int, default=0)
parser.add_argument("-device_id", "--device_id" , help="GPU device id to use", type=int, default=0)
parser.add_argument("-v_cfl", "--v_cfl"         , help="vspace cfl", type=float, default=0.2)
parser.add_argument("-vx_max", "--vx_max"        , help="normalized velocity coordinate max cuttoff (B-Splines)", type=float, default=8)
parser.add_argument("-benchmark", "--benchmark" , help="benchmark run", type=int, default=0)
parser.add_argument("-eta_z", "--eta_z"         , help="diffusion in z-space", type=float, default=0.0)

@nb.njit(parallel=True)
def grad1_p_11(u, dx):
    """
    1st order fd with upwinding 
    for modes traveling left to right
    """
    idx = 1.0 / dx
    Du=np.zeros_like(u)
    
    #Du[i] = (u[i]-u[i-1])/dx
    Du[1:] = (u[1:] - u[0:-1]) * idx

    #shifted only for the bdy
    Du[0]  = (u[1] - u[0]) * idx
    return Du

@nb.njit(parallel=True)
def grad1_m_11(u, dx):
    """
    1st order fd with upwinding 
    for modes traveling right to left
    """
    idx = 1.0 / dx
    Du=np.zeros_like(u)
    
    #Du[i] = (u[i+1]-u[i])/dx
    Du[:-1] = (u[1:] - u[0:-1]) * idx
    
    #shifted only for the bdy
    Du[-1]  = (u[-1] - u[-2]) * idx
    return Du

def grad1_p_11_mat(grid_z):
    """
    csr-format of the 1st order FD upwind direction left to right
    """
    n    = len(grid_z)
    dz   = grid_z[1]-grid_z[0]
    idz  = 1.0 /dz
    
    rows = np.array([],dtype=int)
    cols = np.array([],dtype=int)
    data = np.array([],dtype=float)

    rows = np.append(rows, np.array(range(1,n)))
    cols = np.append(cols, np.array(range(1,n)))
    data = np.append(data, np.ones(n-1) * idz)

    rows = np.append(rows, np.array(range(1,n)))
    cols = np.append(cols, np.array(range(0,n-1)))
    data = np.append(data, -np.ones(n-1) * idz)

    #bc-left
    rows = np.append(rows, np.array([0,0]))
    cols = np.append(cols, np.array([0,1]))
    data = np.append(data, np.array([-1,1]) * idz)

    return scipy.sparse.csr_matrix((data, (rows, cols)), shape=(n,n))

def grad1_11_mat(grid_z):
    """
    csr-format of the 1st order FD central scheme
    """
    n    = len(grid_z)
    dz   = grid_z[1]-grid_z[0]
    idz  = 1.0/dz
    
    rows = np.array([],dtype=int)
    cols = np.array([],dtype=int)
    data = np.array([],dtype=float)

    rows = np.append(rows, np.array(range(1,n-1)))
    cols = np.append(cols, np.array(range(2,n)))
    data = np.append(data, np.ones(n-2) * idz * 0.5)

    rows = np.append(rows, np.array(range(1,n-1)))
    cols = np.append(cols, np.array(range(0,n-2)))
    data = np.append(data, -np.ones(n-2) * idz * 0.5)

    #bc-left
    rows = np.append(rows, np.array([0,0]))
    cols = np.append(cols, np.array([0,1]))
    data = np.append(data, np.array([-1,1]) * idz)

    #bc-right
    rows = np.append(rows, np.array([n-1,n-1]))
    cols = np.append(cols, np.array([n-2,n-1]))
    data = np.append(data, np.array([-1,1]) * idz)

    return scipy.sparse.csr_matrix((data, (rows, cols)), shape=(n,n))

def grad1_m_11_mat(grid_z):
    """
    csr-format of the 1st order FD with upwind direction right to left. 
    """
    n    = len(grid_z)
    dz   = grid_z[1]-grid_z[0]
    idz  = 1.0 /dz
    
    rows = np.array([],dtype=int)
    cols = np.array([],dtype=int)
    data = np.array([],dtype=float)

    rows = np.append(rows, np.array(range(0,n-1)))
    cols = np.append(cols, np.array(range(0,n-1)))
    data = np.append(data, -np.ones(n-1) * idz)

    rows = np.append(rows, np.array(range(0,n-1)))
    cols = np.append(cols, np.array(range(1,n)))
    data = np.append(data, np.ones(n-1) * idz)

    #bc-right
    rows = np.append(rows, np.array([n-1,n-1]))
    cols = np.append(cols, np.array([n-2,n-1]))
    data = np.append(data, np.array([-1,1]) * idz)

    return scipy.sparse.csr_matrix((data, (rows, cols)), shape=(n,n))

def laplace_op_fd_11(grid_z):
    """
    laplace operator with FD approximation 
    """
    n   = len(grid_z)
    dx  = grid_z[1] - grid_z[0]
    idx2 = 1/dx**2

    # i,i-1    
    rows = np.array(range(1,n-1))
    cols = np.array(range(0,n-2))
    data = np.ones(n-2) * idx2

    #i, i
    rows = np.append(rows, np.array(range(1,n-1)))
    cols = np.append(cols, np.array(range(1,n-1)))
    data = np.append(data, np.ones(n-2) * idx2 * (-2))

    # i, i+1
    rows = np.append(rows, np.array(range(1,n-1)))
    cols = np.append(cols, np.array(range(2,n)))
    data = np.append(data, np.ones(n-2) * idx2)

    # # left boundary 
    rows = np.append(rows, np.array([0]))
    cols = np.append(cols, np.array([0]))
    data = np.append(data, np.array([1.0]))

    # # right boundary. 
    rows = np.append(rows, np.array([n-1]))
    cols = np.append(cols, np.array([n-1]))
    data = np.append(data, np.array([1.0]))


    Lmat = scipy.sparse.csr_array((data, (rows, cols)), shape=(n,n))
    #print(Lmat.toarray() / idx2)
    return Lmat

def lm_modes_to_vt_pts_op(grid_z, spec_sp: sp.SpectralExpansionSpherical , vt_pts):
    """
    Galerkin to collocation projection. P_gc
    hf_c =np.dot(hf.reshape((len(grid_v), num_p, num_sh)), P_gc)
    """
    num_p  = spec_sp._p + 1
    num_sh = len(spec_sp._sph_harm_lm)

    Y_l_vt = spec_sp.Vq_sph(vt_pts, np.zeros_like(vt_pts))
    return Y_l_vt

def vt_pts_to_lm_modes_op(grid_z, spec_sp: sp.SpectralExpansionSpherical , vt_pts):
    """
    collocation to sph basis. P_cg
    hf     = np.dot(hf_c,P_cg).reshape((len(grid_z), num_p*num_sh))
    """
    num_p  = spec_sp._p + 1
    num_sh = len(spec_sp._sph_harm_lm)

    glx, glw     = basis.Legendre().Gauss_Pn(len(vt_pts))
    vq_theta     = np.arccos(glx)
    assert (vq_theta == vt_pts).all(), "collocation points does not match with the theta quadrature points"
    
    Y_vt_l = np.transpose(np.matmul( spec_sp.Vq_sph(vt_pts, np.zeros_like(vt_pts)), np.diag(glw) ) ) * 2 * np.pi
    return Y_vt_l

def create_vec_lm(grid_z, spec_sp : sp.SpectralExpansionSpherical):
    """
    create a vector in a phase space (space, velocity space)
    """
    num_p  = spec_sp._p+1
    num_sh = len(spec_sp._sph_harm_lm)
    return np.zeros((len(grid_z), num_p*num_sh))

def create_vec_vt(grid_z, spec_sp : sp.SpectralExpansionSpherical, grid_vt):
    """
    create a vector in a phase space (space, velocity space)
    """
    num_p  = spec_sp._p+1
    return np.zeros((len(grid_z), num_p * len(grid_vt)))

class EField_1D():
    """
    1d solver for the electric field, poisson solver
    """
    def __init__(self, grid_z) -> None:
        self._grid_z = grid_z
        self._epsilon_0  = 55.26349406e6 # e^2 ev^-1 m^-1
        return

    def fd1_laplace_1d_mat(self):
        """
        Assembles the laplace operator with
        1st order finite difference approximation. 
        """
        grid_z = self._grid_z
        return laplace_op_fd_11(grid_z)

    def e_field_op(self):
        grid_z = self._grid_z
        n   = len(grid_z)
        dz  = grid_z[1] - grid_z[0]
        idz = 1/dz

        # i,i-1    
        rows = np.array(range(1,n-1))
        cols = np.array(range(0,n-2))
        data = -np.ones(n-2) * idz * 0.5

        # i, i+1
        rows = np.append(rows, np.array(range(1,n-1)))
        cols = np.append(cols, np.array(range(2,n)))
        data = np.append(data, np.ones(n-2) * idz * 0.5)

        # # left boundary 
        rows = np.append(rows, np.array([0,0]))
        cols = np.append(cols, np.array([0,1]))
        data = np.append(data, np.array([-1.0 * idz, 1.0 * idz]))

        # # right boundary. 
        rows = np.append(rows, np.array([n-1,n-1]))
        cols = np.append(cols, np.array([n-1,n-2]))
        data = np.append(data, np.array([1.0 * idz, -1.0 * idz]))

        op=scipy.sparse.csr_array((data, (rows, cols)), shape=(n,n))
        #print(op.toarray()/idz)
        return op

    def host_to_device(self):
        self._inverse_laplacian = cp_sparse.csr_matrix(self._inverse_laplacian)
        self._e_op              = cp_sparse.csr_matrix(self._e_op)
        
        self._xp_module  = cp

        return

    def device_to_host(self):
        self._inverse_laplacian = self._inverse_laplacian.get()
        self._e_op              = self._e_op.get() 
        
        self._xp_module  = np

        return

    def setup(self,args):
        self._ne_ni_nfac = args.ne_fac  # 1/m^3
        self._inverse_laplacian = scipy.sparse.csr_matrix(BEUtils.choloskey_inv(-1.0 * self.fd1_laplace_1d_mat().toarray())) * (1.0 / self._epsilon_0)
        self._e_op      = self.e_field_op()
        self._xp_module = np

        return
        
    def poission_solve(self, ni, ne, freq_f, t_c, v0):
        xp        = self._xp_module
        f_rhs     = (ni-ne) * self._ne_ni_nfac
        f_rhs[0]  = -v0 * xp.sin(2 * xp.pi * freq_f * t_c) * self._epsilon_0
        f_rhs[-1] = 0.0
        u         = self._inverse_laplacian.dot(f_rhs)
        return u

    def compute_electric_field(self, u):
        return -self._e_op.dot(u)
        
class Fluids_1D():
    """
    Fluid equation solver, for n_i
    """
    def __init__(self, grid_z) -> None:
        self._grid_z = grid_z
        self._idz    = 1.0/(self._grid_z[1] - self._grid_z[0])

        num_z    = len(self._grid_z)
        self._mu         = 0.0
        self._D          = 0.0

        self._ne         = np.zeros(num_z)
        self._nn         = np.zeros(num_z)
        self._reaction_k = np.zeros(num_z)
        self._v_field    = np.zeros(num_z)
        
        return

    def host_to_device(self):
        self._nn          = cp.asarray(self._nn)
        self._ne          = cp.asarray(self._ne)
        self._reaction_k  = cp.asarray(self._reaction_k)
        self._v_field     = cp.asarray(self._v_field)

        self._flx_grad1_p = cp_sparse.csr_matrix(self._flx_grad1_p)
        self._flx_grad1_m = cp_sparse.csr_matrix(self._flx_grad1_m)
        self._flx_p       = cp_sparse.csr_matrix(self._flx_p)
        self._flx_m       = cp_sparse.csr_matrix(self._flx_m)

        
        self._xp_module  = cp

        return

    def device_to_host(self):
        self._nn         = cp.asnumpy(self._nn)
        self._ne         = cp.asnumpy(self._ne)
        self._reaction_k = cp.asnumpy(self._reaction_k)
        self._v_field    = cp.asnumpy(self._v_field)

        self._flx_grad1_p = self._flx_grad1_p.get()
        self._flx_grad1_m = self._flx_grad1_m.get()
        self._flx_p       = self._flx_p.get()
        self._flx_m       = self._flx_m.get()
        
        self._xp_module  = np
        return

    def setup(self, args):
        self._t_curr     = 0.0
        self._xp_module  = np

        n    = len(self._grid_z)
        idz  = 1.0 / (self._grid_z[1]-self._grid_z[0])

        rows = np.array([],dtype=int);  cols = np.array([],dtype=int); data = np.array([],dtype=float)
        # i, i+1
        rows = np.append(rows, np.array(range(1,n-1)))
        cols = np.append(cols, np.array(range(2,n  )))
        data = np.append(data, np.ones(n-2) * idz)
        #i, i
        rows = np.append(rows, np.array(range(1,n-1)))
        cols = np.append(cols, np.array(range(1,n-1)))
        data = np.append(data, -np.ones(n-2) * idz)

        self._flx_grad1_p = scipy.sparse.csr_array((data, (rows, cols)), shape=(n,n))

        
        rows = np.array([],dtype=int);  cols = np.array([],dtype=int); data = np.array([],dtype=float)
        # i, i
        rows = np.append(rows, np.array(range(1,n-1)))
        cols = np.append(cols, np.array(range(1,n-1)))
        data = np.append(data, np.ones(n-2) * idz)
        #i, i-1
        rows = np.append(rows, np.array(range(1,n-1)))
        cols = np.append(cols, np.array(range(0,n-2)))
        data = np.append(data, -np.ones(n-2) * idz)

        self._flx_grad1_m = scipy.sparse.csr_array((data, (rows, cols)), shape=(n,n))
        
        rows = np.array([],dtype=int);  cols = np.array([],dtype=int); data = np.array([],dtype=float)
        # i, i+1
        rows = np.append(rows, np.array(range(1,n-1)))
        cols = np.append(cols, np.array(range(2,n  )))
        data = np.append(data, np.ones(n-2) * 0.5)
        #i, i
        rows = np.append(rows, np.array(range(1,n-1)))
        cols = np.append(cols, np.array(range(1,n-1)))
        data = np.append(data, np.ones(n-2) * 0.5)

        self._flx_p = scipy.sparse.csr_array((data, (rows, cols)), shape=(n,n))

        rows = np.array([],dtype=int);  cols = np.array([],dtype=int); data = np.array([],dtype=float)
        # i, i
        rows = np.append(rows, np.array(range(1,n-1)))
        cols = np.append(cols, np.array(range(1,n-1)))
        data = np.append(data, np.ones(n-2) * 0.5)
        #i, i-1
        rows = np.append(rows, np.array(range(1,n-1)))
        cols = np.append(cols, np.array(range(0,n-2)))
        data = np.append(data, np.ones(n-2) * 0.5)

        self._flx_m = scipy.sparse.csr_array((data, (rows, cols)), shape=(n,n))

        # print(self._flx_grad1_p.toarray()/idz)
        # print(self._flx_grad1_m.toarray()/idz)

        # print(self._flx_p.toarray())
        # print(self._flx_m.toarray())

        if USE_GPU:
            @cp.fuse()
            def rhs_a(j_p,j_m, r_rate, nn, ne):
                return -(j_p - j_m)  + r_rate * nn * ne
            
            @cp.fuse()
            def rhs_j(flx_y_mu_idz, flx_g_v, flx_g_y_D_idz):
                return -flx_y_mu_idz * flx_g_v - flx_g_y_D_idz

            self._rhs_a = rhs_a
            self._rhs_j = rhs_j
            
        else:
            
            def rhs_j(flx_y_mu_idz, flx_g_v, flx_g_y_D_idz):
                return -flx_y_mu_idz * flx_g_v - flx_g_y_D_idz
            
            def rhs_a(j_p,j_m, r_rate, nn, ne):
                return -(j_p - j_m)  + r_rate * nn * ne

            self._rhs_a = rhs_a
            self._rhs_j = rhs_j
        
        return

    def apply_bc(self, t, y):
        D      = self._D
        mu     = self._mu
        v      = self._v_field
        idz    = self._idz
        
        # E_0 dot e_k > 0 set incoming flux to be zero
        E_0 = max(  -(v[1]-v[0])  * idz, 0.0)
        
        # E_L dot e_k < 0 set incoming flux to be zero
        E_L = min( -(v[-1]-v[-2]) * idz, 0.0)

        #print(E_0, E_L)

        y[0]  = y[1]  * (D * idz - mu * E_0 * 0.5 ) / (D * idz + mu * E_0 * 0.5)
        y[-1] = y[-2] * (D * idz + mu * E_L * 0.5)  / (D * idz - mu * E_L * 0.5)

        return y

    def rhs(self, t, y):
        n      = len(self._grid_z)
        idz    = self._idz
        v      = self._v_field
        mu     = self._mu
        D      = self._D
        r_rate = self._reaction_k
        
        ne     = self._ne
        nn     = self._nn

        y_mu_idz  = y * (mu * idz)
        y_D_idz   = y * (D * idz)

        flx_p_y_mu_idz = self._flx_p.dot(y_mu_idz)
        flx_m_y_mu_idz = self._flx_m.dot(y_mu_idz)
        flx_gp_v       = self._flx_grad1_p.dot(v)
        flx_gm_v       = self._flx_grad1_m.dot(v)
        flx_gp_y_D_idz = self._flx_grad1_p.dot(y_D_idz)
        flx_gm_y_D_idz = self._flx_grad1_m.dot(y_D_idz)

        j_p   = self._rhs_j(flx_p_y_mu_idz, flx_gp_v, flx_gp_y_D_idz) #-self._flx_p.dot(y) * (mu * idz) * self._flx_grad1_p.dot(v) - (D*idz) * self._flx_grad1_p.dot(y)
        j_m   = self._rhs_j(flx_m_y_mu_idz, flx_gm_v, flx_gm_y_D_idz) #-self._flx_m.dot(y) * (mu * idz) * self._flx_grad1_m.dot(v) - (D*idz) * self._flx_grad1_m.dot(y)
        return self._rhs_a(j_p, j_m, r_rate, nn, ne)
    
    def integrate(self, u, dt):
        ut =  u + dt * self.rhs(self._t_curr, u)
        self._t_curr+=dt
        ut =  self.apply_bc(self._t_curr, ut)
        return ut

    def integrate_rk2(self, u, dt):
        k1 = dt * self.rhs(self._t_curr, u)
        s1 = self.apply_bc(self._t_curr + 0.5 * dt, u + 0.5 * k1)

        k2 = dt * self.rhs(self._t_curr + 0.5 * dt, s1)
        u  = u + (1.0/6.0) * (k1 + 2 * k2)
        self._t_curr+=dt

        u  = self.apply_bc(self._t_curr, u)
        return u
    
    def integrate_rk4(self, u, dt):
        k1 = dt * self.rhs(self._t_curr, u)
        s1 = self.apply_bc(self._t_curr + 0.5 * dt , u + 0.5 * k1)

        k2 = dt * self.rhs(self._t_curr + 0.5 * dt , s1)
        s2 = self.apply_bc(self._t_curr + 0.5 * dt , u + 0.5 * k2)

        k3 = dt * self.rhs(self._t_curr + 0.5 * dt , s2)
        s3 = self.apply_bc(self._t_curr + dt , u + k3)

        k4 = dt * self.rhs(self._t_curr + dt , s3)
 
        # Update next value of y
        u = u + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)
        self._t_curr+=dt
        u  = self.apply_bc(self._t_curr, u)

        return u
        
class Boltzmann_1D():
    """
    1d boltzmann equation solver with collocation method in v_\theta and z direction. 
    """
    def __init__(self,grid_z, grid_vt) -> None:

        self._grid_z   = grid_z
        self._grid_vt  = grid_vt
        self._E_field  = 0.0
        
        return

    def setup(self, args):
        """
        assembles the velocity space operators
        i.e., v-space advection and v-space collision operator
        """
        vth                            = collisions.electron_thermal_velocity(args.electron_volt*args.basis_scale * collisions.TEMP_K_1EV)
        maxwellian                     = BEUtils.get_maxwellian_3d(vth,collisions.MAXWELLIAN_N)
        c_gamma                        = np.sqrt(2*collisions.ELECTRON_CHARGE_MASS_RATIO)
        if "g2Smooth" in args.collisions or "g2" in args.collisions or "g2step" in args.collisions or "g2Regul" in args.collisions:
            sig_pts = np.array([np.sqrt(15.76) * c_gamma/vth])
        else:
            sig_pts = None 
        
        if len(np.where(sig_pts> args.vx_max)[0]) > 0:
            print("provided vx_max %2E does not include reaction threshold %s" %(args.vx_max, str(sig_pts)))
            sys.exit(0)

        k_domain = (0, args.vx_max)
        ev_range = ((k_domain[0] * vth / c_gamma)**2, (k_domain[1] * vth / c_gamma)**2)
        #(np.sqrt(ev_range[0]) * c_gamma / vth, np.sqrt(ev_range[1]) * c_gamma / vth)

        self._c_gamma        = c_gamma
        self._eavg_to_K      = (2/(3*scipy.constants.Boltzmann))
        self._v_space_domain = (np.sqrt(ev_range[0]) * c_gamma / vth, np.sqrt(ev_range[1]) * c_gamma / vth)
        
        print("target ev range : (%.4E, %.4E) ----> knots domain : (%.4E, %.4E)" %(ev_range[0], ev_range[1], k_domain[0],k_domain[1]))
        if(sig_pts is not None):
            print("singularity pts : ", sig_pts, "v/vth and" , (sig_pts * vth/c_gamma)**2,"eV")

        params.BEVelocitySpace.SPH_HARM_LM  = [[l,0] for l in range(args.l_max+1)]
        params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER = args.NUM_P_RADIAL
        basis.BSPLINE_BASIS_ORDER=args.spline_order
        basis.XLBSPLINE_NUM_Q_PTS_PER_KNOT=args.spline_q_pts_per_knot
        if (args.radial_poly == "bspline"):
            bb     = basis.BSpline(k_domain, args.spline_order, params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER+1, sig_pts=sig_pts, knots_vec=None)
            params.BEVelocitySpace.NUM_Q_VR = bb._num_knot_intervals * args.spline_q_pts_per_knot
            params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER = bb._num_p
            spec_sp               = sp.SpectralExpansionSpherical(params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER-1, bb,params.BEVelocitySpace.SPH_HARM_LM)
            spec_sp._num_q_radial = params.BEVelocitySpace.NUM_Q_VR
            self._dv              = (spec_sp._basis_p._t[args.spline_order + 1] - spec_sp._basis_p._t[args.spline_order])
            self._e_field_max     = args.v_cfl * self._dv * vth / args.T_DT / collisions.ELECTRON_CHARGE_MASS_RATIO
            print("v_thermal = %8E \t dv = %8E dt=%8E E_max =%8E"%(vth, self._dv, args.T_DT, self._e_field_max))
            
        else:
            raise NotImplementedError

        params.BEVelocitySpace.NUM_Q_VT  = args.quad_theta
        params.BEVelocitySpace.NUM_Q_VP  = args.quad_phi
        params.BEVelocitySpace.NUM_Q_CHI = args.quad_s_theta
        params.BEVelocitySpace.NUM_Q_PHI = args.quad_s_phi

        mw_vth   = maxwellian
        vth_curr = vth
        self._maxwellian = mw_vth
        self._vth        = vth_curr

        FOp=0
        collision_op = colOpSp.CollisionOpSP(spec_sp = spec_sp)
        self._reaction_op  = dict()
        if "g0" in args.collisions:
            g0  = collisions.eAr_G0()
            g0.reset_scattering_direction_sp_mat()
            FOp = FOp + collisions.AR_NEUTRAL_N * collision_op.assemble_mat(g0, mw_vth, vth_curr)
            self._reaction_op["g0"] = BEUtils.reaction_rates_op(spec_sp, g0, self._maxwellian, self._vth)
            
        if "g0ConstNoLoss" in args.collisions:
            g0  = collisions.eAr_G0_NoEnergyLoss(cross_section="g0Const")
            g0.reset_scattering_direction_sp_mat()
            FOp = FOp + collisions.AR_NEUTRAL_N * collision_op.assemble_mat(g0, mw_vth, vth_curr)
            self._reaction_op["g0ConstNoLoss"] = BEUtils.reaction_rates_op(spec_sp, g0, self._maxwellian, self._vth)
            
        if "g0NoLoss" in args.collisions:
            g0  = collisions.eAr_G0_NoEnergyLoss()
            g0.reset_scattering_direction_sp_mat()
            FOp = FOp + collisions.AR_NEUTRAL_N * collision_op.assemble_mat(g0, mw_vth, vth_curr)
            self._reaction_op["g0NoLoss"] = BEUtils.reaction_rates_op(spec_sp, g0, self._maxwellian, self._vth)
            
        if "g0Const" in args.collisions:
            g0  = collisions.eAr_G0(cross_section="g0Const")
            g0.reset_scattering_direction_sp_mat()
            FOp = FOp + collisions.AR_NEUTRAL_N * collision_op.assemble_mat(g0, mw_vth, vth_curr)
            self._reaction_op["g0Const"] = BEUtils.reaction_rates_op(spec_sp, g0, self._maxwellian, self._vth)
            
        if "g2" in args.collisions:
            g2  = collisions.eAr_G2()
            g2.reset_scattering_direction_sp_mat()
            FOp = FOp + collisions.AR_NEUTRAL_N * collision_op.assemble_mat(g2, mw_vth, vth_curr)
            self._reaction_op["g2"] = BEUtils.reaction_rates_op(spec_sp, g2, self._maxwellian, self._vth)

        if "g2Smooth" in args.collisions:
            g2  = collisions.eAr_G2(cross_section="g2Smooth")
            g2.reset_scattering_direction_sp_mat()
            FOp = FOp + collisions.AR_NEUTRAL_N * collision_op.assemble_mat(g2, mw_vth, vth_curr)
            self._reaction_op["g2Smooth"] = BEUtils.reaction_rates_op(spec_sp, g2, self._maxwellian, self._vth)

        if "g2Regul" in args.collisions:
            g2  = collisions.eAr_G2(cross_section="g2Regul")
            g2.reset_scattering_direction_sp_mat()
            FOp = FOp + collisions.AR_NEUTRAL_N * collision_op.assemble_mat(g2, mw_vth, vth_curr)
            self._reaction_op["g2Regul"] = BEUtils.reaction_rates_op(spec_sp, g2, self._maxwellian, self._vth)
            
        if "g2Const" in args.collisions:
            g2  = collisions.eAr_G2(cross_section="g2Const")
            g2.reset_scattering_direction_sp_mat()
            FOp = FOp + collisions.AR_NEUTRAL_N * collision_op.assemble_mat(g2, mw_vth, vth_curr)
            self._reaction_op["g2Const"] = BEUtils.reaction_rates_op(spec_sp, g2, self._maxwellian, self._vth)
            
        if "g2step" in args.collisions:
            g2  = collisions.eAr_G2(cross_section="g2step")
            g2.reset_scattering_direction_sp_mat()
            FOp = FOp + collisions.AR_NEUTRAL_N * collision_op.assemble_mat(g2, mw_vth, vth_curr)
            self._reaction_op["g2step"] = BEUtils.reaction_rates_op(spec_sp, g2, self._maxwellian, self._vth)
            
        if "g2smoothstep" in args.collisions:
            g2  = collisions.eAr_G2(cross_section="g2smoothstep")
            g2.reset_scattering_direction_sp_mat()
            FOp = FOp + collisions.AR_NEUTRAL_N * collision_op.assemble_mat(g2, mw_vth, vth_curr)
            self._reaction_op["g2smoothstep"] = BEUtils.reaction_rates_op(spec_sp, g2, self._maxwellian, self._vth)
            
        Cmat       = FOp
        Emat       = spec_sp.compute_advection_matix() * (collisions.ELECTRON_CHARGE_MASS_RATIO/self._vth)

        self._spec_sp = spec_sp
        self._col_op  = collision_op

        self._Cmat    = Cmat
        self._Emat    = Emat 

        self._lm_to_vt = lm_modes_to_vt_pts_op(self._grid_z, self._spec_sp, self._grid_vt)
        self._vt_to_lm = vt_pts_to_lm_modes_op(self._grid_z, self._spec_sp, self._grid_vt)

        self._Dx_p     = grad1_p_11_mat(self._grid_z)#grad1_p_11_mat(self._grid_z)
        self._Dx_m     = grad1_m_11_mat(self._grid_z)#grad1_m_11_mat(self._grid_z)
        self._Dx_c     = grad1_11_mat(self._grid_z)

        num_p          = spec_sp._p+1
        num_sh         = len(spec_sp._sph_harm_lm)
        self._mmat     = spec_sp.compute_mass_matrix()
        self._inv_mmat = BEUtils.choloskey_inv(self._mmat)

        self._mmat_radial     = self._mmat[0::num_sh, 0::num_sh]
        self._inv_mmat_radial = BEUtils.choloskey_inv(self._mmat_radial)
        
        
        [gx, gw] = spec_sp._basis_p.Gauss_Pn(spec_sp._num_q_radial)
        vqr = spec_sp.Vq_r(gx, 0, 1)

        self._Wpk_mat = np.zeros((num_p, num_p))
        for i in range(num_p):
            for j in range(num_p):
                self._Wpk_mat[i,j] = vth_curr * np.dot(gw, vqr[i,:] * vqr[j,:] * gx**3)
        

        self._mass_op      = BEUtils.mass_op(spec_sp, None, 8, 2, 1.0) * (self._vth**3)
        self._temp_op      = BEUtils.temp_op(spec_sp, None, 8, 2, 1.0) * (self._vth**5) * (0.5 * collisions.MASS_ELECTRON * self._eavg_to_K / collisions.TEMP_K_1EV)
        
        self._idx_p_upwind =  np.array([k * len(self._grid_vt) + m for k in range(num_p) for m in np.where(self._grid_vt <= 0.5 * np.pi)[0]]).reshape(-1) 
        self._idx_m_upwind =  np.array([k * len(self._grid_vt) + m for k in range(num_p) for m in np.where(self._grid_vt >  0.5 * np.pi)[0]]).reshape(-1)

        # print(self._idx_p_upwind)
        # print(self._idx_m_upwind) 

        self._cos_vt   = np.array([np.cos(self._grid_vt) for m in range(num_p)]).reshape(-1)
        # print(np.cos(self._grid_vt))
        # print(self._cos_vt)
        
        self._cos_vt_p = self._cos_vt[self._idx_p_upwind]
        self._cos_vt_m = self._cos_vt[self._idx_m_upwind]

        #print(self._cos_vt_p)
        #print(self._cos_vt_m)

        self._Dzy_kvt  = np.zeros((len(self._grid_z), num_p * (len(self._grid_vt))))

        self._vt_bdy_p = np.where(self._grid_vt <= 0.5 * np.pi)[0]
        self._vt_bdy_m = np.where(self._grid_vt >  0.5 * np.pi)[0]
        # print(self._vt_bdy_p)
        # print(self._vt_bdy_m)

        self._t_curr        = 0.0
        self._xp_module     = np

        # zero out the sph integration for higher numerical accuracy , this is exactly zero due to the orthogonality of sph
        self._mass_op[1::num_sh]=0.0
        self._temp_op[1::num_sh]=0.0
        
        tmp_dict=dict()
        for (reaction, rates_op) in self._reaction_op.items():
            rates_op[1::num_sh]=0.0
            tmp_dict[reaction]=rates_op
        
        self._reaction_op.clear()
        self._reaction_op=tmp_dict

        self._eta_x         = args.eta_z
        self._lop_diffusion = (scipy.sparse.identity(len(self._grid_z), format='csr') - self._eta_x * args.T_DT * laplace_op_fd_11(grid_z=self._grid_z)).toarray()
        self._lop_diffusion = scipy.sparse.csr_matrix(BEUtils.choloskey_inv(self._lop_diffusion))

        # w, u = scipy.linalg.eig(self._lop_diffusion.toarray())
        # print(w)
        


        return

    def host_to_device(self):
        self._Cmat            = cp.asarray(self._Cmat)
        self._Emat            = cp.asarray(self._Emat)
        self._lm_to_vt        = cp.asarray(self._lm_to_vt)
        self._vt_to_lm        = cp.asarray(self._vt_to_lm)

        self._Dx_p            = cp_sparse.csr_matrix(self._Dx_p) 
        self._Dx_m            = cp_sparse.csr_matrix(self._Dx_m)
        self._Dx_c            = cp_sparse.csr_matrix(self._Dx_c)
        
        self._lop_diffusion   = cp_sparse.csr_matrix(self._lop_diffusion)
        self._mmat            = cp.asarray(self._mmat)
        self._inv_mmat        = cp.asarray(self._inv_mmat)
        self._mmat_radial     = cp.asarray(self._mmat_radial)
        self._inv_mmat_radial = cp.asarray(self._inv_mmat_radial)
        self._Wpk_mat         = cp.asarray(self._Wpk_mat)
        
        self._cos_vt          = cp.asarray(self._cos_vt)
        
        self._cos_vt_p        = cp.asarray(self._cos_vt_p)
        self._cos_vt_m        = cp.asarray(self._cos_vt_m)

        self._Dzy_kvt         = cp.asarray(self._Dzy_kvt)
        self._mass_op         = cp.asarray(self._mass_op)
        self._temp_op         = cp.asarray(self._temp_op)
        self._vt_bdy_p        = cp.asarray(self._vt_bdy_p)
        self._vt_bdy_m        = cp.asarray(self._vt_bdy_m)

        for r,rr_op in self._reaction_op.items():
            self._reaction_op[r] = cp.asarray(rr_op)

        self._xp_module       = cp

        return

    def device_to_host(self):
        self._xp_module       = np

        self._Cmat            = cp.asnumpy(self._Cmat)
        self._Emat            = cp.asnumpy(self._Emat)
        self._lm_to_vt        = cp.asnumpy(self._lm_to_vt)
        self._vt_to_lm        = cp.asnumpy(self._vt_to_lm)
        
        self._Dx_p            = self._Dx_p.get()
        self._Dx_m            = self._Dx_m.get()
        self._Dx_c            = self._Dx_c.get()

        self._lop_diffusion   = self._lop_diffusion.get()
        self._mmat            = cp.asnumpy(self._mmat)
        self._inv_mmat        = cp.asnumpy(self._inv_mmat)
        self._mmat_radial     = cp.asnumpy(self._mmat_radial)
        self._inv_mmat_radial = cp.asnumpy(self._inv_mmat_radial)
        self._Wpk_mat         = cp.asnumpy(self._Wpk_mat)
        
        self._cos_vt          = cp.asnumpy(self._cos_vt)
        self._cos_vt_p        = cp.asnumpy(self._cos_vt_p)
        self._cos_vt_m        = cp.asnumpy(self._cos_vt_m)

        self._Dzy_kvt         = cp.asnumpy(self._Dzy_kvt)
        self._mass_op         = cp.asnumpy(self._mass_op)
        self._temp_op         = cp.asnumpy(self._temp_op)
        self._vt_bdy_p        = cp.asnumpy(self._vt_bdy_p)
        self._vt_bdy_m        = cp.asnumpy(self._vt_bdy_m)

        for r,rr_op in self._reaction_op.items():
            self._reaction_op[r] = cp.asnumpy(rr_op)
        
        return

    def rhs_xspace(self, t, y):

        xp      = self._xp_module
        spec_sp = self._spec_sp
        num_p   = spec_sp._p + 1
        num_sh  = len(spec_sp._sph_harm_lm)

        num_vt  = len(self._grid_vt)
        num_z   = len(self._grid_z)

        Dzy_kvt    = self._Dzy_kvt
        yy         = y.reshape((num_z, num_p *num_vt))

        Dzy_kvt[:,self._idx_p_upwind] = self._Dx_p.dot(yy[:,self._idx_p_upwind]) * self._cos_vt_p
        Dzy_kvt[:,self._idx_m_upwind] = self._Dx_m.dot(yy[:,self._idx_m_upwind]) * self._cos_vt_m

        Dzy_kvt    = Dzy_kvt.transpose().reshape((num_p, num_vt, num_z)).reshape((num_p, num_vt * num_z))
        Dzy_kvt    = xp.matmul(self._inv_mmat_radial,xp.matmul(self._Wpk_mat, Dzy_kvt))
        Dzy_kvt    = Dzy_kvt.reshape((num_p , num_vt, num_z)).reshape((num_p * num_vt, num_z)).transpose()

        Dzy_kvt    = -Dzy_kvt.reshape(num_z * num_p * num_vt)
        return Dzy_kvt

    def rhs_vspace(self, t, y):

        xp      = self._xp_module
        spec_sp = self._spec_sp
        num_p   = spec_sp._p + 1
        num_sh  = len(spec_sp._sph_harm_lm)

        num_vt  = len(self._grid_vt)
        num_z   = len(self._grid_z)

        yy         = y.reshape((num_z, num_p, num_vt))
        yy         = xp.transpose(xp.dot(yy,self._vt_to_lm).reshape((num_z, num_p*num_sh)))

        CpE_klm    = xp.matmul(self._Cmat, yy)
        CpE_klm   += xp.matmul(self._Emat, yy) * (self._E_field)
        CpE_klm    = xp.transpose(CpE_klm)

        CpE_kvt    = xp.dot(CpE_klm.reshape(num_z, num_p , num_sh), self._lm_to_vt).reshape((num_z, num_p * num_vt))
        CpE_kvt    = xp.transpose(CpE_kvt).reshape((num_p, num_vt, num_z)).reshape((num_p, num_vt * num_z))
        CpE_kvt    = xp.matmul(self._inv_mmat_radial, CpE_kvt)
        CpE_kvt    = CpE_kvt.reshape((num_p , num_vt, num_z)).reshape((num_p * num_vt, num_z)).transpose()
        CpE_kvt    = CpE_kvt.reshape(num_z * num_p * num_vt) 

        return CpE_kvt

    def rhs(self, t, y):
        xp      = self._xp_module
        spec_sp = self._spec_sp
        num_p   = spec_sp._p + 1
        num_sh  = len(spec_sp._sph_harm_lm)

        num_vt  = len(self._grid_vt)
        num_z   = len(self._grid_z)

        yy         = y.reshape((num_z, num_p, num_vt))
        yy         = xp.transpose(xp.dot(yy,self._vt_to_lm).reshape((num_z, num_p*num_sh)))

        CpE_klm    = xp.matmul(self._Cmat, yy)
        CpE_klm   += xp.matmul(self._Emat, yy) * (self._E_field)
        CpE_klm    = xp.transpose(CpE_klm)

        CpE_kvt    = xp.dot(CpE_klm.reshape(num_z, num_p , num_sh), self._lm_to_vt).reshape((num_z, num_p * num_vt))
        CpE_kvt    = xp.transpose(CpE_kvt).reshape((num_p, num_vt, num_z)).reshape((num_p, num_vt * num_z))
        CpE_kvt    = xp.matmul(self._inv_mmat_radial, CpE_kvt)
        CpE_kvt    = CpE_kvt.reshape((num_p , num_vt, num_z)).reshape((num_p * num_vt, num_z)).transpose()

        Dzy_kvt    = self._Dzy_kvt
        yy         = y.reshape((num_z, num_p *num_vt))
        
        #Dzy_kvt    = self._Dx_c.dot(yy) * self._cos_vt
        Dzy_kvt[:,self._idx_p_upwind] = self._Dx_p.dot(yy[:,self._idx_p_upwind]) * self._cos_vt_p
        Dzy_kvt[:,self._idx_m_upwind] = self._Dx_m.dot(yy[:,self._idx_m_upwind]) * self._cos_vt_m

        Dzy_kvt    = Dzy_kvt.transpose().reshape((num_p, num_vt, num_z)).reshape((num_p, num_vt * num_z))
        Dzy_kvt    = xp.matmul(self._inv_mmat_radial,xp.matmul(self._Wpk_mat, Dzy_kvt))
        Dzy_kvt    = Dzy_kvt.reshape((num_p , num_vt, num_z)).reshape((num_p * num_vt, num_z)).transpose()
        y_rhs      = (CpE_kvt - Dzy_kvt).reshape(num_z * num_p * num_vt) 

        return y_rhs

    def apply_bc(self, t, y):
        xp      = self._xp_module
        spec_sp = self._spec_sp
        num_p   = spec_sp._p + 1
        num_sh  = len(spec_sp._sph_harm_lm)

        num_vt  = len(self._grid_vt)
        num_z   = len(self._grid_z)

        y = y.reshape((num_z, num_p, num_vt))

        y[0 , :, self._vt_bdy_p]  = 0.0
        y[-1, :, self._vt_bdy_m]  = 0.0

        # y[0 , :, :]  = 0.0
        # y[-1, :, :]  = 0.0

        # ww = self._cos_vt.reshape(num_vt, num_p).transpose()
        # y[0  , :, :]   = 1e-28 #1e-24 - 1e-25 * ww   
        # y[-1  , :, :]  = 1e-28 #1e-24 - 1e-25 * ww   
        
        
        return y.reshape(num_z * num_p * num_vt)

    def integrate(self, u, dt):
        u = u + dt * self.rhs(self._t_curr, u)
        self._t_curr += 1.0 * dt
        u = self.apply_bc(self._t_curr, u)
        
        # #operator split method
        # u = u + 0.5 * dt * self.rhs_vspace(self._t_curr, u)
        # self._t_curr +=  0.5 * dt
        # u = self.apply_bc(self._t_curr, u)
        # u = u + 0.5 * dt * self.rhs_xspace(self._t_curr, u)
        # self._t_curr += 0.5 * dt
        # u = self.apply_bc(self._t_curr, u)
        return u

    def integrate_rk2(self, u, dt):
        k1 = dt * self.rhs(self._t_curr, u)
        s1 = self.apply_bc(self._t_curr, u + 0.5 * k1)

        k2 = dt * self.rhs(self._t_curr + 0.5 * dt, s1)
        u  = u + (1.0/6.0) * (k1 + 2 * k2)
        self._t_curr+=dt
        u  = self.apply_bc(self._t_curr, u)

        return u       

    def integrate_rk4(self, u, dt):
        k1 = dt * self.rhs(self._t_curr, u)
        s1 = self.apply_bc(self._t_curr + 0.5 * dt , u + 0.5 * k1)

        k2 = dt * self.rhs(self._t_curr + 0.5 * dt , s1)
        s2 = self.apply_bc(self._t_curr + 0.5 * dt , u + 0.5 * k2)

        k3 = dt * self.rhs(self._t_curr + 0.5 * dt , s2)
        s3 = self.apply_bc(self._t_curr + dt , u + k3)

        k4 = dt * self.rhs(self._t_curr + dt , s3)
 
        # Update next value of y
        u = u + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)
        self._t_curr+=dt
        u  = self.apply_bc(self._t_curr, u)

        return u

    def compute_ne(self, f_kvt):
        xp      = self._xp_module
        spec_sp = self._spec_sp
        num_p   = spec_sp._p + 1
        num_sh  = len(spec_sp._sph_harm_lm)

        num_vt  = len(self._grid_vt)
        num_z   = len(self._grid_z)
        yy      = xp.dot(f_kvt.reshape((num_z, num_p , num_vt)),self._vt_to_lm).reshape((num_z, num_p*num_sh))
        ne      = xp.dot(self._mass_op, yy.transpose()).reshape(num_z)
        return ne

    def compute_avg_energy(self, f_kvt):
        xp      = self._xp_module
        spec_sp = self._spec_sp
        num_p   = spec_sp._p + 1
        num_sh  = len(spec_sp._sph_harm_lm)

        num_vt  = len(self._grid_vt)
        num_z   = len(self._grid_z)
        yy      = xp.dot(f_kvt.reshape((num_z, num_p , num_vt)),self._vt_to_lm).reshape((num_z, num_p*num_sh))
        m0      = self.compute_ne(f_kvt)
        Te      = xp.dot(self._temp_op, yy.transpose()).reshape(num_z)  / m0
        return Te
    
    def compute_reaction_rates(self, rates_op_dict, f_kvt):
        xp      = self._xp_module
        spec_sp = self._spec_sp
        num_p   = spec_sp._p + 1
        num_sh  = len(spec_sp._sph_harm_lm)

        num_vt  = len(self._grid_vt)
        num_z   = len(self._grid_z)
        
        yy      = xp.dot(f_kvt.reshape((num_z, num_p , num_vt)),self._vt_to_lm).reshape((num_z, num_p*num_sh))
        yy      = yy.transpose()

        # we need the normalization \int_{\varepsilon} \int_{\omega}\varepslison^{1/2} b_k(v/vth) Y_lm (\omega) d\omega d\varepsilon
        m0      = xp.dot(self._mass_op, yy).reshape(num_z) * (2 / self._c_gamma**3)

        ki  =  dict()
        for (reaction, rates_op) in rates_op_dict.items():
            rr = np.dot(rates_op, yy).reshape(num_z) / m0
            ki[reaction] = rr
            
            
        return ki

class GlowSim_1D():
    """
    Main glow simulation driver for 1d. 
    """
    def __init__(self, args) -> None:
        self._grid_z   = np.linspace(0, args.Lz, args.num_z)
        self._grid_vt  = np.arccos(basis.Legendre().Gauss_Pn(args.quad_theta)[0])

        self._rf_freq  = args.rf_freq
        self._v0_z0    = args.v0_z0

        self._ts_atol  = args.ts_atol
        self._ts_rtol  = args.ts_rtol

        self._boltzmann_solver = Boltzmann_1D(self._grid_z, self._grid_vt)
        self._e_field_solver   = EField_1D(self._grid_z)
        self._fluids_solver    = Fluids_1D(self._grid_z)

        collisions.AR_NEUTRAL_N = args.n0
        collisions.MAXWELLIAN_N = args.ne

        return

    def read_initial_data(self, fname, Np, L):
        xp = -np.cos(np.pi*np.linspace(0,Np-1,Np)/(Np-1))
        xr = (xp+1)*L * 0.5
        
        D = np.load(fname)

        ne = D[0:Np].reshape(len(xr)) 
        ni = D[Np:2*Np].reshape(len(xr))
        nb = D[2*Np:3*Np].reshape(len(xr))
        
        neE = D[3*Np:].reshape(len(xr))
        Te = (2./3) * neE / ne

        ne = scipy.interpolate.interp1d(xr, ne)
        ni = scipy.interpolate.interp1d(xr, ni)
        Te = scipy.interpolate.interp1d(xr, Te)

        return ne, ni, Te

    def init(self, t0, t1, args):
        self._t_begin = t0
        self._t_end   = t1

        self._step_id = 0
        self._t_curr  = self._t_begin

        self._boltzmann_solver.setup(args)
        self._e_field_solver.setup(args)
        self._fluids_solver.setup(args)

        spec_sp = self._boltzmann_solver._spec_sp
        grid_z  = self._grid_z
        grid_vt = self._grid_vt

        num_z   = len(grid_z)
        num_p   = spec_sp._p + 1 
        num_sh  = len(spec_sp._sph_harm_lm)
        num_vt  = len(grid_vt)

        approx_ne, approx_ni, approx_te = self.read_initial_data(fname='newton_3spec_CN_Np100.npy', Np=100, L=args.Lz)
        
        self._ne_init = approx_ne(grid_z)
        self._ni_init = approx_ni(grid_z)
        self._Te_init = approx_te(grid_z)

        
        fname           = 'newton_3spec_CN_Np100_pde_sp%d_Nr%d_lmax%d_nz%d_vx_max_%.4E.npy'%(args.spline_order,args.NUM_P_RADIAL,args.l_max, args.num_z, args.vx_max)
        init_data_file  = pathlib.Path(fname)
        if not (init_data_file.is_file()):
            print("Generating initial data file %s"%(fname))
            self._fx_kvt = create_vec_lm(grid_z, spec_sp)
            m_fac        = 1.0 / ((np.pi**1.5) * self._boltzmann_solver._vth**3)
            mass_op      = self._boltzmann_solver._mass_op
            
            for i in range(len(grid_z)):
                vth_i  = collisions.electron_thermal_velocity(self._Te_init[i] * collisions.TEMP_K_1EV) 
                vratio = vth_i / self._boltzmann_solver._vth
                hv     = lambda v,vt,vp : self._ne_init[i] * m_fac * (np.exp(-((v/vratio)**2)) / vratio**3)
                self._fx_kvt[i,:] = BEUtils.function_to_basis(spec_sp, hv, self._boltzmann_solver._maxwellian, None, None, None, Minv=self._boltzmann_solver._inv_mmat)
            np.save(fname, self._fx_kvt)
        
        print("loading initial data from %s"%(fname))
        self._fx_kvt = np.load(fname)
        self._fx_kvt[0 ,:] = 0.0
        self._fx_kvt[-1,:] = 0.0

        # print(self._boltzmann_solver._lm_to_vt.shape)
        # print(self._boltzmann_solver._lm_to_vt)

        # print(self._boltzmann_solver._vt_to_lm.shape)
        # print(self._boltzmann_solver._vt_to_lm)
        # print(np.sum(self._boltzmann_solver._vt_to_lm, axis=0))

        self._fx_kvt = np.dot(self._fx_kvt.reshape((num_z, num_p, num_sh)) , self._boltzmann_solver._lm_to_vt).reshape(num_z * num_p * num_vt)
        print("applying bte bc on initial data")
        self._fx_kvt = self._boltzmann_solver.apply_bc(self._t_begin, self._fx_kvt)
        self._ne     = self._boltzmann_solver.compute_ne(self._fx_kvt)
        self._ni     = self._ni_init
        
        # print(self._ne)
        # print(self._ni)
        # self._ni     = np.array(self._ne) 
        # test initial data
        # self._ni = np.max(self._ne) * np.exp(-(self._grid_z - 0.5 * args.Lz)**2/1e-6 ) #np.array(self._ne)
        # self._ni[self._grid_z < 0.5 * args.Lz]=0.0

        self._fluids_solver._nn   = np.ones_like(grid_z) * 1.0 / self._e_field_solver._ne_ni_nfac
        self._fluids_solver._ne   = self._ne
        self._fluids_solver._mu   = 4.65e21 / collisions.AR_NEUTRAL_N
        self._fluids_solver._D    = 2.07e20 / collisions.AR_NEUTRAL_N

        all_reaction_rates        = self._boltzmann_solver.compute_reaction_rates(self._boltzmann_solver._reaction_op, self._fx_kvt)
        
        if "g2" in all_reaction_rates:
            self._fluids_solver._reaction_k   = all_reaction_rates["g2"] 
        else:
            self._fluids_solver._reaction_k   = np.zeros(num_z)
        
        self._boltzmann_solver._t_curr  = self._t_begin
        self._fluids_solver._t_curr     = self._t_begin
        
        self._e_field     = np.ones_like(self._ni) * args.E_field
        self._e_potential = np.ones_like(self._ni) * args.E_field
        self._xp_module   = np

        if(USE_GPU):
            self._e_field_solver.host_to_device()
            self._fluids_solver.host_to_device()
            self._boltzmann_solver.host_to_device()

            self._ne          = cp.asarray(self._ne)
            self._ni          = cp.asarray(self._ni)
            self._fx_kvt      = cp.asarray(self._fx_kvt)
            self._e_field     = cp.asarray(self._e_field)
            self._e_potential = cp.asarray(self._e_potential)

            self._xp_module   = cp 
            

        xp        = cp.get_array_module(self._ni)
        self._cycle_avg_e_poten   = xp.zeros_like(glow_sim_1d._e_potential)
        self._cycle_avg_e_field   = xp.zeros_like(glow_sim_1d._e_field)
        self._cycle_avg_ni        = xp.zeros_like(glow_sim_1d._ni)
        self._cycle_avg_ne        = xp.zeros_like(glow_sim_1d._ne)
        self._cycle_avg_fx_kvt    = xp.zeros_like(glow_sim_1d._fx_kvt)

        return

    def integrate(self, dt):
        ## solve for the E field
        self._e_potential = self._e_field_solver.poission_solve(self._ni, self._ne, self._rf_freq, self._t_curr, self._v0_z0)
        self._e_field     = self._e_field_solver.compute_electric_field(self._e_potential)
        
        ## solve for the E field
        self._boltzmann_solver._E_field   = self._e_field
        self._fluids_solver._v_field      = self._e_potential

        self._fx_kvt                      = self._boltzmann_solver.integrate_rk2(self._fx_kvt, dt) 
        self._ni                          = self._fluids_solver.integrate_rk2(self._ni, dt)
        
        self._ne                          = self._boltzmann_solver.compute_ne(self._fx_kvt)
        self._fluids_solver._ne           = self._ne
        all_reaction_rates                = self._boltzmann_solver.compute_reaction_rates(self._boltzmann_solver._reaction_op, self._fx_kvt)
        if "g2" in all_reaction_rates:
            self._fluids_solver._reaction_k   = all_reaction_rates["g2"]
        
        self._t_curr  += dt
        self._step_id += 1

        return 

    def plot(self, e_potential, e_field, ne, ni, fx_kvt, z_loc, fprefix,plot_pannel=True):

        num_z_plots = len(z_loc)
        grid_z  = self._grid_z
        grid_vt = self._grid_vt
        
        rows  = 2 
        cols  = 3

        if plot_pannel:
            fig = plt.figure(figsize=(12, 8), dpi=300)
            plt.suptitle("T=%4E (s)"%(self._t_curr))

            plt.subplot(rows, cols, 1)
            plt.semilogy(grid_z, self._ni_init[:],'r--',label='initial')
            plt.semilogy(grid_z, ni[:],label='pde')
            plt.xlabel("z (m)")
            plt.ylabel("# density (%.2E / m^3)"%(args.ne_fac))
            plt.grid()
            plt.legend()
            plt.title("ni")
            
            plt.subplot(rows, cols, 2)
            plt.semilogy(grid_z, self._ne_init[:],'r--',label='initial')
            plt.semilogy(grid_z, ne[:], label='pde')
            plt.legend()
            plt.xlabel("z (m)")
            plt.ylabel("# density (%.2E / m^3)"%(args.ne_fac))
            plt.grid()
            plt.title("ne")
            

            plt.subplot(rows, cols, 3)
            plt.plot(grid_z, e_field[:])
            plt.xlabel("z (m)")
            plt.ylabel("V/m")
            plt.grid()
            plt.title("E")
            
            c_rates = self._boltzmann_solver.compute_reaction_rates(self._boltzmann_solver._reaction_op, fx_kvt[:])
            plt.subplot(rows, cols, 4)
            for key, rr in c_rates.items():
                color = next(plt.gca()._get_lines.prop_cycler)['color']
                plt.semilogy(grid_z, rr, label="%s"%(COLLISOIN_NAMES[key]), color=color)

            plt.legend()
            plt.xlabel("z (m)")
            plt.ylabel("rate coefficient m^3/s")
            plt.grid()

            temp_list=self._boltzmann_solver.compute_avg_energy(fx_kvt[:])
            plt.subplot(rows, cols, 5)
            plt.plot(grid_z, self._Te_init[:], 'r--', label='initial')
            plt.plot(grid_z, temp_list[:], label='pde')
            plt.legend()
            plt.xlabel("z (m)")
            plt.ylabel("temperature (eV)")
            plt.grid()

            plt.subplot(rows, cols, 6)
            plt.plot(grid_z, e_potential[:])
            plt.xlabel("z (m)")
            plt.ylabel("Volt (V)")
            plt.grid()

            plt.tight_layout()
            fig.savefig("%s_qoi.png"%(fprefix))
            plt.close()

        fig = plt.figure(figsize=(8, 8), dpi=300)        
        plt.title("T=%4E (s)"%(self._t_curr))
        spec_sp = self._boltzmann_solver._spec_sp
        gx, gw  = spec_sp._basis_p.Gauss_Pn(params.BEVelocitySpace.NUM_Q_VR)
        mass_b  = np.dot(spec_sp.Vq_r(gx,0,1),gw) * self._boltzmann_solver._vth

        num_p  = spec_sp._p +1
        num_sh = len(spec_sp._sph_harm_lm)
        num_z  = len(grid_z)
        num_vt = len(grid_vt)

        fx = np.transpose(fx_kvt.reshape((num_z, num_p * num_vt))).reshape(num_p, num_vt, num_z).reshape(num_p, num_vt*num_z)
        fx = np.dot(mass_b, fx).reshape(num_vt, num_z)

        cols=3
        for ci in range(cols):
            plt.plot(grid_vt, fx[:,ci],'x--',label='z= %ddz'%(ci))
        
        for ci in range(cols):
            plt.plot(grid_vt, fx[:, ci-cols],'x--',label='z= L - %ddz'%(abs(ci-cols)))
            
        plt.legend()            
        plt.xlabel("theta")
        plt.grid()
        fig.savefig("%s_vt_z_plt.png"%(fprefix))
        plt.close()
        
        
        if (True):
            #z_idx = ((z_loc - grid_z[0])/(grid_z[1]-grid_z[0])).astype(int)
            spec_sp   = self._boltzmann_solver._spec_sp
            k_domain  = self._boltzmann_solver._v_space_domain
            c_gamma   = np.sqrt(2 * collisions.ELECTRON_CHARGE_MASS_RATIO)
            mw        = self._boltzmann_solver._maxwellian
            vth       = self._boltzmann_solver._vth
            ev_domain = ((k_domain[0] * vth /c_gamma)**2 , min(40.0, (k_domain[1] * vth /c_gamma)**2))
            ev_pts    = np.linspace(ev_domain[0] + 1e-3, 0.9 * ev_domain[1], (spec_sp._p+1)*2)

            num_p  = spec_sp._p +1
            num_sh = len(spec_sp._sph_harm_lm)
            num_z  = len(grid_z)
            num_vt = len(grid_vt)

            rows  = max(num_z_plots//num_sh,0)
            cols  = 3

            fig = plt.figure(figsize=(24, 14), dpi=300)
            plt.suptitle("T=%4E (s)"%(self._t_curr))
            p_vt_to_lm = self._boltzmann_solver._vt_to_lm

            fx = fx_kvt.reshape((num_z, num_p , num_vt))
            fx = np.dot(fx, p_vt_to_lm).reshape((num_z, num_p * num_sh))
            for ci in range(cols):
                plt.subplot(2, cols, ci+1)
                radial_comp = BEUtils.compute_radial_components(ev_pts, spec_sp, fx[ci,:], mw, vth, 1)
                for lm_idx, lm in enumerate(spec_sp._sph_harm_lm):
                    plt.semilogy(ev_pts, np.abs(radial_comp[lm_idx, :]),'--',label="l=%d"%(lm_idx))
                
                plt.grid()
                plt.legend()
                plt.xlabel("energy (eV)")
                plt.title("z=%d dz"%(ci))
                    
            for ci in range(cols):
                plt.subplot(2, cols, cols + ci+1)
                radial_comp = BEUtils.compute_radial_components(ev_pts, spec_sp, fx[ci-cols,:], mw, vth, 1)
                for lm_idx, lm in enumerate(spec_sp._sph_harm_lm):
                    plt.semilogy(ev_pts, np.abs(radial_comp[lm_idx, :]),'--',label="l=%d"%(lm_idx))
                
                plt.xlabel("energy (eV)")
                plt.grid()
                plt.title("z=L-%ddz"%(abs(ci+1-cols)))        
                plt.legend()
                

            fig.savefig("%s_f0f1.png"%(fprefix))
            plt.close()
        

def benchmark_cpu(f, a, warmup, n_repeat):
    
    for i in range(warmup):
        f(*a)

    start_cpu = time.perf_counter()
    for i in range(n_repeat):
        f(*a)
    end_cpu = time.perf_counter()

    return (end_cpu-start_cpu)/n_repeat

def benchmark_gpu(f, a, warmup, n_repeat):
    
    cp.cuda.Stream.null.synchronize()
    for i in range(warmup):
        f(*a)
    cp.cuda.Stream.null.synchronize()
    
    
    start_cpu = time.perf_counter()
    for i in range(n_repeat):
        f(*a)
    cp.cuda.Stream.null.synchronize()
    end_cpu = time.perf_counter()
    
    t_cpu = (end_cpu - start_cpu)/n_repeat
    t_gpu = (end_cpu - start_cpu)/n_repeat
    cp.cuda.Stream.null.synchronize()
    return t_gpu, t_cpu


    
if __name__== "__main__":
    args         = parser.parse_args()
    print("solver arguments ",args)

    COLLISOIN_NAMES=dict()
    for  col_idx, col in enumerate(args.collisions):
        COLLISOIN_NAMES[col]=col

    COLLISOIN_NAMES["g0"] = "elastic"
    COLLISOIN_NAMES["g2"] = "ionization"

    USE_GPU=bool(args.gpu)

    if USE_GPU:
        try:
            num_gpus = cp.cuda.runtime.getDeviceCount()
            assert args.device_id < num_gpus , "device_id >= num_gpus"
            cp.cuda.runtime.setDevice(args.device_id)
        except Exception as e:
            print(e)
            num_gpus=0

        print("Cupy number of devices = %d" %(num_gpus))        
        if num_gpus == 0:
            print("No GPU devices found. Reverting back to CPU only code")
            USE_GPU=False

    glow_sim_1d = GlowSim_1D(args)

    if (args.benchmark):
        n_warm_up    = 5
        n_iterations = 5 
        glow_sim_1d.init(0,args.T_END, args)
        
        if USE_GPU:
            t_all, t_all_cpu       = benchmark_gpu(glow_sim_1d.integrate, (args.T_DT,), n_warm_up, n_iterations)
            t_psolve, t_psolve_cpu = benchmark_gpu(glow_sim_1d._e_field_solver.poission_solve, (glow_sim_1d._ne, glow_sim_1d._ni, args.rf_freq, args.T_DT, args.v0_z0), n_warm_up, n_iterations)
            t_esolve, t_esolve_cpu = benchmark_gpu(glow_sim_1d._e_field_solver.compute_electric_field, (glow_sim_1d._e_potential,), n_warm_up, n_iterations)
            t_bte, t_bte_cpu       = benchmark_gpu(glow_sim_1d._boltzmann_solver.integrate, (glow_sim_1d._fx_kvt, args.T_DT,), n_warm_up, n_iterations)
            t_fluid, t_fluid_cpu   = benchmark_gpu(glow_sim_1d._fluids_solver.integrate, (glow_sim_1d._ni, args.T_DT,), n_warm_up, n_iterations)
        else:
            t_all    = benchmark_cpu(glow_sim_1d.integrate, (args.T_DT,), n_warm_up, n_iterations)
            t_psolve = benchmark_cpu(glow_sim_1d._e_field_solver.poission_solve, (glow_sim_1d._ne, glow_sim_1d._ni, args.rf_freq, args.T_DT, args.v0_z0), n_warm_up, n_iterations)
            t_esolve = benchmark_cpu(glow_sim_1d._e_field_solver.compute_electric_field, (glow_sim_1d._e_potential,), n_warm_up, n_iterations)
            t_bte    = benchmark_cpu(glow_sim_1d._boltzmann_solver.integrate, (glow_sim_1d._fx_kvt, args.T_DT,), n_warm_up, n_iterations)
            t_fluid  = benchmark_cpu(glow_sim_1d._fluids_solver.integrate, (glow_sim_1d._ni, args.T_DT,), n_warm_up, n_iterations)

        print("Total glow_sim 1d timestep = %8E"%(t_all))
        print("\t|---poisson solve        = %8E"%(t_psolve))
        print("\t|---efield compute       = %8E"%(t_esolve))
        print("\t|---boltzmann step       = %8E"%(t_bte))
        print("\t|---fluid (ion) step     = %8E"%(t_fluid))

        sys.exit(0)

    if args.restore==0:
        glow_sim_1d.init(0,args.T_END, args)
    else:
        with open('glow_sim.checkpoint', 'rb') as handle:
            glow_sim_1d = pickle.load(handle)

    nsteps_to_plot = 4
    ts_freq        = int(args.T_END / args.T_DT / nsteps_to_plot)

  
    is_valid=True
    while glow_sim_1d._t_curr < args.T_END:
        if glow_sim_1d._step_id % 1000 ==0:
            print("glow sim t = %.8E (s)" %(glow_sim_1d._t_curr))
            print("\t ne (min, max)\t = (%.8E, %.8E)" %(glow_sim_1d._xp_module.min(glow_sim_1d._ne), glow_sim_1d._xp_module.max(glow_sim_1d._ne)) )
            print("\t ni (min, max)\t = (%.8E, %.8E)" %(glow_sim_1d._xp_module.min(glow_sim_1d._ni), glow_sim_1d._xp_module.max(glow_sim_1d._ni)) )
            print("\t E  (min, max)\t = (%.8E, %.8E)" %(glow_sim_1d._xp_module.min(glow_sim_1d._e_field), glow_sim_1d._xp_module.max(glow_sim_1d._e_field)) )
            
            e_max = glow_sim_1d._xp_module.max(glow_sim_1d._xp_module.abs(glow_sim_1d._e_field))
            if glow_sim_1d._boltzmann_solver._e_field_max < e_max:
                print("Vspace CFL violation max Efield %2E and current max %2E"%(glow_sim_1d._boltzmann_solver._e_field_max, e_max))
                sys.exit(0)

            print(glow_sim_1d._ne[0])
            print(glow_sim_1d._ne[-1])

            # e_poten = cp.asnumpy(glow_sim_1d._e_potential)
            # e_field = cp.asnumpy(glow_sim_1d._e_field)
            # ne      = cp.asnumpy(glow_sim_1d._ne)
            # ni      = cp.asnumpy(glow_sim_1d._ni)
            # fx_kvt  = cp.asnumpy(glow_sim_1d._fx_kvt)

            # glow_sim_1d._e_field_solver.device_to_host()
            # glow_sim_1d._fluids_solver.device_to_host()
            # glow_sim_1d._boltzmann_solver.device_to_host()

            # z_pts   = np.linspace(glow_sim_1d._grid_z[1], glow_sim_1d._grid_z[-2], 6)
            # fprefix = ".glow_discharge_1d3v_" + "_".join(args.collisions) + "_v0_z0" + str(args.v0_z0) + "_E" + str(args.E_field) + "_poly_" + str(args.radial_poly)+ "_sp_"+ str(args.spline_order) + "_nr" + str(args.NUM_P_RADIAL)+ "_lmax" + str(args.l_max) + "_qpn" + str(args.spline_q_pts_per_knot) + "_ev"+ "%2E"%(args.electron_volt) + "_bscale" + str(args.basis_scale) +"_dt"+"%.4E"%(args.T_DT) + "_T"+ "%.2E"%(glow_sim_1d._t_curr) 
            # glow_sim_1d.plot(e_poten, e_field, ne, ni, fx_kvt, z_pts, fprefix,plot_pannel=False)


            # glow_sim_1d._e_field_solver.host_to_device()
            # glow_sim_1d._fluids_solver.host_to_device()
            # glow_sim_1d._boltzmann_solver.host_to_device()


            if len(cp.where(glow_sim_1d._ne<0)[0]):
                print(cp.where(glow_sim_1d._ne<0))
                #is_valid=False
            cp.cuda.Stream.null.synchronize()

        # if glow_sim_1d._step_id % 10000 ==0:
        #     with open('glow_sim.checkpoint', 'wb') as handle:
        #         pickle.dump(glow_sim_1d, handle, protocol=pickle.HIGHEST_PROTOCOL)


            # print("\t ne (min, max)\t = (%.8E, %.8E)" %(xp.min(ne), xp.max(ne)) )
            # print("\t ni (min, max)\t = (%.8E, %.8E)" %(xp.min(ni), xp.max(ni)) )
            # print("\t E  (min, max)\t = (%.8E, %.8E)" %(xp.min(e_field), xp.max(e_field)) )
        
        glow_sim_1d._cycle_avg_e_poten += glow_sim_1d._e_potential
        glow_sim_1d._cycle_avg_e_field += glow_sim_1d._e_field
        glow_sim_1d._cycle_avg_ne      += glow_sim_1d._ne
        glow_sim_1d._cycle_avg_ni      += glow_sim_1d._ni
        glow_sim_1d._cycle_avg_fx_kvt  += glow_sim_1d._fx_kvt
        
        glow_sim_1d.integrate(args.T_DT)

        glow_sim_1d._cycle_avg_e_poten += glow_sim_1d._e_potential
        glow_sim_1d._cycle_avg_e_field += glow_sim_1d._e_field
        glow_sim_1d._cycle_avg_ne      += glow_sim_1d._ne
        glow_sim_1d._cycle_avg_ni      += glow_sim_1d._ni
        glow_sim_1d._cycle_avg_fx_kvt  += glow_sim_1d._fx_kvt
        
        if (is_valid==False):
            break


    if glow_sim_1d._xp_module == cp:
        e_poten = (0.5 * args.T_DT/glow_sim_1d._t_curr) * cp.asnumpy(glow_sim_1d._cycle_avg_e_poten)
        e_field = (0.5 * args.T_DT/glow_sim_1d._t_curr) * cp.asnumpy(glow_sim_1d._cycle_avg_e_field)
        ne      = (0.5 * args.T_DT/glow_sim_1d._t_curr) * cp.asnumpy(glow_sim_1d._cycle_avg_ne)
        ni      = (0.5 * args.T_DT/glow_sim_1d._t_curr) * cp.asnumpy(glow_sim_1d._cycle_avg_ni)
        fx_kvt  = (0.5 * args.T_DT/glow_sim_1d._t_curr) * cp.asnumpy(glow_sim_1d._cycle_avg_fx_kvt)
        
        glow_sim_1d._e_field_solver.device_to_host()
        glow_sim_1d._fluids_solver.device_to_host()
        glow_sim_1d._boltzmann_solver.device_to_host()

        cp.cuda.Stream.null.synchronize()
    else:
        e_poten = (0.5 * args.T_DT/glow_sim_1d._t_curr) * glow_sim_1d._cycle_avg_e_poten
        e_field = (0.5 * args.T_DT/glow_sim_1d._t_curr) * glow_sim_1d._cycle_avg_e_field
        ne      = (0.5 * args.T_DT/glow_sim_1d._t_curr) * glow_sim_1d._cycle_avg_ne
        ni      = (0.5 * args.T_DT/glow_sim_1d._t_curr) * glow_sim_1d._cycle_avg_ni
        fx_kvt  = (0.5 * args.T_DT/glow_sim_1d._t_curr) * glow_sim_1d._cycle_avg_fx_kvt


    z_pts   = np.linspace(glow_sim_1d._grid_z[1], glow_sim_1d._grid_z[-2], 6)
    fprefix = "glow_discharge_1d3v_" + "_".join(args.collisions) + "_v0_z0%.2E"%(args.v0_z0) + "_poly_" + str(args.radial_poly)+ "_sp_"+ str(args.spline_order) +"_nz"+str(args.num_z)+ "_nr" + str(args.NUM_P_RADIAL)+ "_lmax" + str(args.l_max) + "_qpn" + str(args.spline_q_pts_per_knot) + "_qvt"+str(args.quad_theta) + "_ev"+ "%.2E"%(args.electron_volt) + "_bscale" + str(args.basis_scale) + "_ne_fac%.2E"%(args.ne_fac) + "_vx_max_%.2E"%(args.vx_max) +"_dt"+"%.2E"%(args.T_DT) + "_T"+ "%.2E"%(args.T_END) 
    glow_sim_1d.plot(e_poten, e_field, ne, ni, fx_kvt, z_pts, fprefix)


