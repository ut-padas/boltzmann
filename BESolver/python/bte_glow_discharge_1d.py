"""
@brief : Boltzmann solver in 1d-space glow discharge problem
"""
from cProfile import label
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
from time import perf_counter as time, sleep
# set the threading layer before any parallel target compilation
nb.config.THREADING_LAYER = 'threadsafe'


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

parser.add_argument("-Lz", "--Lz" , help="1d length of the torch in (m)", type=float, default=0.0254)
parser.add_argument("-num_z", "--num_z" , help="number of points in z-direction", type=int, default=50)
parser.add_argument("-v0_z0", "--v0_z0" , help="voltage at z=0 (v)", type=float, default=100.0)
parser.add_argument("-rf_freq", "--rf_freq" , help="voltage oscillation freq (Hz)", type=float, default=13.56e6)

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

def grad1_p_11_mat(grid_z):
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
    data = np.append(data, np.array([1,-1]) * idz)

    return scipy.sparse.csr_matrix((data, (rows, cols)), shape=(n,n))

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

def grad1_m_11_mat(grid_z):

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
    def __init__(self, grid_z) -> None:
        self._grid_z = grid_z
        self._epsilon_0 = 55.26349406e6 # e^2 ev^-1 m^-1

    def fd1_laplace_1d_mat(self):
        """
        Assembles the laplace operator with
        1st order finite difference approximation. 
        """
        grid_z = self._grid_z
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
        # rows = np.append(rows, np.array([0,0]))
        # cols = np.append(cols, np.array([0,1]))
        # data = np.append(data, np.array([-2 *idx2, 1.0 * idx2]))

        # # right boundary. 
        # rows = np.append(rows, np.array([n-1, n-1]))
        # cols = np.append(cols, np.array([n-1, n-2]))
        # data = np.append(data, np.array([-2*idx2, 1.0 * idx2]))

        # # left boundary 
        rows = np.append(rows, np.array([0]))
        cols = np.append(cols, np.array([0]))
        data = np.append(data, np.array([1.0 * idx2]))

        # # right boundary. 
        rows = np.append(rows, np.array([n-1]))
        cols = np.append(cols, np.array([n-1]))
        data = np.append(data, np.array([1.0 * idx2]))


        Lmat = scipy.sparse.csr_array((data, (rows, cols)), shape=(n,n))
        #print(Lmat.toarray() / idx2)
        return Lmat

    def setup(self,args):
        self._inverse_laplacian = scipy.sparse.csr_matrix(BEUtils.choloskey_inv(-1.0 * self.fd1_laplace_1d_mat().toarray()))
        
    def poission_solve(self, ni, ne, freq_f, t_c, v0):
        f_rhs = (ni-ne) * (1.0 /self._epsilon_0)

        grid_z = self._grid_z
        n   = len(grid_z)
        dx  = grid_z[1] - grid_z[0]
        idx2 = 1/dx**2
        
        f_rhs[0]  = -idx2 * v0 * np.sin(2 * np.pi * freq_f * t_c)
        f_rhs[-1] = 0.0

        u     = self._inverse_laplacian.dot(f_rhs)
        
        # print("ne", ne)
        # print("ni", ni)
        #print("rhs", f_rhs, v0 * np.sin(2 * np.pi * freq_f * t_c), u)
        #print("u", u)

        return u

    def compute_electric_field(self, u):
        grid_z = self._grid_z
        n      = len(grid_z)
        idz    = 1.0 / (grid_z[1]-grid_z[0])

        ef = np.zeros(len(grid_z))
        
        ef[1:n-1] = 0.5 * (u[2:n] -u[0:n-2]) * idz
        ef[0]     = (u[1]-u[0])   * idz
        ef[-1]    = (u[-1]-u[-2]) * idz

        return -ef
        

class Fluids_1D():
    def __init__(self, grid_z) -> None:
        self._grid_z = grid_z
        self._idz    = 1.0/(self._grid_z[1] - self._grid_z[0])

        num_z    = len(self._grid_z)
        self._ne = np.zeros(num_z)
        self._nn = np.zeros(num_z)

        self._reaction_k = np.zeros(num_z)
        self._mu         = 0.0
        self._D          = 0.0
        self._v_field    = np.zeros(num_z)
        self._rhs_v0     = np.zeros(num_z)

        self._rhs_v1     = np.zeros(num_z)
        self._rhs_v2     = np.zeros(num_z)


        
    def setup(self, args):
        self._t_curr = 0.0
        pass

    def apply_bc(self, t, y):
        D      = self._D
        mu     = self._mu
        v      = self._v_field
        idz    = self._idz
        
        # E_0 dot e_k > 0 set incoming flux to be zero
        E_0 = max(  -(v[1]-v[0])  * idz, 0.0)
        
        # E_L dot e_k < 0 set incoming flux to be zero
        E_L = max( (v[-1]-v[-2]) * idz, 0.0)

        y[0]  = y[1]  * (D * idz - mu * E_0 * 0.5 ) / (D * idz + mu * E_0 * 0.5)
        y[-1] = y[-2] * (D * idz - mu * E_L * 0.5)  / (D * idz + mu * E_L * 0.5)
        
        y[y<0]=0.0

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

        y_rhs  = self._rhs_v0
        j_p    = self._rhs_v1
        j_m    = self._rhs_v2
        
        j_p[1:n-1] = mu * (-( v[2:n] - v[1:n-1]) * idz) * (y[2:n] + y[1:n-1])   * 0.5 - D * idz * (y[2:n]-y[1:n-1])
        j_m[1:n-1] = mu * (-(v[1:n-1]- v[0:n-2]) * idz) * (y[1:n-1] + y[0:n-2]) * 0.5 - D * idz * (y[1:n-1]-y[0:n-2])

        y_rhs[1:n-1] = -idz * (j_p[1:n-1]-j_m[1:n-1]) + r_rate[1:n-1] * ne[1:n-1] * nn[1:n-1]
        
        # y_rhs[1:n-1] = -idz * ( 0.5 * idz * mu * (v[1:n-1] -v[2:n]) * (y[2:n] + y[1:n-1])  -  0.5 * idz * mu * (v[0:n-2] - v[1:n-1]) * (y[1:n-1] + y[0:n-2]) ) + D * idz**2 * (y[2:n] -2 * y[1:n-1] + y[0:n-2]) + r_rate[1:n-1] * ne[1:n-1] * nn[1:n-1]
        
        return y_rhs
    
    def integrate(self, u, dt):
        ut =  u + dt * self.rhs(self._t_curr, u)
        self._t_curr+=dt
        ut =  self.apply_bc(self._t_curr, ut)
        return ut
        
class Boltzmann_1D():
    def __init__(self,grid_z, grid_vt) -> None:
        self._grid_z  = grid_z
        self._grid_vt = grid_vt
        self._E_field = 0.0

    def setup(self, args):
        """
        assembles the velocity space operators
        i.e., v-space advection and v-space collision operator
        """
        self._t_curr = 0.0

        vth                            = collisions.electron_thermal_velocity(args.electron_volt*args.basis_scale * collisions.TEMP_K_1EV)
        maxwellian                     = BEUtils.get_maxwellian_3d(vth,collisions.MAXWELLIAN_N)
        c_gamma                        = np.sqrt(2*collisions.ELECTRON_CHARGE_MASS_RATIO)
        if "g2Smooth" in args.collisions or "g2" in args.collisions or "g2step" in args.collisions or "g2Regul" in args.collisions:
            sig_pts = np.array([np.sqrt(15.76) * c_gamma/vth])
        else:
            sig_pts = None 
        
        ev_range = ((0*vth/c_gamma)**2, (6*vth/c_gamma)**2)
        k_domain = (np.sqrt(ev_range[0]) * c_gamma / vth, np.sqrt(ev_range[1]) * c_gamma / vth)

        self._v_space_domain = (np.sqrt(ev_range[0]) * c_gamma / vth, np.sqrt(ev_range[1]) * c_gamma / vth)
        
        print("target ev range : (%.4E, %.4E) ----> knots domain : (%.4E, %.4E)" %(ev_range[0], ev_range[1], k_domain[0],k_domain[1]))
        if(sig_pts is not None):
            print("singularity pts : ", sig_pts, "v/vth and" , (sig_pts * vth/c_gamma)**2,"eV")

        params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER = args.NUM_P_RADIAL
        basis.BSPLINE_BASIS_ORDER=args.spline_order
        basis.XLBSPLINE_NUM_Q_PTS_PER_KNOT=args.spline_q_pts_per_knot
        if (args.radial_poly == "bspline"):
            bb     = basis.BSpline(k_domain, args.spline_order, params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER+1, sig_pts=sig_pts, knots_vec=None)
            params.BEVelocitySpace.NUM_Q_VR = bb._num_knot_intervals * args.spline_q_pts_per_knot
            params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER = bb._num_p
            spec_sp               = sp.SpectralExpansionSpherical(params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER-1, bb,params.BEVelocitySpace.SPH_HARM_LM)
            spec_sp._num_q_radial = params.BEVelocitySpace.NUM_Q_VR
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
        Emat       = spec_sp.compute_advection_matix()

        self._spec_sp = spec_sp
        self._col_op  = collision_op

        self._Cmat    = Cmat
        self._Emat    = Emat 

        self._lm_to_vt = lm_modes_to_vt_pts_op(self._grid_z, self._spec_sp, self._grid_vt)
        self._vt_to_lm = vt_pts_to_lm_modes_op(self._grid_z, self._spec_sp, self._grid_vt)

        self._Dx_p     = grad1_p_11_mat(self._grid_z)
        self._Dx_m     = grad1_m_11_mat(self._grid_z)
        
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
                self._Wpk_mat[i,j] = np.dot(gw, vqr[i,:] * vqr[j,:] * gx**3 * vth_curr)
        

        self._mass_op      = BEUtils.mass_op(spec_sp, None, 8, 2, 1.0)
        self._temp_op      = BEUtils.temp_op(spec_sp, None, 8, 2, 1.0)
        
        self._idx_p_upwind =  np.array([m + np.array(range(num_p)) * len(self._grid_vt) for m in np.where(self._grid_vt <= 0.5 * np.pi)[0]]).reshape(-1) 
        self._idx_m_upwind =  np.array([m + np.array(range(num_p)) * len(self._grid_vt) for m in np.where(self._grid_vt >  0.5 * np.pi)[0]]).reshape(-1) 

        self._cos_vt  = np.array([np.cos(self._grid_vt) for m in range(num_p)]).reshape(-1)
        self._Dzy_kvt = np.zeros((len(self._grid_z), num_p * (len(self._grid_vt))))
        
    def rhs(self, t, y):
        spec_sp = self._spec_sp
        num_p   = spec_sp._p + 1
        num_sh  = len(spec_sp._sph_harm_lm)

        num_vt  = len(self._grid_vt)
        num_z   = len(self._grid_z)

        yy   = y.reshape((num_z, num_p, num_vt))
        yy  = np.dot(yy,self._vt_to_lm).reshape((num_z, num_p*num_sh))

        CpE_klm    = np.matmul(self._Cmat, np.transpose(yy))
        CpE_klm   += np.matmul(self._Emat, np.transpose(yy)) * (self._E_field * collisions.ELECTRON_CHARGE_MASS_RATIO/self._vth)
        CpE_klm    = np.transpose(CpE_klm)

        # CpE_klm   = np.dot(y,self._vt_to_lm).reshape((num_z, num_p*num_sh))
        # for z_id in range(num_z):
        #     FOp              = self._Cmat + self._Emat * (self._E_field[z_id] * collisions.ELECTRON_CHARGE_MASS_RATIO/self._vth)
        #     CpE_klm[z_id,:]  = np.matmul(FOp, CpE_klm[z_id , :])

        CpE_kvt = np.dot(CpE_klm.reshape(num_z, num_p , num_sh), self._lm_to_vt).reshape((num_z, num_p * num_vt))
        CpE_kvt = np.transpose(CpE_kvt).reshape((num_p, num_vt, num_z)).reshape((num_p, num_vt * num_z))
        CpE_kvt = np.matmul(self._inv_mmat_radial, CpE_kvt)
        CpE_kvt = CpE_kvt.reshape((num_p , num_vt, num_z)).reshape((num_p * num_vt, num_z)).transpose()

        Dzy_kvt = self._Dzy_kvt
        yy      = y.reshape((num_z, num_p *num_vt))

        Dzy_kvt[:,self._idx_p_upwind] = self._Dx_p.dot(yy[:,self._idx_p_upwind]) * self._cos_vt[self._idx_p_upwind]
        Dzy_kvt[:,self._idx_m_upwind] = self._Dx_m.dot(yy[:,self._idx_m_upwind]) * self._cos_vt[self._idx_m_upwind]

        # Dzy_kvt = np.zeros((num_z, num_p, num_vt))
        # yy      = y.reshape((num_z, num_p, num_vt))
        # dz      = (self._grid_z[1]-self._grid_z[0])
        # for vt_idx, vt in enumerate(self._grid_vt):
        #     if vt <= np.pi/2:
        #         for k in range(num_p):
        #             Dzy_kvt[:, k, vt_idx] = grad1_p_11(yy[:,k,vt_idx],dz) * np.cos(vt)
        #     else:
        #         for k in range(num_p):
        #             Dzy_kvt[:, k, vt_idx] = grad1_m_11(yy[:,k,vt_idx],dz) * np.cos(vt)
        # Dzy_kvt = Dzy_kvt.reshape((num_z, num_p * num_vt))

        Dzy_kvt = Dzy_kvt.transpose().reshape((num_p, num_vt, num_z)).reshape((num_p, num_vt * num_z))
        Dzy_kvt = np.matmul(self._inv_mmat_radial,np.matmul(self._Wpk_mat, Dzy_kvt))
        Dzy_kvt = Dzy_kvt.reshape((num_p , num_vt, num_z)).reshape((num_p * num_vt, num_z)).transpose()

        y_rhs   = (CpE_kvt - Dzy_kvt).reshape(num_z * num_p * num_vt) 
        return y_rhs

    def apply_bc(self, t, y):
        spec_sp = self._spec_sp
        num_p   = spec_sp._p + 1
        num_sh  = len(spec_sp._sph_harm_lm)

        num_vt  = len(self._grid_vt)
        num_z   = len(self._grid_z)

        y = y.reshape((num_z, num_p, num_vt))

        y[0 , :, np.where(self._grid_vt<=np.pi/2)] = 0.0
        y[-1, :, np.where(self._grid_vt>np.pi/2)]  = 0.0

        return y.reshape(num_z * num_p * num_vt)

    def integrate(self, u, dt):
        u1 = u + dt * self.rhs(self._t_curr, u)
        u1 = self.apply_bc(self._t_curr, u1)
        self._t_curr += dt
        return u1

    def compute_ne(self, f_kvt):
        spec_sp = self._spec_sp
        num_p   = spec_sp._p + 1
        num_sh  = len(spec_sp._sph_harm_lm)

        num_vt  = len(self._grid_vt)
        num_z   = len(self._grid_z)
        yy      = np.dot(f_kvt.reshape((num_z, num_p , num_vt)),self._vt_to_lm).reshape((num_z, num_p*num_sh))
        ne      = np.dot(self._mass_op, yy.transpose()).reshape(num_z) * self._vth**3 * self._maxwellian(0)

        return ne

    def compute_avg_energy(self, f_kvt):
        spec_sp = self._spec_sp
        num_p   = spec_sp._p + 1
        num_sh  = len(spec_sp._sph_harm_lm)

        temp_op = self._temp_op

        num_vt  = len(self._grid_vt)
        num_z   = len(self._grid_z)
        yy      = np.dot(f_kvt.reshape((num_z, num_p , num_vt)),self._vt_to_lm).reshape((num_z, num_p*num_sh))
        Te      = np.dot(self._temp_op, yy.transpose()).reshape(num_z) * self._vth**5 * self._maxwellian(0) * 0.5 * collisions.MASS_ELECTRON * (2/(3*scipy.constants.Boltzmann)) / self.compute_ne(f_kvt) / collisions.TEMP_K_1EV

        return Te
        


    
    def compute_reaction_rates(self, rates_op_dict, f_kvt):
        spec_sp = self._spec_sp
        num_p   = spec_sp._p + 1
        num_sh  = len(spec_sp._sph_harm_lm)

        num_vt  = len(self._grid_vt)
        num_z   = len(self._grid_z)
        
        yy      = np.dot(f_kvt.reshape((num_z, num_p , num_vt)),self._vt_to_lm).reshape((num_z, num_p*num_sh))
        m0      = np.dot(self._mass_op, yy.transpose()).reshape(num_z) * self._vth**3 * self._maxwellian(0)

        ki  =  dict()
        for (reaction, rates_op) in rates_op_dict.items():
            ki[reaction] = np.dot(rates_op, yy.transpose()).reshape(num_z) #/ m0
            
        return ki

class GlowSim_1D():
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

        vratio    = np.sqrt(1.0/args.basis_scale)
        hv        = lambda v,vt,vp : np.exp(-((v/vratio)**2)) / vratio**3

        h0      = BEUtils.function_to_basis(spec_sp, hv, self._boltzmann_solver._maxwellian, None, None, None, Minv=self._boltzmann_solver._inv_mmat)
        mass_op = self._boltzmann_solver._mass_op
        m0      = np.dot(mass_op, h0) * self._boltzmann_solver._vth**3 * self._boltzmann_solver._maxwellian(0)

        self._fx_kvt = create_vec_lm(grid_z, spec_sp)
        for i in range(len(grid_z)):
            self._fx_kvt[i,:] = h0

        # print(self._boltzmann_solver._lm_to_vt.shape)
        # print(self._boltzmann_solver._lm_to_vt)

        # print(self._boltzmann_solver._vt_to_lm.shape)
        # print(self._boltzmann_solver._vt_to_lm)
        # print(np.sum(self._boltzmann_solver._vt_to_lm, axis=0))

        self._fx_kvt = np.dot(self._fx_kvt.reshape((num_z, num_p, num_sh)) , self._boltzmann_solver._lm_to_vt).reshape(num_z * num_p * num_vt)
        #self._fx_kvt = self._boltzmann_solver.apply_bc(self._t_begin, self._fx_kvt)

        self._ne     = self._boltzmann_solver.compute_ne(self._fx_kvt)
        self._ni     = np.array(self._ne) 
        # test initial data
        # self._ni = np.max(self._ne) * np.exp(-(self._grid_z - 0.5 * args.Lz)**2/1e-6 ) #np.array(self._ne)
        # self._ni[self._grid_z < 0.5 * args.Lz]=0.0

        self._fluids_solver._nn   = np.ones_like(grid_z) * collisions.AR_NEUTRAL_N
        self._fluids_solver._ne   = self._ne
        self._fluids_solver._mu = 4.65e21 / collisions.AR_NEUTRAL_N
        self._fluids_solver._D  = 2.07e20 / collisions.AR_NEUTRAL_N

        all_reaction_rates        = self._boltzmann_solver.compute_reaction_rates(self._boltzmann_solver._reaction_op, self._fx_kvt)
        if "g2" in all_reaction_rates:
            self._fluids_solver._reaction_k   = all_reaction_rates["g2"] 
        else:
            self._fluids_solver._reaction_k   = np.zeros(num_z)
        
        self._boltzmann_solver._t_curr  = self._t_begin
        self._fluids_solver._t_curr     = self._t_begin
        
        self._e_field     = np.ones_like(self._ni) * args.E_field
        self._e_potential = np.ones_like(self._ni) * args.E_field

        return

    def integrate(self, dt):
        # e_field solve
        self._e_potential = self._e_field_solver.poission_solve(self._ni, self._ne, self._rf_freq, self._t_curr + dt, self._v0_z0)
        self._e_field     = self._e_field_solver.compute_electric_field(self._e_potential)
        
        # # E field to Boltzmann solver
        self._boltzmann_solver._E_field   = self._e_field
        self._fx_kvt                      = self._boltzmann_solver.integrate(self._fx_kvt, dt) 
        self._ne                          = self._boltzmann_solver.compute_ne(self._fx_kvt)
        self._ne[self._ne < 0]            = 0.0
        

        # set info from Boltzmann to fluids, 
        self._fluids_solver._ne           = self._ne
        self._fluids_solver._v_field      = self._e_potential

        all_reaction_rates                = self._boltzmann_solver.compute_reaction_rates(self._boltzmann_solver._reaction_op, self._fx_kvt)
        if "g2" in all_reaction_rates:
            self._fluids_solver._reaction_k   = all_reaction_rates["g2"]
        
        
        self._ni                  = self._fluids_solver.integrate(self._ni, dt)
        
        self._t_curr  += dt
        self._step_id += 1

        return 

    def plot(self, e_potential, e_field, ne, ni, fx_kvt, z_loc, ts,  fprefix):
        
        num_z_plots = len(z_loc)
        grid_z  = self._grid_z
        grid_vt = self._grid_vt
        z_idx = ((z_loc - grid_z[0])/(grid_z[1]-grid_z[0])).astype(int)


        rows  = 2 #num_z_plots//2 + 1
        cols  = 3

        fig = plt.figure(figsize=(12, 8), dpi=300)

        plt.subplot(rows, cols, 1)
        for i in range(len(ts)):
            color = next(plt.gca()._get_lines.prop_cycler)['color']
            plt.plot(grid_z, ni[i,:], label="t=%.2E (s)"%(ts[i]), color=color)

        plt.xlabel("z (m)")
        plt.ylabel("# density (1/m^3)")
        plt.grid()
        plt.title("ni")
        plt.legend()

        plt.subplot(rows, cols, 2)
        for i in range(len(ts)):
            color = next(plt.gca()._get_lines.prop_cycler)['color']
            plt.plot(grid_z, ne[i,:], label="t=%.2f (s)"%(ts[i]), color=color)

        plt.xlabel("z (m)")
        plt.ylabel("# density (1/m^3)")
        plt.grid()
        plt.title("ne")
        

        plt.subplot(rows, cols, 3)
        for i in range(len(ts)):
            color = next(plt.gca()._get_lines.prop_cycler)['color']
            plt.plot(grid_z, e_field[i,:], label="t=%.2f (s)"%(ts[i]), color=color)

        plt.xlabel("z (m)")
        plt.ylabel("V/m")
        plt.grid()
        plt.title("E")
        
        c_rates=list()
        for i in range(len(ts)):
            c_rates.append(self._boltzmann_solver.compute_reaction_rates(self._boltzmann_solver._reaction_op, fx_kvt[i,:]))

        plt.subplot(rows, cols, 4)
        for i in [-1]:#range(len(ts)):
            for key, rr in c_rates[i].items():
                color = next(plt.gca()._get_lines.prop_cycler)['color']
                plt.semilogy(grid_z, rr, label="%s"%(key), color=color)

        plt.legend()
        plt.xlabel("z (m)")
        plt.ylabel("rate coefficient m^3/s")
        plt.grid()

        temp_list=list()
        for i in range(len(ts)):
            temp_list.append(self._boltzmann_solver.compute_avg_energy(fx_kvt[i,:]))

        plt.subplot(rows, cols, 5)
        for i in range(len(ts)):
            color = next(plt.gca()._get_lines.prop_cycler)['color']
            plt.loglog(grid_z, temp_list[i], label="t=%.2f (s)"%(ts[i]), color=color)

        plt.xlabel("z (m)")
        plt.ylabel("mean energy (eV)")
        plt.grid()

        plt.subplot(rows, cols, 6)
        for i in range(len(ts)):
            color = next(plt.gca()._get_lines.prop_cycler)['color']
            plt.plot(grid_z, e_potential[i], label="t=%.2f (s)"%(ts[i]), color=color)

        plt.xlabel("z (m)")
        plt.ylabel("Voltage(V)")
        plt.grid()

        plt.tight_layout()
        fig.savefig("%s_qoi.png"%(fprefix))
        plt.close()
        
        
        if (num_z_plots>0):

            rows  = num_z_plots//2 + 1
            cols  = 3

            fig = plt.figure(figsize=(5 * cols, 4 * rows), dpi=300)


            spec_sp   = self._boltzmann_solver._spec_sp
            k_domain  = self._boltzmann_solver._v_space_domain
            c_gamma   = np.sqrt(2 * collisions.ELECTRON_CHARGE_MASS_RATIO)
            mw        = self._boltzmann_solver._maxwellian
            vth       = self._boltzmann_solver._vth
            ev_domain = ((k_domain[0] * vth /c_gamma)**2 , (k_domain[1] * vth /c_gamma)**2)
            ev_pts    = np.linspace(ev_domain[0], 0.9 * ev_domain[1], (spec_sp._p+1)*2)

            num_p  = spec_sp._p +1
            num_sh = len(spec_sp._sph_harm_lm)
            num_z  = len(grid_z)
            num_vt = len(grid_vt)

            p_vt_to_lm = self._boltzmann_solver._vt_to_lm

            plt_offset = 1
            for w in range(num_z_plots//2):
                for ii_idx, ii in enumerate([2*w, 2*w +1]):
                    if (ii < num_z_plots):
                        radial_comp = list()
                        for i in range(len(ts)):
                            fx = fx_kvt[i].reshape((num_z, num_p , num_vt))
                            # print("a", fx[0,:, np.where(grid_vt<=np.pi/2)])
                            # print("b", fx[0,:, np.where(grid_vt>np.pi/2)])

                            # print("c", fx[-1,:, np.where(grid_vt<=np.pi/2)])
                            # print("d", fx[-1,:, np.where(grid_vt>np.pi/2)])

                            fx = np.dot(fx, p_vt_to_lm).reshape((num_z, num_p * num_sh))

                            # print("a l=0", fx[0,0::num_sh])
                            # print("a l=1", fx[0,1::num_sh])

                            # print("c l=0", fx[-1,0::num_sh])
                            # print("c l=1", fx[-1,1::num_sh])
                            radial_comp.append(BEUtils.compute_radial_components(ev_pts, spec_sp, fx[z_idx[ii],:], mw, vth, 1))

                            
                        for lm_idx, lm in enumerate(spec_sp._sph_harm_lm):
                            plt.subplot(rows, cols, plt_offset + ii_idx * num_sh + lm_idx)
                            for i in range(len(ts)):
                                color = next(plt.gca()._get_lines.prop_cycler)['color']
                                plt.semilogy(ev_pts, np.abs(radial_comp[i][lm_idx, :]),color=color)
                            
                            plt.xlabel("energy (eV)")
                            plt.ylabel("f_%d"%(lm[0]))
                            plt.grid()
                            plt.title("z=%.2E"%(grid_z[z_idx[ii]]))               
                
                plt_offset +=4
                        



            plt.tight_layout()
            fig.savefig("%s_f0f1.png"%(fprefix))
            plt.close()
        
        




if __name__== "__main__":
    args         = parser.parse_args()
    print("solver arguments ",args)

    glow_sim_1d = GlowSim_1D(args)
    glow_sim_1d.init(0,args.T_END, args)

    nsteps_to_plot = 4
    ts_freq        = int(args.T_END / args.T_DT / nsteps_to_plot)

    e_poten = list()
    e_field = list()
    ni      = list()
    ne      = list()
    fx_kvt  = list()
    ts      = list()
    
    while glow_sim_1d._t_curr < args.T_END:
        if glow_sim_1d._step_id % 100 ==0:
            np.set_printoptions(precision=2)
            print("glow sim t = %.8E (s)" %(glow_sim_1d._t_curr))
            print("\t ne (min, max)\t", np.min(glow_sim_1d._ne), np.max(glow_sim_1d._ne))
            print("\t ni (min, max)\t", np.min(glow_sim_1d._ni), np.max(glow_sim_1d._ni))
            print("\t E  (min, max)\t", np.min(glow_sim_1d._e_field), np.max(glow_sim_1d._e_field))
            np.set_printoptions(precision=16)


        if glow_sim_1d._step_id % ts_freq ==0:
            
            ts.append(glow_sim_1d._t_curr)
            e_poten.append(glow_sim_1d._e_potential)
            e_field.append(glow_sim_1d._e_field)
            ni.append(glow_sim_1d._ni)
            ne.append(glow_sim_1d._ne)
            fx_kvt.append(glow_sim_1d._fx_kvt)

        glow_sim_1d.integrate(args.T_DT)

    e_poten = np.array(e_poten)
    e_field = np.array(e_field)
    ni      = np.array(ni)
    ne      = np.array(ne)
    fx_kvt  = np.array(fx_kvt)
    ts      = np.array(ts)
    
    z_pts   = np.linspace(glow_sim_1d._grid_z[1], glow_sim_1d._grid_z[-2], 6)

    fprefix = "glow_discharge_1d3v_" + "_".join(args.collisions) + "_E" + str(args.E_field) + "_poly_" + str(args.radial_poly)+ "_sp_"+ str(args.spline_order) + "_nr" + str(args.NUM_P_RADIAL)+"_qpn_" + str(args.spline_q_pts_per_knot) + "_ev"+ "%2E"%(args.electron_volt) + "_bscale" + str(args.basis_scale) +"_dt"+"%.4E"%(args.T_DT) + "_T"+ "%.2E"%(args.T_END)
    glow_sim_1d.plot(e_poten, e_field, ne, ni, fx_kvt, z_pts, ts, fprefix)

