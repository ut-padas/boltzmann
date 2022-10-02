"""
@brief : Boltzmann solver in 1d-space glow discharge problem
"""
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

collisions.MAXWELLIAN_N=1
collisions.AR_NEUTRAL_N=3.22e22
collisions.AR_IONIZED_N=collisions.AR_NEUTRAL_N 
parser = argparse.ArgumentParser()

parser.add_argument("-Nr", "--NUM_P_RADIAL"                   , help="Number of polynomials in radial direction", type=int, default=64)
parser.add_argument("-T", "--T_END"                           , help="Simulation time", type=float, default=1e-4)
parser.add_argument("-dt", "--T_DT"                           , help="Simulation time step size ", type=float, default=1e-12)
parser.add_argument("-ts_atol", "--ts_atol"                   , help="absolute ts tol", type=float, default=1e-20)
parser.add_argument("-ts_rtol", "--ts_rtol"                   , help="relative ts tol", type=float, default=1e-8)
parser.add_argument("-l_max",   "--l_max"                     , help="max polar modes in SH expansion", type=int, default=1)

parser.add_argument("-c", "--collisions"     , help="collisions included (g0, g0Const, g0NoLoss, g2, g2Const)",nargs='+', type=str, default=["g0","g2"])
parser.add_argument("-ev", "--electron_volt" , help="initial electron volt", type=float, default=2)
parser.add_argument("-bscale", "--basis_scale"                , help="basis electron volt", type=float, default=1.0)

parser.add_argument("-q_vr", "--quad_radial"                  , help="quadrature in r"        , type=int, default=200)
parser.add_argument("-q_vt", "--quad_theta"                   , help="quadrature in polar"    , type=int, default=4)
parser.add_argument("-q_vp", "--quad_phi"                     , help="quadrature in azimuthal", type=int, default=4)
parser.add_argument("-q_st", "--quad_s_theta"                 , help="quadrature in scattering polar"    , type=int, default=4)
parser.add_argument("-q_sp", "--quad_s_phi"                   , help="quadrature in scattering azimuthal", type=int, default=4)
parser.add_argument("-radial_poly", "--radial_poly"           , help="radial basis", type=str, default="bspline")
parser.add_argument("-sp_order", "--spline_order"             , help="b-spline order", type=int, default=2)
parser.add_argument("-spline_qpts", "--spline_q_pts_per_knot" , help="q points per knots", type=int, default=4)

parser.add_argument("-Lz", "--Lz" , help="1d length of the torch in (m)", type=float, default=0.0254)
parser.add_argument("-num_z", "--num_z" , help="number of points in z-direction", type=float, default=100)
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

        # left boundary 
        rows = np.append(rows, np.array([0,0]))
        cols = np.append(cols, np.array([0,1]))
        data = np.append(data, np.array([-2 *idx2, 1.0 * idx2]))

        # right boundary. 
        rows = np.append(rows, np.array([n-1, n-1]))
        cols = np.append(cols, np.array([n-1, n-2]))
        data = np.append(data, np.array([-2*idx2, 1.0 * idx2]))

        Lmat = scipy.sparse.csr_array((data, (rows, cols)), shape=(n,n)).toarray()
        return Lmat

    def setup(self,args):
        self._inverse_laplacian = BEUtils.choloskey_inv(-1.0 * self.fd1_laplace_1d_mat())

    def electric_field_solve(self, ni, ne, freq_f, t_c, v0):
        grid_z = self._grid_z
        dz    = (grid_z[1]-grid_z[0])

        f_rhs = grad1_p_11(ni-ne, dz) * scipy.constants.e / scipy.constants.epsilon_0

        u    = np.matmul(self._inverse_laplacian, f_rhs)
        # bc on u Poisson solve 
        u[0]  = v0 * np.sin(2*np.pi* freq_f * t_c)
        u[-1] = 0.0
        
        e_field = -grad1_p_11(u, dz)
        return e_field, u

class Fluids_1D():
    def __init__(self, grid_z) -> None:
        self._grid_z = grid_z

        num_z    = len(self._grid_z)
        self._ne = np.zeros(num_z)
        self._n0 = np.zeros(num_z)

        self._ki = 0.0
        self._mu_i = 0.0
        self._Di   = 0.0
        self._E_field = np.zeros(num_z)

        self._idz = 1.0/(self._grid_z[1] - self._grid_z[0])

        self._flx_p1 = np.zeros(len(self._grid_z))
        self._flx_m1 = np.zeros(len(self._grid_z))

    def setup(self, args):
        pass

    def rhs(self, t, y):
        idz = self._idz
        self._flx_p1[:-1] = self._mu_i * y[1:] * self._E_field[1:]  - self._Di * (y[1:]-y[0:-1]) * idz
        self._flx_p1[-1]  = self._mu_i * y[-1] * self._E_field[-1] # bc on right  

        self._flx_m1[1:]  = self._mu_i * y[0:-1] * self._E_field[0:-1] - self._Di * (y[1:]-y[0:-1]) * idz
        self._flx_m1[0]   = self._mu_i * y[0] * self._E_field[0] # bc on left  
        
        return  -0.5 * idz * (self._flx_p1-self._flx_m1) + self._ki * self._ne * self._n0
        
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
        vth                            = collisions.electron_thermal_velocity(args.electron_volt*args.basis_scale * collisions.TEMP_K_1EV)
        maxwellian                     = BEUtils.get_maxwellian_3d(vth,collisions.MAXWELLIAN_N)
        c_gamma                        = np.sqrt(2*collisions.ELECTRON_CHARGE_MASS_RATIO)
        if "g2Smooth" in args.collisions or "g2" in args.collisions or "g2step" in args.collisions or "g2Regul" in args.collisions:
            sig_pts = np.array([np.sqrt(15.76) * c_gamma/vth])
        else:
            sig_pts = None 
        
        ev_range = ((0*vth/c_gamma)**2, (4*vth/c_gamma)**2)
        k_domain = (np.sqrt(ev_range[0]) * c_gamma / vth, np.sqrt(ev_range[1]) * c_gamma / vth)
        
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
        FOp=0
        collision_op = colOpSp.CollisionOpSP(spec_sp = spec_sp)
        if "g0" in args.collisions:
            g0  = collisions.eAr_G0()
            g0.reset_scattering_direction_sp_mat()
            FOp = FOp + collisions.AR_NEUTRAL_N * collision_op.assemble_mat(g0, mw_vth, vth_curr)
            
        if "g0ConstNoLoss" in args.collisions:
            g0  = collisions.eAr_G0_NoEnergyLoss(cross_section="g0Const")
            g0.reset_scattering_direction_sp_mat()
            FOp = FOp + collisions.AR_NEUTRAL_N * collision_op.assemble_mat(g0, mw_vth, vth_curr)
            
        if "g0NoLoss" in args.collisions:
            g0  = collisions.eAr_G0_NoEnergyLoss()
            g0.reset_scattering_direction_sp_mat()
            FOp = FOp + collisions.AR_NEUTRAL_N * collision_op.assemble_mat(g0, mw_vth, vth_curr)
            
        if "g0Const" in args.collisions:
            g0  = collisions.eAr_G0(cross_section="g0Const")
            g0.reset_scattering_direction_sp_mat()
            FOp = FOp + collisions.AR_NEUTRAL_N * collision_op.assemble_mat(g0, mw_vth, vth_curr)
            
        if "g2" in args.collisions:
            g2  = collisions.eAr_G2()
            g2.reset_scattering_direction_sp_mat()
            FOp = FOp + collisions.AR_NEUTRAL_N * collision_op.assemble_mat(g2, mw_vth, vth_curr)

        if "g2Smooth" in args.collisions:
            g2  = collisions.eAr_G2(cross_section="g2Smooth")
            g2.reset_scattering_direction_sp_mat()
            FOp = FOp + collisions.AR_NEUTRAL_N * collision_op.assemble_mat(g2, mw_vth, vth_curr)
            
        if "g2Regul" in args.collisions:
            g2  = collisions.eAr_G2(cross_section="g2Regul")
            g2.reset_scattering_direction_sp_mat()
            FOp = FOp + collisions.AR_NEUTRAL_N * collision_op.assemble_mat(g2, mw_vth, vth_curr)
            
        if "g2Const" in args.collisions:
            g2  = collisions.eAr_G2(cross_section="g2Const")
            g2.reset_scattering_direction_sp_mat()
            FOp = FOp + collisions.AR_NEUTRAL_N * collision_op.assemble_mat(g2, mw_vth, vth_curr)
            
        if "g2step" in args.collisions:
            g2  = collisions.eAr_G2(cross_section="g2step")
            g2.reset_scattering_direction_sp_mat()
            FOp = FOp + collisions.AR_NEUTRAL_N * collision_op.assemble_mat(g2, mw_vth, vth_curr)
            
        if "g2smoothstep" in args.collisions:
            g2  = collisions.eAr_G2(cross_section="g2smoothstep")
            g2.reset_scattering_direction_sp_mat()
            FOp = FOp + collisions.AR_NEUTRAL_N * collision_op.assemble_mat(g2, mw_vth, vth_curr)
            
        Cmat   = FOp
        Emat   = spec_sp.compute_advection_matix()

        self._spec_sp = spec_sp
        self._col_op  = collision_op

        self._Cmat    = Cmat
        self._EMat    = Emat 

        self._lm_to_vt = lm_modes_to_vt_pts_op(self._grid_z, self._spec_sp, self._grid_vt)
        self._vt_to_lm = vt_pts_to_lm_modes_op(self._grid_z, self._spec_sp, self._grid_vt)
        self._maxwellian = mw_vth
        self._vth        = vth_curr

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
        self._rates_op_ion = BEUtils.reaction_rates_op(spec_sp, collisions.eAr_G2(), self._maxwellian, self._vth)
    
    def rhs(self, t, y):
        spec_sp = self._spec_sp
        num_p   = spec_sp._p + 1
        num_sh  = len(spec_sp._sph_harm_lm)

        num_vt  = len(self._grid_vt)
        num_z   = len(self._grid_z)

        y = y.reshape((num_z, num_p, num_vt))

        y[0 , :, np.where(self._grid_vt<=np.pi/2)] = 0.0
        y[-1, :, np.where(self._grid_vt>np.pi/2)]  = 0.0

        y_klm   = np.dot(y,self._vt_to_lm).reshape((num_z, num_p*num_sh))
        CpE_klm = np.zeros_like(y_klm)

        for z_id in range(num_z):
            FOp              = self._Cmat + self._EMat * (self._E_field[z_id] * collisions.ELECTRON_CHARGE_MASS_RATIO/self._vth)
            CpE_klm[z_id,:]  = np.matmul(FOp, y_klm[z_id , :])

        CpE_kvt = np.dot(CpE_klm.reshape(num_z, num_p , num_sh), self._lm_to_vt).reshape((num_z, num_p * num_vt))
        
        CpE_kvt = np.transpose(CpE_kvt).reshape((num_p, num_vt, num_z)).reshape((num_p, num_vt * num_z))
        CpE_kvt = np.matmul(self._inv_mmat_radial, CpE_kvt)
        CpE_kvt = CpE_kvt.reshape((num_p , num_vt, num_z)).reshape((num_p * num_vt, num_z)).transpose()

        yy      = y.reshape((num_z, num_p, num_vt))
        dz      = (self._grid_z[1]-self._grid_z[0])

        Dzy_kvt = np.zeros((num_z, num_p, num_vt))
        for vt_idx, vt in enumerate(self._grid_vt):
            if vt <= np.pi/2:
                for k in range(num_p):
                    Dzy_kvt[:, k, vt_idx] = grad1_p_11(yy[:,k,vt_idx],dz) * np.cos(vt)
            else:
                for k in range(num_p):
                    Dzy_kvt[:, k, vt_idx] = grad1_m_11(yy[:,k,vt_idx],dz) * np.cos(vt)

        Dzy_kvt = Dzy_kvt.reshape((num_z, num_p * num_vt)).transpose().reshape((num_p, num_vt, num_z)).reshape((num_p, num_vt * num_z))
        Dzy_kvt = np.matmul(self._inv_mmat_radial,np.matmul(self._Wpk_mat, Dzy_kvt))
        Dzy_kvt = Dzy_kvt.reshape((num_p , num_vt, num_z)).reshape((num_p * num_vt, num_z)).transpose()

        y_rhs   = (CpE_kvt - Dzy_kvt).reshape(num_z * num_p * num_vt) 
        return y_rhs

    def compute_ne(self, f_kvt):
        spec_sp = self._spec_sp
        num_p   = spec_sp._p + 1
        num_sh  = len(spec_sp._sph_harm_lm)

        num_vt  = len(self._grid_vt)
        num_z   = len(self._grid_z)
        yy      = np.dot(f_kvt.reshape((num_z, num_p , num_vt)),self._vt_to_lm).reshape((num_z, num_p*num_sh))
        ne      = np.dot(self._mass_op, yy.transpose()).reshape(num_z) * self._vth**3 * self._maxwellian(0)
        
        return ne
    
    def compute_reaction_rates(self, rates_op, f_kvt):
        spec_sp = self._spec_sp
        num_p   = spec_sp._p + 1
        num_sh  = len(spec_sp._sph_harm_lm)

        num_vt  = len(self._grid_vt)
        num_z   = len(self._grid_z)
        
        yy      = np.dot(f_kvt.reshape((num_z, num_p , num_vt)),self._vt_to_lm).reshape((num_z, num_p*num_sh))
        m0      = np.dot(self._mass_op, yy.transpose()).reshape(num_z) * self._vth**3 * self._maxwellian(0)
        ki      = np.dot(rates_op, yy.transpose()).reshape(num_z) / m0

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
        self._ne     = self._boltzmann_solver.compute_ne(self._fx_kvt)
        self._ni     = np.array(self._ne)

        self._fluids_solver._n0   = np.ones_like(grid_z) * collisions.AR_NEUTRAL_N
        self._fluids_solver._ne   = self._ne
        self._fluids_solver._mu_i = 4.65e21
        self._fluids_solver._Di   = 2.07e20
        self._fluids_solver._ki   = self._boltzmann_solver.compute_reaction_rates(self._boltzmann_solver._rates_op_ion, self._fx_kvt)
        
        self._ode_bte = scipy.integrate.ode(self._boltzmann_solver.rhs)
        self._ode_bte.set_integrator("dopri5", verbosity=1, rtol = self._ts_rtol, atol = self._ts_atol, nsteps=1e4)
        self._ode_bte.set_initial_value(self._fx_kvt, t=self._t_begin)
        
        self._ode_ni  = scipy.integrate.ode(self._fluids_solver.rhs)
        self._ode_ni.set_integrator("dopri5", verbosity=1, rtol = self._ts_rtol, atol = self._ts_atol, nsteps=1e4)
        self._ode_ni.set_initial_value(self._ni, t=self._t_begin)

        self._e_field = np.zeros_like(self._ni)

    def step(self,dt):
        print("ts-stepping")
        self._e_field , _  = self._e_field_solver.electric_field_solve(self._ni, self._fluids_solver._ne, self._rf_freq, self._t_curr, self._v0_z0)

        self._boltzmann_solver._E_field = self._e_field
        self._fx_kvt = self._ode_bte.integrate(self._t_curr + dt)
        

        self._ne                  = self._boltzmann_solver.compute_ne(self._fx_kvt)
        self._fluids_solver._ne   = self._ne
        self._fluids_solver._ki   = self._boltzmann_solver.compute_reaction_rates(self._boltzmann_solver._rates_op_ion, self._fx_kvt)
        
        self._ni                  = self._ode_ni.integrate(self._t_curr + dt)
        

        self._t_curr +=dt
        return 

    

if __name__== "__main__":
    args         = parser.parse_args()
    print("solver arguments ",args)

    glow_sim_1d = GlowSim_1D(args)
    glow_sim_1d.init(0,args.T_END, args)

    while glow_sim_1d._t_curr < args.T_DT * 5:#args.T_END:
        print("E " , glow_sim_1d._e_field)
        print("ne ", glow_sim_1d._ne)
        print("ni ", glow_sim_1d._ni)


        glow_sim_1d.step(args.T_DT)
        
