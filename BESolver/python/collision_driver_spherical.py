"""
@package Boltzmann collision operator solver. 
"""

from cProfile import label
import enum
import scipy
from sympy import arg
import basis
import spec_spherical as sp
import numpy as np
import collision_operator_spherical as colOpSp
import collisions 
import parameters as params
import os
from time import perf_counter as time
import utils as BEUtils
import argparse
import scipy.integrate
from scipy.integrate import ode
import scipy.linalg
import matplotlib.pyplot as plt

class CollissionMode(enum.Enum):
    ELASTIC_ONLY=0
    ELASTIC_W_EXCITATION=1
    ELASTIC_W_IONIZATION=2
    ELASTIC_W_EXCITATION_W_IONIZATION=3

collisions.AR_NEUTRAL_N=3.22e22
collisions.MAXWELLIAN_N=1
collisions.AR_IONIZED_N=collisions.AR_NEUTRAL_N #collisions.MAXWELLIAN_N

def spec_tail(cf, num_p, num_sh):
    return np.linalg.norm(cf[(num_p//2) * num_sh :])

def spec_tail_timeseries(cf, num_p, num_sh):
    return np.array([np.linalg.norm(cf[i, (num_p//2) * num_sh :])/len(cf[i, (num_p//2) * num_sh :]) for i in range(data.shape[0])])


def svd_truncate(FOp,r_tol=1e-6):
    u, s, v = np.linalg.svd(FOp)
    FOp_svd = np.matmul(u, np.matmul(np.diag(s), v))  
    print("SVD rel. error : %.12E"%(np.linalg.norm(FOp_svd-FOp)/np.linalg.norm(FOp)))
    st = np.where(s/s[0]>r_tol) 
    s_k = st[0][-1]
    print("truncated at =%d out of %d "%(s_k,len(s)))
    FOp_svd = np.matmul(u[:, 0:s_k], np.matmul(np.diag(s[0:s_k]), v[0:s_k,:]))  
    print("SVD after truncation rel. error : %.12E"%(np.linalg.norm(FOp_svd-FOp)/np.linalg.norm(FOp)))
    return FOp_svd


def constant_r_eval(spec_sp : sp.SpectralExpansionSpherical, cf, r):
    theta = np.linspace(0,np.pi,50)
    phi   = np.linspace(0,2*np.pi,100)
    grid  = np.meshgrid(theta, phi, indexing='ij')
    
    num_p  = spec_sp._p + 1
    num_sh = len(spec_sp._sph_harm_lm)
    
    b_eval = np.array([ np.exp(-r**2) * spec_sp.basis_eval_radial(r,k,l) * spec_sp.basis_eval_spherical(grid[0],grid[1],l,m) for lm_idx, (l,m) in enumerate(spec_sp._sph_harm_lm) for k in range(num_p)])
    b_eval = b_eval.reshape(num_sh,num_p, -1)
    b_eval = np.swapaxes(b_eval,0,1)
    b_eval = b_eval.reshape(num_p * num_sh,-1)
    
    return np.dot(cf, b_eval).reshape(50,100), theta, phi
    

def solve_collop(collOp : colOpSp.CollisionOpSP, maxwellian, vth, E_field, t_end, dt, collisions_included):
    spec_sp = collOp._spec
    Mmat = spec_sp.compute_mass_matrix()
    Minv = spec_sp.inverse_mass_mat(Mmat=Mmat)
    mm_inv_error=np.linalg.norm(np.matmul(Mmat,Minv)-np.eye(Mmat.shape[0]))/np.linalg.norm(np.eye(Mmat.shape[0]))
    print("cond(M) = %.4E"%np.linalg.cond(Mmat))
    print("|I-M M^{-1}| error = %.12E"%(mm_inv_error))
    
    # if not (spec_sp.get_radial_basis_type()==basis.BasisType.CHEBYSHEV_POLY or spec_sp.get_radial_basis_type()==basis.BasisType.SPLINES):
    #     raise NotImplementedError

    MVTH  = vth
    MNE   = maxwellian(0) * (np.sqrt(np.pi)**3) * (vth**3)
    MTEMP = collisions.electron_temperature(MVTH)
    print("==========================================================================")
    vratio = np.sqrt(1.0/args.basis_scale)

    if (args.radial_poly == "maxwell" or args.radial_poly == "laguerre"):
        hv    = lambda v,vt,vp : np.exp((v**2)*(1.-1./(vratio**2)))/vratio**3
    elif (args.radial_poly == "maxwell_energy"):
        hv    = lambda v,vt,vp : np.exp((v**2)*(1.-1./(vratio**2)))/vratio**3
    elif(args.radial_poly == "chebyshev"):
        hv    = lambda v,vt,vp : np.exp(-((v/vratio)**2)) / vratio**3
    elif (args.radial_poly == "bspline"):
        hv    = lambda v,vt,vp : np.exp(-((v/vratio)**2)) / vratio**3
    else:
        raise NotImplementedError

    h_init    = BEUtils.function_to_basis(spec_sp,hv,maxwellian,None,None,None,Minv=Minv)
    h_t       = np.array(h_init)
    
    ne_t      = MNE
    mw_vth    = BEUtils.get_maxwellian_3d(vth,ne_t)
    m0_t0     = BEUtils.moment_n_f(spec_sp,h_t,mw_vth,vth,0,None,None,None,1)
    temp_t0   = BEUtils.compute_avg_temp(collisions.MASS_ELECTRON,spec_sp,h_t,mw_vth,vth,None,None,None,m0_t0,1)
    
    vth_curr  = vth 
    print("Initial Ev : "   , temp_t0 * collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT)
    print("Initial mass : " , m0_t0 )


    FOp = 0
    t1=time()
    if "g0" in collisions_included:
        g0  = collisions.eAr_G0()
        g0.reset_scattering_direction_sp_mat()
        FOp = FOp + collisions.AR_NEUTRAL_N * collOp.assemble_mat(g0, mw_vth, vth_curr)

    if "g0NoLoss" in collisions_included:
        g0noloss  = collisions.eAr_G0_NoEnergyLoss()
        g0noloss.reset_scattering_direction_sp_mat()
        FOp = FOp + collisions.AR_NEUTRAL_N * collOp.assemble_mat(g0noloss, mw_vth, vth_curr)
    
    if "g0ConstNoLoss" in collisions_included:
        g0noloss  = collisions.eAr_G0_NoEnergyLoss(cross_section="g0Const")
        g0noloss.reset_scattering_direction_sp_mat()
        FOp = FOp + collisions.AR_NEUTRAL_N * collOp.assemble_mat(g0noloss, mw_vth, vth_curr)

    if "g0Const" in collisions_included:
        g0const  = collisions.eAr_G0(cross_section="g0Const")
        g0const.reset_scattering_direction_sp_mat()
        FOp = FOp + collisions.AR_NEUTRAL_N * collOp.assemble_mat(g0const, mw_vth, vth_curr)

    if "g2" in collisions_included:
        g2  = collisions.eAr_G2()
        g2.reset_scattering_direction_sp_mat()
        FOp = FOp + collisions.AR_NEUTRAL_N * collOp.assemble_mat(g2, mw_vth, vth_curr)

    if "g2Smooth" in collisions_included:
        g2Smooth  = collisions.eAr_G2(cross_section="g2Smooth")
        g2Smooth.reset_scattering_direction_sp_mat()
        FOp = FOp + collisions.AR_NEUTRAL_N * collOp.assemble_mat(g2Smooth, mw_vth, vth_curr)

    if "g2Regul" in collisions_included:
        g2Regul  = collisions.eAr_G2(cross_section="g2Regul")
        g2Regul.reset_scattering_direction_sp_mat()
        FOp = FOp + collisions.AR_NEUTRAL_N * collOp.assemble_mat(g2Regul, mw_vth, vth_curr)

    if "g2Const" in collisions_included:
        g2  = collisions.eAr_G2(cross_section="g2Const")
        g2.reset_scattering_direction_sp_mat()
        FOp = FOp + collisions.AR_NEUTRAL_N * collOp.assemble_mat(g2, mw_vth, vth_curr)

    if "g2step" in collisions_included:
        g2  = collisions.eAr_G2(cross_section="g2step")
        g2.reset_scattering_direction_sp_mat()
        FOp = FOp + collisions.AR_NEUTRAL_N * collOp.assemble_mat(g2, mw_vth, vth_curr)

    if "g2smoothstep" in collisions_included:
        g2  = collisions.eAr_G2(cross_section="g2smoothstep")
        g2.reset_scattering_direction_sp_mat()
        FOp = FOp + collisions.AR_NEUTRAL_N * collOp.assemble_mat(g2, mw_vth, vth_curr)

    t2=time()
    print("Assembled the collision op. for Vth : ", vth_curr)
    print("Collision Operator assembly time (s): ",(t2-t1))
    num_p   = spec_sp._p + 1
    num_sh  = len(spec_sp._sph_harm_lm)
    # # np.set_printoptions(precision=2)
    # # print(FOp[0::num_sh,0::num_sh])
    # # print(FOp[1::num_sh,1::num_sh])
    # if len(dg_idx)>2:
    #     for lm_idx, lm in enumerate(spec_sp._sph_harm_lm):
    #         FOp[dg_idx[2] * num_sh + lm_idx :: num_sh , dg_idx[1] * num_sh + lm_idx] = FOp[dg_idx[2] * num_sh + lm_idx::num_sh, dg_idx[2] * num_sh + lm_idx] 
    #         FOp[dg_idx[2] * num_sh + lm_idx :: num_sh , dg_idx[2] * num_sh + lm_idx] = 0
    #         #FOp[dg_idx[2] * num_sh + lm_idx, dg_idx[2] * num_sh + lm_idx]=-1
    #         #FOp[dg_idx[2] * num_sh + lm_idx, dg_idx[1] * num_sh + lm_idx]=1

    # # print(FOp[0::num_sh,0::num_sh])
    # # print(FOp[1::num_sh,1::num_sh])
    FOp     = np.matmul(Minv, FOp)
    MVTH  = vth
    MNE   = maxwellian(0) * (np.sqrt(np.pi)**3) * (vth**3)
    MTEMP = collisions.electron_temperature(MVTH)

    def dg_collop_bdy(spec_sp,yy):
        dg_idx  = spec_sp._basis_p._dg_idx
        if (len(dg_idx)>2):
            for qs_idx, qs in enumerate(spec_sp._sph_harm_lm):
                yy[dg_idx[1]*num_sh + qs_idx] = 0.5 * (yy[dg_idx[1]*num_sh + qs_idx] + yy[dg_idx[2]*num_sh + qs_idx])
                yy[dg_idx[2]*num_sh + qs_idx] = 0.5 * (yy[dg_idx[1]*num_sh + qs_idx] + yy[dg_idx[2]*num_sh + qs_idx])

    def f_rhs(t,y):
        
        dg_collop_bdy(spec_sp,y)
        x=np.matmul(FOp,y)
        dg_collop_bdy(spec_sp,x)
        return x

    sol = scipy.integrate.solve_ivp(f_rhs, (0,t_end), h_init, max_step=dt, method='RK45',atol=1e-40, rtol=2.220446049250313e-14, t_eval=np.linspace(0,t_end,10))
    
    for i in range(0,sol.y.shape[1]):
        dg_collop_bdy(spec_sp, sol.y[:,i])
    
    for i in range(0,sol.y.shape[1]):
        current_mass     = np.dot(sol.y[:,i],mass_op) * vth**3 * maxwellian(0)
        current_temp     = np.dot(sol.y[:,i],temp_op) * vth**5 * maxwellian(0) * 0.5 * collisions.MASS_ELECTRON * eavg_to_K / current_mass
        print("time %.4E mass = %.14E temp= %.8E"%(sol.t[i],current_mass,current_temp))    
    

    return np.transpose(sol.y)

parser = argparse.ArgumentParser()
parser.add_argument("-Nr", "--NUM_P_RADIAL"                   , help="Number of polynomials in radial direction", nargs='+', type=int, default=16)
parser.add_argument("-T", "--T_END"                           , help="Simulation time", type=float, default=1e-4)
parser.add_argument("-dt", "--T_DT"                           , help="Simulation time step size ", type=float, default=1e-7)
parser.add_argument("-o",  "--out_fname"                      , help="output file name", type=str, default='coll_op')
parser.add_argument("-ts_tol", "--ts_tol"                     , help="adaptive timestep tolerance", type=float, default=1e-15)
parser.add_argument("-l_max", "--l_max"                       , help="max polar modes in SH expansion", type=int, default=1)
parser.add_argument("-c", "--collisions"                      , help="collisions included (g0, g0Const, g0NoLoss, g2, g2Const)",nargs='+', type=str, default=["g0Const"])
parser.add_argument("-ev", "--electron_volt"                  , help="initial electron volt", type=float, default=0.25)
parser.add_argument("-bscale", "--basis_scale"                , help="basis electron volt", type=float, default=1.0)
parser.add_argument("-q_vr", "--quad_radial"                  , help="quadrature in r"        , type=int, default=200)
parser.add_argument("-q_vt", "--quad_theta"                   , help="quadrature in polar"    , type=int, default=8)
parser.add_argument("-q_vp", "--quad_phi"                     , help="quadrature in azimuthal", type=int, default=8)
parser.add_argument("-q_st", "--quad_s_theta"                 , help="quadrature in scattering polar"    , type=int, default=8)
parser.add_argument("-q_sp", "--quad_s_phi"                   , help="quadrature in scattering azimuthal", type=int, default=8)
parser.add_argument("-radial_poly", "--radial_poly"           , help="radial basis", type=str, default="bspline")
parser.add_argument("-sp_order", "--spline_order"             , help="b-spline order", type=int, default=1)
parser.add_argument("-spline_qpts", "--spline_q_pts_per_knot" , help="q points per knots", type=int, default=7)
parser.add_argument("-E", "--E_field"                         , help="Electric field in V/m", type=float, default=100)
parser.add_argument("-dv", "--dv_target"                      , help="target displacement of distribution in v_th units", type=float, default=0)
parser.add_argument("-nt", "--num_timesteps"                  , help="target number of time steps", type=float, default=100)
parser.add_argument("-steady", "--steady_state"               , help="Steady state or transient", type=bool, default=True)
parser.add_argument("-bolsig", "--bolsig_dir"                 , help="Bolsig directory", type=str, default="../../Bolsig/")
parser.add_argument("-sweep_values", "--sweep_values"         , help="Values for parameter sweep", nargs='+', type=int, default=[24, 48, 96])
parser.add_argument("-sweep_param", "--sweep_param"           , help="Paramter to sweep: Nr, ev, bscale, E, radial_poly", type=str, default="Nr")

args = parser.parse_args()
print(args)

run_data=list()
ev           = np.linspace(args.electron_volt/50.,100.*args.electron_volt,1000)

params.BEVelocitySpace.SPH_HARM_LM = [[i,0] for i in range(args.l_max+1)]
num_sph_harm = len(params.BEVelocitySpace.SPH_HARM_LM)

eedf         = np.zeros((len(args.sweep_values),len(ev)))
eedf_initial = np.zeros((len(args.sweep_values),len(ev)))
temperature  = list()
radial       = np.zeros((len(args.sweep_values), num_sph_harm, len(ev)))
radial_base  = np.zeros((len(args.sweep_values), num_sph_harm, len(ev)))
radial_cg    = np.zeros((len(args.sweep_values), num_sph_harm, len(ev)))
radial_projection = np.zeros((len(args.sweep_values), num_sph_harm, len(ev)))


SPLINE_ORDER = args.spline_order
basis.BSPLINE_BASIS_ORDER=SPLINE_ORDER
basis.XLBSPLINE_NUM_Q_PTS_PER_KNOT=args.spline_q_pts_per_knot
vth_factor_temp=[1,0.95, 0.9, 0.85, 0.8]

for i, value in enumerate(args.sweep_values):
    BASIS_EV                       = args.electron_volt*args.basis_scale
    collisions.MAXWELLIAN_TEMP_K   = BASIS_EV * collisions.TEMP_K_1EV
    collisions.ELECTRON_THEMAL_VEL = collisions.electron_thermal_velocity(collisions.MAXWELLIAN_TEMP_K) 
    VTH                            = collisions.ELECTRON_THEMAL_VEL
    maxwellian                     = BEUtils.get_maxwellian_3d(VTH,collisions.MAXWELLIAN_N)
    c_gamma   = np.sqrt(2*collisions.ELECTRON_CHARGE_MASS_RATIO)
    ev_range  = ((0*VTH/c_gamma)**2, 1.01 * ev[-1])
    k_domain  = (np.sqrt(ev_range[0]) * c_gamma / VTH, np.sqrt(ev_range[1]) * c_gamma / VTH)

    if "g2Smooth" in args.collisions or "g2" in args.collisions or "g2step" in args.collisions or "g2Regular" in args.collisions:
        sig_pts = np.array([np.sqrt(15.76) * c_gamma/VTH])
    else:
        sig_pts = np.array([0.5 * (k_domain[0] + k_domain[1])]) #None
    
    print("target ev range : (%.4E, %.4E) ----> knots domain : (%.4E, %.4E)" %(ev_range[0], ev_range[1], k_domain[0],k_domain[1]))
    if(sig_pts is not None):
        print("sig energy = ", (sig_pts*VTH/c_gamma)**2, " v/vth = ", sig_pts)


    if args.sweep_param == "Nr":
        args.NUM_P_RADIAL = value
    # elif args.sweep_param == "l_max":
    #     args.l_max = value
    elif args.sweep_param == "ev":
        args.electron_volt = value
        args.electron_volt_basis = value
    elif args.sweep_param == "bscale":
        args.basis_scale = value
    elif args.sweep_param == "E":
        args.E_field = value
    elif args.sweep_param == "radial_poly":
        args.radial_poly = value
    elif args.sweep_param == "q_vr":
        args.quad_radial = value

    params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER = args.NUM_P_RADIAL
    tt_vec = None
    if (args.radial_poly == "maxwell"):
        params.BEVelocitySpace.NUM_Q_VR  = args.quad_radial
        spec_sp  = sp.SpectralExpansionSpherical(params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER, basis.Maxwell(), params.BEVelocitySpace.SPH_HARM_LM)
        spec_sp._num_q_radial = params.BEVelocitySpace.NUM_Q_VR
    elif (args.radial_poly == "maxwell_energy"):
        params.BEVelocitySpace.NUM_Q_VR  = args.quad_radial
        spec_sp  = sp.SpectralExpansionSpherical(params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER, basis.MaxwellEnergy(), params.BEVelocitySpace.SPH_HARM_LM)
        spec_sp._num_q_radial = params.BEVelocitySpace.NUM_Q_VR
    elif (args.radial_poly == "laguerre"):
        params.BEVelocitySpace.NUM_Q_VR  = args.quad_radial
        spec_sp  = sp.SpectralExpansionSpherical(params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER, basis.Laguerre(), params.BEVelocitySpace.SPH_HARM_LM)
        spec_sp._num_q_radial = params.BEVelocitySpace.NUM_Q_VR
    elif (args.radial_poly == "chebyshev"):
        params.BEVelocitySpace.NUM_Q_VR  = args.quad_radial
        spec_sp  = sp.SpectralExpansionSpherical(params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER, basis.Chebyshev(domain=(-1,1), window=k_domain) , params.BEVelocitySpace.SPH_HARM_LM)
        spec_sp._num_q_radial = params.BEVelocitySpace.NUM_Q_VR
    elif (args.radial_poly == "bspline"):
        r_mode                          = basis.BasisType.SPLINES
        max_lev                         = int(np.ceil(np.log2(params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER)))

        def refine_func(x):
            y    = np.zeros_like(x)
            x_ev = (x * VTH / c_gamma)**2
            collisions_included = args.collisions
            for c in collisions_included:
                y+= VTH * (x**3) * maxwellian(x) * collisions.Collisions.synthetic_tcs(x_ev, c) * collisions.AR_NEUTRAL_N
            
            return maxwellian(x)

        tt_vec                          = basis.BSpline.adaptive_fit(refine_func, k_domain, sp_order=SPLINE_ORDER, min_lev=4, max_lev=max_lev, sig_pts=sig_pts, atol=1e-40, rtol=1e-12)
        bb                              = basis.BSpline(k_domain, SPLINE_ORDER, params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER+1, sig_pts=sig_pts, knots_vec=tt_vec)
        params.BEVelocitySpace.NUM_Q_VR = bb._num_knot_intervals * args.spline_q_pts_per_knot
        params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER = bb._num_p
        if args.sweep_param == "Nr":
            args.sweep_values[i] = params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER
        spec_sp               = sp.SpectralExpansionSpherical(params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER-1, bb,params.BEVelocitySpace.SPH_HARM_LM)
        spec_sp._num_q_radial = params.BEVelocitySpace.NUM_Q_VR

    params.BEVelocitySpace.NUM_Q_VT  = args.quad_theta
    params.BEVelocitySpace.NUM_Q_VP  = args.quad_phi
    params.BEVelocitySpace.NUM_Q_CHI = args.quad_s_theta
    params.BEVelocitySpace.NUM_Q_PHI = args.quad_s_phi
    params.BEVelocitySpace.VELOCITY_SPACE_DT = args.T_DT
    
    print("""===========================Parameters ======================""")
    print("\tMAXWELLIAN_N        : ", collisions.MAXWELLIAN_N)
    print("\tELECTRON_THEMAL_VEL : ", VTH," ms^-1")
    print("\tBASIS_EV            : ", BASIS_EV,"eV")
    print("\tDT : ", params.BEVelocitySpace.VELOCITY_SPACE_DT, " s")
    print("""============================================================""")
    params.print_parameters()

    cf      = colOpSp.CollisionOpSP(spec_sp=spec_sp)

    if (args.radial_poly == "chebyshev"):
        if sig_pts is not None and sig_pts[0] < k_domain[1] and sig_pts[0] >= k_domain[0]:
            spec_sp._r_grid   = [(k_domain[0], sig_pts[0]) , (sig_pts[0], sig_pts[1]), (sig_pts[1] , k_domain[1])]
            spec_sp._r_basis_p= [basis.Chebyshev(domain=(-1,1), window=ele_domain) for ele_domain in spec_sp._r_grid]
        else:
            d_intervals    = np.linspace(k_domain[0],k_domain[1],1)
            spec_sp._r_grid   = [(d_intervals[i], d_intervals[i+1]) for i in range(len(d_intervals)-1)]
            spec_sp._r_basis_p= [basis.Chebyshev(domain=(-1,1), window=ele_domain) for ele_domain in spec_sp._r_grid]
        
    mass_op   = BEUtils.mass_op(spec_sp, None, 64, 2, 1)
    temp_op   = BEUtils.temp_op(spec_sp, None, 64, 2, 1)
    avg_vop   = BEUtils.mean_velocity_op(spec_sp, None, 64, 4, 1)
    eavg_to_K = (2/(3*scipy.constants.Boltzmann))
    ev_fac    = (collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT)

    data               = solve_collop(cf, maxwellian, VTH, args.E_field, args.T_END, args.T_DT, collisions_included=args.collisions)
    radial_base[i,:,:] = BEUtils.compute_radial_components(ev, spec_sp, data[0,:], maxwellian, VTH, 1)
    radial[i, :, :]    = BEUtils.compute_radial_components(ev, spec_sp, data[-1,:], maxwellian, VTH, 1)
    scale              = 1./( np.trapz(radial[i,0,:]*np.sqrt(ev),x=ev) )
    radial[i, :, :]   *= scale

    eedf_initial[i,:]  = BEUtils.get_eedf(ev, spec_sp, data[0, :], maxwellian, VTH) 
    eedf[i,:]          = BEUtils.get_eedf(ev, spec_sp, data[-1,:], maxwellian, VTH) 
    
    run_data.append(data)



if (1):
    fig = plt.figure(figsize=(15, 9), dpi=300)

    num_subplots = num_sph_harm + 1

    for i, value in enumerate(args.sweep_values):
        data=run_data[i]
        lbl = args.sweep_param+"="+str(value)

        # spherical components plots
        for l_idx in range(num_sph_harm):

            plt.subplot(2, num_subplots, 1 + l_idx)
            color = next(plt.gca()._get_lines.prop_cycler)['color']
            plt.semilogy(ev,  abs(radial[i, l_idx]), '-', label=lbl, color=color)
            # if l_idx == 0 and i==len(args.sweep_values)-1:
            #     plt.semilogy(ev,  abs(radial_base[-1,l_idx]), ':', label=lbl+" (base)", color=color)
            plt.xlabel("Energy, eV")
            plt.ylabel("Radial component")
            plt.grid(visible=True)
            if l_idx == 0:
                plt.legend()

            plt.subplot(2, num_subplots, num_subplots + 1+l_idx)
            color = next(plt.gca()._get_lines.prop_cycler)['color']
            plt.plot(np.abs(data[-1,l_idx::num_sph_harm]),label=lbl, color=color)
            
            plt.title(label="l=%d"%l_idx)
            plt.yscale('log')
            plt.xlabel("coeff #")
            plt.ylabel("abs(coeff)")
            plt.grid(visible=True)
            if l_idx == 0:
                plt.legend()
        

        plt.subplot(2, num_subplots, num_sph_harm+1)
        plt.semilogy(ev, np.abs(eedf[i,:]), label=lbl)
        if i==len(args.sweep_values)-1:
            plt.semilogy(ev, np.abs(eedf_initial[i,:]),'--', label=lbl+" (initial)")
        plt.legend()
        plt.grid(visible=True)
        plt.xlabel("Energy, eV")
        plt.ylabel("eedf")
        
        
            
    fig.subplots_adjust(hspace=0.3)
    fig.subplots_adjust(wspace=0.4)

    if (args.radial_poly == "bspline"):
        fig.suptitle("Collisions: " + str(args.collisions) + ", E = " + str(args.E_field) + ", polys = " + str(args.radial_poly)+", sp_order= " + str(args.spline_order) + ", Nr = " + str(args.NUM_P_RADIAL) + ", bscale = " + str(args.basis_scale) + " (sweeping " + args.sweep_param + ")" + "q_per_knot="+str(args.spline_q_pts_per_knot))
        # plt.show()
        if len(spec_sp._basis_p._dg_idx)==2:
            plt.savefig("coll_op_cg_" + "_".join(args.collisions) + "_E" + str(args.E_field) + "_poly_" + str(args.radial_poly)+ "_sp_"+ str(args.spline_order) + "_nr" + str(args.NUM_P_RADIAL)+"_qpn_" + str(args.spline_q_pts_per_knot) + "_bscale" + str(args.basis_scale) + "_sweeping_" + args.sweep_param + "_tend_" + str(args.T_END) + ".png")
        else:
            plt.savefig("coll_op_dg_" + "_".join(args.collisions) + "_E" + str(args.E_field) + "_poly_" + str(args.radial_poly)+ "_sp_"+ str(args.spline_order) + "_nr" + str(args.NUM_P_RADIAL)+"_qpn_" + str(args.spline_q_pts_per_knot) + "_bscale" + str(args.basis_scale) + "_sweeping_" + args.sweep_param  + "_tend_" + str(args.T_END) + ".png")

    else:
        fig.suptitle("Collisions: " + str(args.collisions) + ", E = " + str(args.E_field) + ", polys = " + str(args.radial_poly) + ", Nr = " + str(args.NUM_P_RADIAL) + ", bscale = " + str(args.basis_scale) + " (sweeping " + args.sweep_param + ")")
        # plt.show()
        plt.savefig("coll_op" + "_".join(args.collisions) + "_E" + str(args.E_field) + "_poly_" + str(args.radial_poly) + "_nr" + str(args.NUM_P_RADIAL) + "_bscale" + str(args.basis_scale) + "_sweeping_" + args.sweep_param + ".png")


