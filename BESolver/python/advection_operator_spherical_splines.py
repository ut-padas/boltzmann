from cProfile import label
import enum
from random import randrange

import basis
import spec_spherical as sp
import scipy.integrate
import numpy as np
import maxpoly
import matplotlib.pyplot as plt
from scipy.special import sph_harm
import utils as BEUtils
import parameters as params
import collisions
import argparse

plt.rcParams.update({
    "text.usetex": False,
    "font.size": 10,
    #"font.family": "Helvetica",
    "lines.linewidth":1.0
})

def create_xlbspline_spec(spline_order, k_domain, Nr, sph_harm_lm, sig_pts=None):
    splines      = basis.BSpline(k_domain,spline_order,Nr+1, sig_pts=sig_pts, knots_vec=None, dg_splines=args.use_dg)
    spec         = sp.SpectralExpansionSpherical(Nr,splines,sph_harm_lm)
    return spec

def assemble_advection_matrix_lp(spec: sp.SpectralExpansionSpherical):
    return spec.compute_advection_matix()

def assemble_advection_matrix_dg(spec: sp.SpectralExpansionSpherical):
    return spec.compute_advection_matix_dg()
    
def backward_euler(FOp,y0,t_end,nsteps):
    dt = t_end/nsteps
    A  = np.linalg.inv(np.eye(FOp.shape[0])+ dt* FOp)
    #A  = np.linalg.matrix_power(A,nsteps)
    for i in range(nsteps):
        y0=np.matmul(A,y0)

    return y0

def solve_advection(nr, sph_lm, sp_order, v_doamin,t_end=5e-1):

    NR            = nr
    L_MODE_MAX    = sph_lm[-1][0]
    V_DOMAIN      = v_doamin
    VTH           = 1.0
    spline_order  = sp_order
    basis.BSPLINE_NUM_Q_PTS_PER_KNOT = args.spline_q_pts_per_knot
    basis.BSPLINE_BASIS_ORDER=spline_order

    num_p = NR+1
    num_sph=len(sph_lm)

    params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER = NR
    params.BEVelocitySpace.SPH_HARM_LM = sph_lm

    sig_pts = np.array([0.5 * (v_doamin[0] + v_doamin[1])])
    print("dg points : ", sig_pts)
    spec_xlbspline = create_xlbspline_spec(spline_order,V_DOMAIN,NR,sph_lm, sig_pts=sig_pts)

    NUM_Q_VR = spec_xlbspline._basis_p._num_knot_intervals * basis.BSPLINE_NUM_Q_PTS_PER_KNOT
    NUM_Q_VT = 4
    NUM_Q_VP = 2

    params.BEVelocitySpace.NUM_Q_VR  = NUM_Q_VR
    params.BEVelocitySpace.NUM_Q_VT  = NUM_Q_VT
    params.BEVelocitySpace.NUM_Q_VP  = NUM_Q_VP
    params.BEVelocitySpace.NUM_Q_CHI = 2
    params.BEVelocitySpace.NUM_Q_PHI = 2
    spec_xlbspline._num_q_radial     = params.BEVelocitySpace.NUM_Q_VR
    params.print_parameters()

    M    = spec_xlbspline.compute_mass_matrix()
    print("mass mat condition number = %.2E"%np.linalg.cond(M))
    Minv = spec_xlbspline.inverse_mass_mat(M) #BEUtils.choloskey_inv(M)
    print("|I-M^{-1} M| = %.16E " %np.linalg.norm(np.matmul(Minv,M)-np.eye(M.shape[0])))
    #print(M[0,:])

    vth_factor   = 1.0
    print("using vth factor : ", vth_factor, "for Nr: ", nr)
    VTH_C        = vth_factor * VTH
    maxwellian   = BEUtils.get_maxwellian_3d(VTH_C, 1)
    hv           = lambda v,vt,vp : np.exp(-v**2)
    h_vec        = BEUtils.function_to_basis(spec_xlbspline,hv,maxwellian,NUM_Q_VR, NUM_Q_VT, NUM_Q_VP,Minv=Minv)

    spec_sp        = spec_xlbspline
    
    advmat, eA, qA = spec_sp.compute_advection_matix_dg(advection_dir=args.Vz/abs(args.Vz))
    advmat      = (advmat) * args.Vz
    #eA = np.kron(np.eye(spec_sp.get_num_radial_domains()), np.kron(np.eye(num_p), eA))
    qA = np.kron(np.eye(spec_sp.get_num_radial_domains()), np.kron(np.eye(num_p), qA))

    coeffs_new  = np.matmul(np.transpose(qA), h_vec)
    coeffs      = np.matmul(np.transpose(qA), h_vec)
    advmat      = np.matmul(Minv, advmat)
        
    
    func = lambda t,a: -np.matmul(advmat, a)  
    sol = scipy.integrate.solve_ivp(func, (0,t_end), coeffs, max_step=DT, method='RK45',atol=1e-15, rtol=2.220446049250313e-14,t_eval=np.linspace(0,t_end,10))
    coeffs_new = np.matmul(qA, sol.y[:,-1])
    coeffs     = np.matmul(qA, coeffs)


    spec_sp=spec_xlbspline
    vth=VTH_C
    current_mw=maxwellian
    mass_op   = BEUtils.mass_op(spec_sp, None, 64, 2, 1)
    temp_op   = BEUtils.temp_op(spec_sp, None, 64, 2, 1)
    avg_vop   = BEUtils.mean_velocity_op(spec_sp, None, 64, 4, 1)
    eavg_to_K = (2/(3*scipy.constants.Boltzmann))
    ev_fac    = (collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT)

    sol.y=np.matmul(qA, sol.y)

    for i in range(0,sol.y.shape[1]):
        current_mass     = np.dot(sol.y[:,i],mass_op) * vth**3 * current_mw(0)
        current_temp     = np.dot(sol.y[:,i],temp_op) * vth**5 * current_mw(0) * 0.5 * collisions.MASS_ELECTRON * eavg_to_K / current_mass
        print("time %.4E mass = %.14E temp= %.8E"%(sol.t[i],current_mass,current_temp))

    #print(np.dot(mass_op[0::num_sph], advmat[0::num_sph,1::num_sph]))
    avg_vop   = BEUtils.mean_velocity_op(spec_xlbspline, None, 64, 32, 1)

    vc_x             = np.dot(avg_vop[0],sol.y[:,0]) * VTH_C**4 * (maxwellian(0)/1)
    vc_y             = np.dot(avg_vop[1],sol.y[:,0]) * VTH_C**4 * (maxwellian(0)/1)
    vc_z             = np.dot(avg_vop[2],sol.y[:,0]) * VTH_C**4 * (maxwellian(0)/1)
    print("vcenter (t=0) = (%.8E, %.8E, %.8E)"%(vc_x, vc_y,vc_z))

    vc_x             = np.dot(avg_vop[0],sol.y[:,-1]) * VTH_C**4 * (maxwellian(0)/1)
    vc_y             = np.dot(avg_vop[1],sol.y[:,-1]) * VTH_C**4 * (maxwellian(0)/1)
    vc_z             = np.dot(avg_vop[2],sol.y[:,-1]) * VTH_C**4 * (maxwellian(0)/1)
    print("vcenter (t=T) = (%.8E, %.8E, %.8E)"%(vc_x, vc_y,vc_z))

    
    #coeffs_new = backward_euler(advmat,h_vec,t_end,1000)
    # func   = lambda t,a: -np.matmul(advmat,a)
    # sol = scipy.integrate.solve_ivp(func, (0,t_end), h_vec,method='BDF',atol=1e-14,rtol=1e-14)
    # print(sol)
    # coeffs_new=sol.y[:,-1]

    return coeffs,coeffs_new,spec_xlbspline

parser = argparse.ArgumentParser()
parser.add_argument("-Nr", "--NUM_P_RADIAL"                   , help="Number of polynomials in radial direction", nargs='+', type=int, default=[32, 64])
parser.add_argument("-T", "--T_END"                           , help="Simulation time", type=float, default=1e-3)
parser.add_argument("-dt", "--T_DT"                           , help="Simulation time step size ", type=float, default=1e-4)
parser.add_argument("-o",  "--out_fname"                      , help="output file name", type=str, default='advection_splines')
parser.add_argument("-ts_tol", "--ts_tol"                     , help="adaptive timestep tolerance", type=float, default=1e-15)
parser.add_argument("-l_max", "--l_max"                       , help="max polar modes in SH expansion", type=int, default=1)
parser.add_argument("-q_vr", "--quad_radial"                  , help="quadrature in r"        , type=int, default=200)
parser.add_argument("-q_vt", "--quad_theta"                   , help="quadrature in polar"    , type=int, default=8)
parser.add_argument("-q_vp", "--quad_phi"                     , help="quadrature in azimuthal", type=int, default=8)
parser.add_argument("-q_st", "--quad_s_theta"                 , help="quadrature in scattering polar"    , type=int, default=8)
parser.add_argument("-q_sp", "--quad_s_phi"                   , help="quadrature in scattering azimuthal", type=int, default=8)
parser.add_argument("-sp_order", "--spline_order"             , help="b-spline order", type=int, default=1)
parser.add_argument("-spline_qpts", "--spline_q_pts_per_knot" , help="q points per knots", type=int, default=4)
parser.add_argument("-E", "--E_field"                         , help="Electric field in V/m", type=float, default=100)
parser.add_argument("-dv", "--dv_target"                      , help="target displacement of distribution in v_th units", type=float, default=0)
parser.add_argument("-nt", "--num_timesteps"                  , help="target number of time steps", type=float, default=100)
parser.add_argument("-dg", "--use_dg"                         , help="enable dg splines", type=int, default=0)
parser.add_argument("-Vz", "--Vz"                             , help="z - speed ", type=float, default=1)
args         = parser.parse_args()
args.Vz=args.Vz

print(args)

num_dofs_all  = [(nr, args.l_max) for nr in args.NUM_P_RADIAL]
error_linf    = np.zeros(len(num_dofs_all))
error_l2      = np.zeros(len(num_dofs_all))
error_linf_2d = np.zeros(len(num_dofs_all))
error_l2_2d   = np.zeros(len(num_dofs_all))

DT       = args.T_DT
T_END    = args.T_END
V_DOMAIN = (0,3)
SP_ORDER = args.spline_order

x = np.linspace(0, 0.9999 * V_DOMAIN[1], 50)
z = np.linspace(0, 0.9999 * V_DOMAIN[1], 50)
quad_grid = np.meshgrid(x,z,indexing='ij')
y = np.zeros_like(quad_grid[0])

sph_coord_init = BEUtils.cartesian_to_spherical(quad_grid[0],y,quad_grid[1])
sph_coord_end  = BEUtils.cartesian_to_spherical(quad_grid[0],y,quad_grid[1]-T_END * args.Vz)

theta = 0.5*np.pi - np.sign(x)*0.5*np.pi
f_num = np.zeros([len(num_dofs_all), len(x)])
f_initial = np.zeros([len(num_dofs_all), len(x)])
f_exact = np.zeros([len(num_dofs_all), len(x)])

f_num_2d     = np.zeros([len(num_dofs_all), len(x), len(z)])
f_initial_2d = np.zeros([len(num_dofs_all), len(x), len(z)])
f_exact_2d   = np.zeros([len(num_dofs_all), len(x), len(z)])

for num_dofs_idx, num_dofs in enumerate(num_dofs_all):
    nr = num_dofs[0]
    l_max = num_dofs[1]
    
    num_p   = nr+1
    sph_lm  = [[l,0] for l in range(l_max+1)]
    num_sph = len(sph_lm)
    print("Nr=%d sph=%s"%(nr,sph_lm))
    c,ct,spec_xlbspline = solve_advection(nr,sph_lm,SP_ORDER,V_DOMAIN,T_END)

    Vq_r_2d    = np.zeros(tuple([l_max+1,num_p]) + sph_coord_init[0].shape)
    Vq_rt_2d   = np.zeros(tuple([l_max+1,num_p]) + sph_coord_init[0].shape)
    
    Vq_r       = np.zeros(tuple([l_max+1,num_p]) + x.shape)
    Vq_rt      = np.zeros(tuple([l_max+1,num_p]) + x.shape) 
    
    for l in range(l_max+1):
        Vq_r_2d[l]   = spec_xlbspline.Vq_r(sph_coord_init[0],l)
        Vq_rt_2d[l]  = spec_xlbspline.Vq_r(sph_coord_end[0],l)

        Vq_r[l]      = spec_xlbspline.Vq_r(np.abs(x),l)
        Vq_rt[l]     = spec_xlbspline.Vq_r(np.abs(x-T_END * args.Vz),l)
        
    f_eval_mat_2d = np.transpose(np.array([spec_xlbspline.basis_eval_spherical(sph_coord_init[1],sph_coord_init[2],lm[0],lm[1]) * Vq_r_2d[lm[0],k,:]   for k in range(num_p) for lm_idx, lm in enumerate(sph_lm)]).reshape(num_p*num_sph,-1))

    f_eval_mat_2d_t = np.transpose(np.array([spec_xlbspline.basis_eval_spherical(sph_coord_init[1],sph_coord_init[2],lm[0],lm[1]) * Vq_rt_2d[lm[0],k,:]   for k in range(num_p) for lm_idx, lm in enumerate(sph_lm)]).reshape(num_p*num_sph,-1))
    
    f_eval_mat=np.transpose(np.array([spec_xlbspline.basis_eval_spherical(theta,0,lm[0],lm[1]) * Vq_r[lm[0],k,:]   for k in range(num_p) for lm_idx, lm in enumerate(sph_lm)]).reshape(num_p*num_sph,-1))
    f_eval_mat_t=np.transpose(np.array([spec_xlbspline.basis_eval_spherical(theta,0,lm[0],lm[1]) * Vq_rt[lm[0],k,:]   for k in range(num_p) for lm_idx, lm in enumerate(sph_lm)]).reshape(num_p*num_sph,-1))

    f_num[num_dofs_idx,:]     = np.dot(f_eval_mat,ct)
    f_initial[num_dofs_idx,:] = np.dot(f_eval_mat,c)
    f_exact[num_dofs_idx,:]   = np.dot(f_eval_mat_t,c)

    f_num_2d[num_dofs_idx,:]     = np.dot(f_eval_mat_2d,ct).reshape((len(x),len(z)))
    f_initial_2d[num_dofs_idx,:] = np.dot(f_eval_mat_2d,c).reshape((len(x),len(z)))
    f_exact_2d[num_dofs_idx,:]   = np.dot(f_eval_mat_2d_t,c).reshape((len(x),len(z)))

for num_dofs_idx, num_dofs in enumerate(num_dofs_all):   
    error_linf[num_dofs_idx] = np.max(abs(f_num[num_dofs_idx,:]-f_exact[-1,:]))
    ii_index=np.argmax(abs(f_num[num_dofs_idx,:]-f_exact[-1,:]))
    print("max : %.14E  = (%.14E, %.14E) occurs at x=%.14E" %(error_linf[num_dofs_idx],f_num[num_dofs_idx,ii_index], f_exact[-1,ii_index] , x[ii_index]))
    error_l2[num_dofs_idx] = np.linalg.norm(f_num[num_dofs_idx,:]-f_exact[-1,:])/np.linalg.norm(f_exact[-1,:])

    error_linf_2d[num_dofs_idx] = np.max(abs(f_num_2d[num_dofs_idx,:]-f_exact_2d[-1,:]))
    error_l2_2d[num_dofs_idx] = np.linalg.norm(f_num_2d[num_dofs_idx,:]-f_exact_2d[-1,:])


fig = plt.figure(figsize=(14, 5), dpi=300)
if len(spec_xlbspline._basis_p._dg_idx)==2:
    fig.suptitle('CG sp_order= %d T=%.4E lmax=%d'%(args.spline_order, T_END, args.l_max)) 
else:
    fig.suptitle('DG sp_order= %d T=%.4E lmax=%d'%(args.spline_order, T_END, args.l_max)) 

plt.subplot(1,3,1)
plt.semilogy(x, np.abs(f_initial[-1,:]), label="initial")
plt.plot(x, np.abs(f_exact[-1,:]), label="exact")
for num_dofs_idx, num_dofs in enumerate(num_dofs_all):
    plt.plot(x, np.abs(f_num[num_dofs_idx,:]), '--', label="(%d,%d)"%(num_dofs[0],num_dofs[1]))

plt.legend()
plt.grid()
plt.ylabel('f(v_z)')
plt.xlabel('$v_z$')


plt.subplot(1,3,2)
plt.contour(quad_grid[0], quad_grid[1], f_initial_2d[-1,:,:], linestyles='solid', colors='grey', linewidths=1,levels=np.logspace(-25, 1, 15, base=10))
plt.contour(quad_grid[0], quad_grid[1], f_exact_2d[-1,:,:], linestyles='dashed', colors='red', linewidths=1, levels=np.logspace(-25, 1, 15, base=10))
ax = plt.contour(quad_grid[0], quad_grid[1], f_num_2d[-1,:,:], linestyles='dotted', colors='blue', linewidths=1, levels=np.logspace(-25, 1, 15, base=10))
plt.gca().set_aspect('equal')


plt.subplot(1,3,3)
plt.semilogy(np.array(num_dofs_all)[:,0], error_l2, '-o')
plt.ylabel('error')
plt.xlabel("Nr")
plt.grid()

if len(spec_xlbspline._basis_p._dg_idx)==2:
    fig.savefig("%s_cg.png"%(args.out_fname), dpi=300)
else:
    fig.savefig("%s_dg.png"%(args.out_fname), dpi=300)

#plt.savefig()
#plt.show()

