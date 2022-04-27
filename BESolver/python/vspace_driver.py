"""
Evolving the 0-d space Boltzmann eq. with collision and advection. 
"""
from ast import arg
import enum
import scipy
import basis
import spec_spherical as sp
import numpy as np
import collision_operator_spherical as colOpSp
import collisions 
import parameters as params
import utils as BEUtils
import argparse
import os
from scipy.integrate import ode

class CollissionMode(enum.Enum):
    ELASTIC_ONLY=0
    ELASTIC_W_EXCITATION=1
    ELASTIC_W_IONIZATION=2
    ELASTIC_W_EXCITATION_W_IONIZATION=3

collisions.AR_NEUTRAL_N=3.22e22
collisions.MAXWELLIAN_N=1e18
collisions.AR_IONIZED_N=collisions.MAXWELLIAN_N

def spec_tail(cf, num_p, num_sh):
    return np.linalg.norm(cf[(num_p//2) * num_sh :])

def spec_tail_timeseries(cf, num_p, num_sh):
    return np.array([np.linalg.norm(cf[i, (num_p//2) * num_sh :])/len(cf[i, (num_p//2) * num_sh :]) for i in range(cf.shape[0])])


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


def constant_r_eval(spec : sp.SpectralExpansionSpherical, cf, r):
    theta = np.linspace(0,np.pi,50)
    phi   = np.linspace(0,2*np.pi,100)
    grid  = np.meshgrid(theta, phi, indexing='ij')
    
    num_p  = spec._p + 1
    num_sh = len(spec._sph_harm_lm)
    
    b_eval = np.array([ np.exp(-r**2) * spec.basis_eval_radial(r,k,l) * spec.basis_eval_spherical(grid[0],grid[1],l,m) for lm_idx, (l,m) in enumerate(spec._sph_harm_lm) for k in range(num_p)])
    b_eval = b_eval.reshape(num_sh,num_p, -1)
    b_eval = np.swapaxes(b_eval,0,1)
    b_eval = b_eval.reshape(num_p * num_sh,-1)
    
    return np.dot(cf, b_eval).reshape(50,100), theta, phi
    



def first_order_split(E : np.array, y0: np.array, cf : colOpSp.CollisionOpSP, t_final : float,  dt :float, v0: np.array):
    """
    E  : electric field for v-space advection 
    cf : collision operator
    dt : operator splitting time scale. 
    v0 : initial mean velocity
    """
    t_initial = 0.0
    t_curr    = 0.0
    spec      = cf._spec
    
    M=spec.compute_mass_matrix()
    Minv = np.linalg.inv(M)
    maxwellian = BEUtils.get_maxwellian_3d(VTH,collisions.MAXWELLIAN_N)

    def collission_rhs(t,y, C, n0):
        return n0 * np.matmul(C,y)

    f_t = y0
    g0  = collisions.eAr_G0()

    ode_solver = ode(collission_rhs,jac=None).set_integrator("dopri5",verbosity=1, rtol = 1e-14, atol=1e-14, nsteps=1e8)
    
    if(np.allclose(E,np.zeros_like(E))):
        "no splitting"

        g0.reset_scattering_direction_sp_mat()
        collission_mat=cf.assemble_mat(g0, maxwellian, VTH, v0)
        collission_mat= np.matmul(Minv,collission_mat)
        ode_solver.set_f_params(collission_mat,collisions.AR_NEUTRAL_N)
        ode_solver.set_initial_value(f_t,t_curr)
        ode_solver.integrate(ode_solver.t + t_final)
        f_t = ode_solver.y
        return [f_t,v0]
    else:
        "do first order operator split" 
        while (t_curr < t_final):
            "A - advection step"
            v0 = v0 - E * dt

            "C - collision step"
            g0.reset_scattering_direction_sp_mat()
            collission_mat=cf.assemble_mat(g0, maxwellian, VTH, v0)
            collission_mat= np.matmul(Minv,collission_mat)
            ode_solver.set_f_params(collission_mat,collisions.AR_NEUTRAL_N)
            ode_solver.set_initial_value(f_t,t_curr + dt)
            ode_solver.integrate(ode_solver.t + dt)

            if(not ode_solver.successful()):
                print("collission operator solve failed\n")
                assert(False)
            
            f_t = ode_solver.y
            
            t_curr+=2*dt

        
        return [f_t,v0]

def second_order_split(e_field, y0: np.array, cf : colOpSp.CollisionOpSP, t_final : float,  dt :float, v0: np.array,mw, vth):
    """
    E  : electric field for v-space advection 
    cf : collision operator
    dt : operator splitting time scale. 
    v0 : initial mean velocity
    """
    t_initial = 0.0
    t_curr    = 0.0
    spec      = cf._spec
    
    M=spec.compute_mass_matrix()
    Minv = np.linalg.pinv(M)
    
    def collission_rhs(t,y, C, n0):
        return n0 * np.matmul(C,y)

    f_t = y0

    g0  = collisions.eAr_G0()
    g2  = collisions.eAr_G2()

    ode_solver = ode(collission_rhs,jac=None).set_integrator("dopri5",verbosity=1, rtol = 1e-14, atol=1e-14, nsteps=1e8)
    ode_solver.set_initial_value(f_t,t_curr)
    
    q_pts_efield = 11
    [xt_1, wt_1] = basis.uniform_simpson( (0,  dt/2), q_pts_efield)
    [xt_2, wt_2] = basis.uniform_simpson( (dt/2, dt), q_pts_efield)
    
    sol_vec=list()
    while (t_curr < t_final):
        sol_vec.append(np.array(ode_solver.y))
        print(v0)
        E = np.dot(e_field(t_curr + xt_1), wt_1)
        "A - advection step"
        v1 = v0 - E 

        "C - collision step"
        g0.reset_scattering_direction_sp_mat()
        collission_mat=cf.assemble_mat(g0, maxwellian, VTH, v1)
        collission_mat= np.matmul(Minv,collission_mat)
        ode_solver.set_f_params(collission_mat,collisions.AR_NEUTRAL_N)
        ode_solver.set_initial_value(f_t,t_curr)
        ode_solver.integrate(ode_solver.t + dt)

        if(not ode_solver.successful()):
            print("collission operator solve failed\n")
            assert(False)
        
        f_t = ode_solver.y
        
        "A- advection step"
        E = np.dot(e_field(t_curr + xt_2), wt_2)
        v0 = v1 - E 
        
        t_curr+=dt
    ts_steps = len(sol_vec)
    return np.array(sol_vec).reshape((ts_steps,-1))

parser = argparse.ArgumentParser()
parser.add_argument("-Nr", "--NUM_P_RADIAL"                   , help="Number of polynomials in radial direction", nargs='+', type=int, default=[4,8,16,32,64,128,256])
parser.add_argument("-T", "--T_END"                           , help="Simulation time", type=float, default=1e-6)
parser.add_argument("-dt", "--T_DT"                           , help="Simulation time step size ", type=float, default=1e-10)
parser.add_argument("-o",  "--out_fname"                      , help="output file name", type=str, default='coll_op')
parser.add_argument("-ts_tol", "--ts_tol"                     , help="adaptive timestep tolerance", type=float, default=1e-15)
parser.add_argument("-l_max", "--l_max"                       , help="max polar modes in SH expansion", type=int, default=0)
parser.add_argument("-c", "--collision_mode"                  , help="collision mode g- elastic with no E loss, g0-elastic", type=str, default="g0")
parser.add_argument("-ev", "--electron_volt"                  , help="initial electron volt", type=float, default=1.0)
parser.add_argument("-q_vr", "--quad_radial"                  , help="quadrature in r"        , type=int, default=270)
parser.add_argument("-q_vt", "--quad_theta"                   , help="quadrature in polar"    , type=int, default=2)
parser.add_argument("-q_vp", "--quad_phi"                     , help="quadrature in azimuthal", type=int, default=2)
parser.add_argument("-q_st", "--quad_s_theta"                 , help="quadrature in scattering polar"    , type=int, default=2)
parser.add_argument("-q_sp", "--quad_s_phi"                   , help="quadrature in scattering azimuthal", type=int, default=2)
parser.add_argument("-radial_poly", "--radial_poly"           , help="radial basis", type=str, default="maxwell")
parser.add_argument("-sp_order", "--spline_order"             , help="b-spline order", type=int, default=2)
parser.add_argument("-spline_qpts", "--spline_q_pts_per_knot" , help="q points per knots", type=int, default=11)
parser.add_argument("-vth_fac", "--vth_fac" , help="expand the function vth_fac *V_TH", type=float, default=1.0)
args = parser.parse_args()
print(args)
run_data=list()
ev           = np.linspace(args.electron_volt/50.,100.*args.electron_volt,1000)
eedf         = np.zeros((len(args.NUM_P_RADIAL),len(ev)))
SPLINE_ORDER = args.spline_order
basis.BSPLINE_BASIS_ORDER=SPLINE_ORDER
for i, nr in enumerate(args.NUM_P_RADIAL):
    params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER = nr
    params.BEVelocitySpace.SPH_HARM_LM = [[i,j] for i in range(args.l_max+1) for j in range(-i,i+1)]
    if (args.radial_poly == "maxwell"):
        r_mode = basis.BasisType.MAXWELLIAN_POLY
        params.BEVelocitySpace.NUM_Q_VR  = args.quad_radial
        
    elif (args.radial_poly == "laguerre"):
        r_mode = basis.BasisType.LAGUERRE
        params.BEVelocitySpace.NUM_Q_VR  = args.quad_radial

    elif (args.radial_poly == "bspline"):
        r_mode = basis.BasisType.SPLINES
        params.BEVelocitySpace.NUM_Q_VR  = basis.BSpline.get_num_q_pts(params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER, SPLINE_ORDER, basis.XLBSPLINE_NUM_Q_PTS_PER_KNOT)

    params.BEVelocitySpace.NUM_Q_VT  = args.quad_theta
    params.BEVelocitySpace.NUM_Q_VP  = args.quad_phi
    params.BEVelocitySpace.NUM_Q_CHI = args.quad_s_theta
    params.BEVelocitySpace.NUM_Q_PHI = args.quad_s_phi
    params.BEVelocitySpace.VELOCITY_SPACE_DT = args.T_DT

    INIT_EV                        = args.electron_volt
    collisions.MAXWELLIAN_TEMP_K   = INIT_EV * collisions.TEMP_K_1EV
    collisions.ELECTRON_THEMAL_VEL = collisions.electron_thermal_velocity(collisions.MAXWELLIAN_TEMP_K) 
    cf    = colOpSp.CollisionOpSP(params.BEVelocitySpace.VELOCITY_SPACE_DIM,params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER,poly_type=r_mode)
    spec  = cf._spec
    VTH   = collisions.ELECTRON_THEMAL_VEL
    
    op_split_dt     = args.v_cfl * args.v_dv/ np.linalg.norm(np.array(args.e_amp))

    print("""===========================Parameters ======================""")
    print("\tMAXWELLIAN_N : ", collisions.MAXWELLIAN_N)
    print("\tELECTRON_THEMAL_VEL : ", VTH," ms^-1")
    #print("\tDT : ", params.BEVelocitySpace().VELOCITY_SPACE_DT, " s")
    print("\ttimestep size : ", op_split_dt, " s")
    print("""============================================================""")
    
    params.print_parameters()
    
    vth_factor = 1.0 #+ 1e-2
    VTH_C      = vth_factor * VTH
    maxwellian = BEUtils.get_maxwellian_3d(VTH_C, collisions.MAXWELLIAN_N)
    #hv         = lambda v,vt,vp : np.ones_like(v) +  np.tan(vt)  
    hv         = lambda v,vt,vp : (vth_factor**3) * np.exp(v**2  -(v*vth_factor)**2)
    h_vec      = BEUtils.function_to_basis(spec,hv,maxwellian,None,None,None)
    #e_field    = lambda t : np.array([args.e_amp[i] * np.sin(2* np.pi * args.e_freq * t) for i in range(3)])
    e_field    = lambda t : np.array([args.e_amp[i] * np.ones_like(t) for i in range(3)])
    
    v0              = np.array([0,0,0])
    data            = second_order_split(e_field, h_vec, cf, args.T_END, op_split_dt, v0,maxwellian, VTH_C)
    eedf[i]         = BEUtils.get_eedf(ev, spec, data[-1,:], maxwellian, VTH_C, 1)
    eedf_initial    = BEUtils.get_eedf(ev, spec, data[0,:], maxwellian, VTH_C, 1)
    run_data.append(data)
    
import matplotlib.pyplot as plt
if (1):
    fig = plt.figure(figsize=(12, 4), dpi=300)
    plt.subplot(1, 5, 1)
    for i, nr in enumerate(args.NUM_P_RADIAL):
        data=run_data[i]
        plt.plot(abs(data[1,:]),label="Nr=%d"%args.NUM_P_RADIAL[i])

    plt.yscale('log')
    plt.xlabel("Coeff. #")
    plt.ylabel("Coeff. magnitude")
    plt.grid()
    plt.legend()

    plt.subplot(1, 5, 2)
    for i, nr in enumerate(args.NUM_P_RADIAL):
        data=run_data[i]
        plt.plot(abs(data[-1,:]),label="Nr=%d"%args.NUM_P_RADIAL[i])

    plt.yscale('log')
    plt.xlabel("Coeff. #")
    plt.ylabel("Coeff. magnitude")
    plt.grid()
    plt.legend()

    plt.subplot(1, 5, 3)
    for i, nr in enumerate(args.NUM_P_RADIAL):
        data_last=run_data[-1]
        data=run_data[i]
        plt.plot(abs(data[-1,:] - data_last[-1,0:len(data[-1,:])]),label="Nr=%d"%args.NUM_P_RADIAL[i])

    plt.yscale('log')
    plt.xlabel("Coeff. #")
    plt.ylabel("Error in coeff. magnitude")
    plt.grid()
    plt.legend()

    ts= np.linspace(0,args.T_END, run_data[i].shape[0])
    plt.subplot(1, 5, 4)
    for i, nr in enumerate(args.NUM_P_RADIAL):
        plt.plot(ts, spec_tail_timeseries(run_data[i],nr, len(params.BEVelocitySpace.SPH_HARM_LM)),label="Nr=%d"%args.NUM_P_RADIAL[i])

    plt.legend()
    plt.yscale('log')
    plt.xlabel("time (s)")
    plt.ylabel("spectral tail l2(h[nr/2: ])")
    plt.grid()


    plt.subplot(1, 5, 5)
    plt.plot(ev, eedf_initial, label="initial")
    for i, nr in enumerate(args.NUM_P_RADIAL):
        data=run_data[i]
        plt.plot(ev, abs(eedf[i]),label="Nr=%d"%args.NUM_P_RADIAL[i])

    # plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.xlabel("energy (ev)")
    plt.ylabel("eedf")
    # plt.tight_layout()
    plt.grid()

    #plt.show()
    plt.savefig("%s_coeff.png"%(args.out_fname))
    
fig = plt.figure(figsize=(10, 4), dpi=300)

if (args.radial_poly == "maxwell"):
    ts= np.linspace(0,args.T_END, run_data[i].shape[0])
    plt.subplot(1, 2, 1)
    for i, nr in enumerate(args.NUM_P_RADIAL):
        plt.plot(ts, spec_tail_timeseries(run_data[i],nr+1, len(params.BEVelocitySpace.SPH_HARM_LM)),label="Nr=%d"%args.NUM_P_RADIAL[i])

    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel("time (s)")
    plt.ylabel("tail l2(h[nr/2: ])")

elif (args.radial_poly == "bspline"):
    plt.subplot(1, 2, 1)
    for i, nr in enumerate(args.NUM_P_RADIAL):
        plt.plot(ev, np.abs(eedf[i]-eedf[-1]),label="Nr=%d"%args.NUM_P_RADIAL[i])

    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.xlabel("energy (ev)")
    plt.ylabel("error in eedf")

plt.subplot(1, 2, 2)
plt.plot(ev, eedf_initial, label="initial")
for i, nr in enumerate(args.NUM_P_RADIAL):
    data=run_data[i]
    plt.plot(ev, eedf[i],label="Nr=%d"%(nr))

#plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.xlabel("energy (ev)")
plt.ylabel("eedf")
plt.tight_layout()

#plt.show()
plt.savefig("%s.png"%(args.out_fname))
plt.close()

fig = plt.figure(figsize=(10, 4), dpi=300)
plt.subplot(1,2,1)
polar_plot, _ , __ = constant_r_eval(spec,run_data[-1][0,:],1)
plt.imshow(polar_plot,extent=[0,2*np.pi,0,np.pi])
plt.colorbar()
plt.xlabel("azimuthal angle")
plt.ylabel("polar angle")
plt.title("initial, v_r = 1")

plt.subplot(1,2,2)
polar_plot, _ , __ = constant_r_eval(spec,run_data[-1][-1,:],1)
plt.imshow(polar_plot,extent=[0,2*np.pi,0,np.pi])
plt.colorbar()
plt.clim(np.min(polar_plot),np.max(polar_plot)*1.01)
plt.xlabel("azimuthal angle")
plt.ylabel("polar angle")
plt.title("final, v_r = 1")
plt.savefig("%s_const_r.png"%(args.out_fname))