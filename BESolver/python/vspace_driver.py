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

parser = argparse.ArgumentParser()
parser.add_argument("-Nr", "--NUM_P_RADIAL", help="Number of polynomials in radial direction", nargs='+', type=int, default=[4,8,16,32,64])
parser.add_argument("-T ", "--T", help="Simulation time", type=float, default=1e-6)
parser.add_argument("-dt", "--dt", help="split timestep",  type=float, default=1e-5)
parser.add_argument("-ts_tol", "--ts_tol", help="adaptive timestep tolerance", type=float, default=1e-15)
parser.add_argument("-c", "--collision_mode", help="collision mode", type=str, default="g0")
parser.add_argument("-ev", "--electron_volt", help="initial electron volt", type=float, default=1.0)
parser.add_argument("-o", "--f_prefix", help="file prefix for plots", type=str, default="vspace_")
args = parser.parse_args()


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

def second_order_split(E : np.array, y0: np.array, cf : colOpSp.CollisionOpSP, t_final : float,  dt :float, v0: np.array):
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
    g2  = collisions.eAr_G2()

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
        "do second order operator split" 
        while (t_curr < t_final):
            "A - advection step"
            v1 = v0 - E * dt/2

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
            v0 = v1 - E * dt/2
            
            t_curr+=dt

        
        return [f_t,v0]


E  = np.array([-1,-1,-1]) * 100 #* collisions.ELECTRON_CHARGE /collisions.MASS_ELECTRON
ev = np.linspace(0,50, 100)
b_eedf_0 = np.zeros((len(args.NUM_P_RADIAL), len(ev)))
b_eedf_t = np.zeros((len(args.NUM_P_RADIAL), len(ev)))
b_eedf_t_e0 = np.zeros((len(args.NUM_P_RADIAL), len(ev)))
for run_k, nr in enumerate(args.NUM_P_RADIAL):
    # set the collision parameters. 
    collisions.AR_NEUTRAL_N=3.22e22
    collisions.MAXWELLIAN_N=1e18
    collisions.AR_IONIZED_N=collisions.MAXWELLIAN_N
    INIT_EV    = args.electron_volt
    collisions.MAXWELLIAN_TEMP_K   = INIT_EV * collisions.TEMP_K_1EV
    collisions.ELECTRON_THEMAL_VEL = collisions.electron_thermal_velocity(collisions.MAXWELLIAN_TEMP_K) 

    params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER = nr #args.NUM_P_RADIAL[0]
    params.BEVelocitySpace.SPH_HARM_LM = [[i,j] for i in range(1) for j in range(-i,i+1)]

    q_mode = sp.QuadMode.SIMPSON
    r_mode = basis.BasisType.SPLINES
    basis.BSPLINE_NUM_Q_PTS_PER_KNOT = 3
    basis.BSPLINE_BASIS_ORDER        = 1
    params.BEVelocitySpace.NUM_Q_VR  = basis.BSpline.get_num_q_pts(params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER,basis.BSPLINE_BASIS_ORDER,basis.BSPLINE_NUM_Q_PTS_PER_KNOT)

    # q_mode = sp.QuadMode.GMX
    # r_mode = basis.BasisType.MAXWELLIAN_POLY
    # params.BEVelocitySpace.NUM_Q_VR  = 300

    params.BEVelocitySpace.NUM_Q_VT  = 2
    params.BEVelocitySpace.NUM_Q_VP  = 2
    params.BEVelocitySpace.NUM_Q_CHI = 2
    params.BEVelocitySpace.NUM_Q_PHI = 2
    print("parameters : ", args)


    VTH   = collisions.ELECTRON_THEMAL_VEL 
    print("""===========================Parameters ======================""")
    print("\tMAXWELLIAN_N : ", collisions.MAXWELLIAN_N)
    print("\tELECTRON_THEMAL_VEL : ", VTH," ms^-1")
    print("\tDT : ", args.dt, " s")
    print("\tT : ",  args.T, " s")
    print("""============================================================""")
    params.print_parameters()

    cf    = colOpSp.CollisionOpSP(params.BEVelocitySpace.VELOCITY_SPACE_DIM,params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER,q_mode,r_mode)
    spec  = cf._spec
    print("E field : ", E)
    v0 = np.array([0,0,0])
    maxwellian = BEUtils.get_maxwellian_3d(VTH,collisions.MAXWELLIAN_N)
    hv         = lambda v,vt,vp : np.ones_like(v)
    f0         = BEUtils.compute_func_projection_coefficients(spec,hv,maxwellian,None,None,None)

    [ft,vt]         = second_order_split(E, f0, cf  ,args.T, args.dt, v0)


    b_eedf_0 [run_k,:]     = BEUtils.get_eedf(ev,spec,f0,maxwellian,VTH)
    b_eedf_t [run_k,:]     = BEUtils.get_eedf(ev,spec,ft,maxwellian,VTH)
    #eedf_zero_e[run_k,:] = BEUtils.get_eedf(ev,spec,ft_e0,maxwellian,VTH)

    if nr == args.NUM_P_RADIAL[-1]:
        [ft_e0,vt_e0]          = second_order_split(np.zeros(3), f0, cf  ,args.T, args.dt, np.zeros(3))
        b_eedf_t_e0[run_k,:]     = BEUtils.get_eedf(ev,spec,ft_e0,maxwellian,VTH)



m_eedf_0 = np.zeros((len(args.NUM_P_RADIAL), len(ev)))
m_eedf_t = np.zeros((len(args.NUM_P_RADIAL), len(ev)))
m_eedf_t_e0 = np.zeros((len(args.NUM_P_RADIAL), len(ev)))
for run_k, nr in enumerate(args.NUM_P_RADIAL):
    # set the collision parameters. 
    collisions.AR_NEUTRAL_N=3.22e22
    collisions.MAXWELLIAN_N=1e18
    collisions.AR_IONIZED_N=collisions.MAXWELLIAN_N
    INIT_EV    = args.electron_volt
    collisions.MAXWELLIAN_TEMP_K   = INIT_EV * collisions.TEMP_K_1EV
    collisions.ELECTRON_THEMAL_VEL = collisions.electron_thermal_velocity(collisions.MAXWELLIAN_TEMP_K) 

    params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER = nr #args.NUM_P_RADIAL[0]
    params.BEVelocitySpace.SPH_HARM_LM = [[i,j] for i in range(1) for j in range(-i,i+1)]

    # q_mode = sp.QuadMode.SIMPSON
    # r_mode = basis.BasisType.SPLINES
    # basis.BSPLINE_NUM_Q_PTS_PER_KNOT = 3
    # basis.BSPLINE_BASIS_ORDER        = 1
    # params.BEVelocitySpace.NUM_Q_VR  = basis.BSpline.get_num_q_pts(params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER,basis.BSPLINE_BASIS_ORDER,basis.BSPLINE_NUM_Q_PTS_PER_KNOT)

    q_mode = sp.QuadMode.GMX
    r_mode = basis.BasisType.MAXWELLIAN_POLY
    params.BEVelocitySpace.NUM_Q_VR  = 300

    params.BEVelocitySpace.NUM_Q_VT  = 2
    params.BEVelocitySpace.NUM_Q_VP  = 2
    params.BEVelocitySpace.NUM_Q_CHI = 2
    params.BEVelocitySpace.NUM_Q_PHI = 2
    print("parameters : ", args)


    VTH   = collisions.ELECTRON_THEMAL_VEL 
    print("""===========================Parameters ======================""")
    print("\tMAXWELLIAN_N : ", collisions.MAXWELLIAN_N)
    print("\tELECTRON_THEMAL_VEL : ", VTH," ms^-1")
    print("\tDT : ", args.dt, " s")
    print("\tT : ",  args.T, " s")
    print("""============================================================""")
    params.print_parameters()

    cf    = colOpSp.CollisionOpSP(params.BEVelocitySpace.VELOCITY_SPACE_DIM,params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER,q_mode,r_mode)
    spec  = cf._spec
    print("E field : ", E)
    v0 = np.array([0,0,0])
    maxwellian = BEUtils.get_maxwellian_3d(VTH,collisions.MAXWELLIAN_N)
    hv         = lambda v,vt,vp : np.ones_like(v)
    f0         = BEUtils.compute_func_projection_coefficients(spec,hv,maxwellian,None,None,None)

    [ft,vt]         = second_order_split(E, f0, cf  ,args.T, args.dt, v0)


    m_eedf_0 [run_k,:]     = BEUtils.get_eedf(ev,spec,f0,maxwellian,VTH)
    m_eedf_t [run_k,:]     = BEUtils.get_eedf(ev,spec,ft,maxwellian,VTH)
    #eedf_zero_e[run_k,:] = BEUtils.get_eedf(ev,spec,ft_e0,maxwellian,VTH)

    if nr == args.NUM_P_RADIAL[-1]:
        [ft_e0,vt_e0]          = second_order_split(np.zeros(3), f0, cf  ,args.T, args.dt, np.zeros(3))
        m_eedf_t_e0[run_k,:]     = BEUtils.get_eedf(ev,spec,ft_e0,maxwellian,VTH)



import matplotlib.pyplot as plt

plt.figure(1)
for run_k, nr in enumerate(args.NUM_P_RADIAL):
    plt.plot(ev,      b_eedf_t[run_k,:]  , '-', label="Nr=%d"%(nr))
    #print(eedf_t[run_k])
    #plt.plot(ev, eedf_zero_e[run_k, : ]  , 'y-o', label="Nr=%d"%(nr))

plt.yscale("log")
plt.xscale("log")
plt.grid()
plt.plot(ev,     b_eedf_t_e0 [-1, : ] , 'y--', label="E=0")   
plt.plot(ev,     b_eedf_0 [-1, : ]    , 'b--', label="t=0")
plt.xlabel("energy (ev)")
plt.ylabel("EEDF")   
plt.title("T=%.2E E=%s" %(args.T, E))
plt.legend()
fig = plt.gcf()
fig.set_size_inches(5, 5)
fig.savefig("%s_bspline_eedf.png"%args.f_prefix, dpi=300)
plt.show()
plt.close()

plt.figure(2)
for run_k, nr in enumerate(args.NUM_P_RADIAL):
    if(run_k < len(args.NUM_P_RADIAL)-1):
        plt.plot(ev, np.abs(b_eedf_t[run_k,:] - b_eedf_t[-1,:])/np.linalg.norm(b_eedf_t[-1,:]), '--', label="Nr=%d"%(nr))

plt.yscale("log")
plt.xscale("log")
plt.xlabel("energy (ev)")
plt.ylabel("error")
plt.grid()
plt.legend()
plt.title("T=%.2E E=%s" %(args.T, E))
fig = plt.gcf()
fig.set_size_inches(5, 5)
fig.savefig("%s_bspline_eedf_conv.png"%args.f_prefix, dpi=300)
plt.show()
plt.close()

plt.figure(3)
for run_k, nr in enumerate(args.NUM_P_RADIAL):
    plt.plot(ev,      m_eedf_t[run_k,:]  , '-', label="Nr=%d"%(nr))

plt.yscale("log")
plt.xscale("log")
plt.xlabel("energy (ev)")
plt.ylabel("EEDF")
plt.grid()
plt.plot(ev,     m_eedf_t_e0 [-1, : ] , 'y--', label="E=0")   
plt.plot(ev,     m_eedf_0 [-1, : ]    , 'b--', label="t=0")   
plt.title("T=%.2E E=%s" %(args.T, E))
plt.legend()
fig = plt.gcf()
fig.set_size_inches(5, 5)
fig.savefig("%s_maxwell_eedf.png"%args.f_prefix, dpi=300)
plt.show()
plt.close()

plt.figure(4)
for run_k, nr in enumerate(args.NUM_P_RADIAL):
    if(run_k < len(args.NUM_P_RADIAL)-1):
        plt.plot(ev, np.abs(m_eedf_t[run_k,:] - m_eedf_t[-1,:])/np.linalg.norm(m_eedf_t[-1,:]), '--', label="Nr=%d"%(nr))

plt.yscale("log")
plt.xscale("log")
plt.xlabel("energy (ev)")
plt.ylabel("error")
plt.grid()
plt.legend()
plt.title("T=%.2E E=%s" %(args.T, E))
fig = plt.gcf()
fig.set_size_inches(5, 5)
fig.savefig("%s_maxwell_eedf_conv.png"%args.f_prefix, dpi=300)
plt.show()
plt.close()


plt.figure(5)
plt.plot(ev,      m_eedf_0[-1,:]  , '-', label="t=0")
plt.plot(ev,      m_eedf_t[-1,:]  , '-', label="Nr=%d maxwell"%(nr))
plt.plot(ev,      b_eedf_t[-1,:]  , '-', label="Nr=%d bspline"%(nr))
plt.yscale("log")
plt.xscale("log")
plt.xlabel("energy (ev)")
plt.ylabel("EEDF")
plt.grid()
plt.legend()
plt.title("T=%.2E E=%s" %(args.T, E))
fig = plt.gcf()
fig.set_size_inches(5, 5)
fig.savefig("%s_bspline_vs_maxwell.png"%args.f_prefix, dpi=300)
plt.show()
plt.close()









     











        
















