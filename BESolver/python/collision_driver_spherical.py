"""
@package Boltzmann collision operator solver. 
"""

import enum
import basis
import spec_spherical as sp
import numpy as np
import collision_operator_spherical as colOpSp
import collisions 
import parameters as params
import os
import ets
import utils
import profiler

t_M = profiler.profile_t("mass_assembly")
t_L = profiler.profile_t("collOp_assembly")

if not os.path.exists('plots'):
    print("creating folder `plots`, output will be written into it")
    os.makedirs('plots')

def ode_first_order_linear(spec_p, FOp, h_init, maxwellian, ts):
    MNE   = collisions.MAXWELLIAN_N
    MTEMP = collisions.MAXWELLIAN_TEMP_K
    MVTH  = collisions.ELECTRON_THEMAL_VEL

    W,Q  = np.linalg.eig(FOp)
    print("Eigenvalues: ")
    print(W)
    Qinv = np.linalg.inv(Q)
    
    y_init = np.dot(Qinv,h_init)

    num_p  = spec_p._p + 1
    num_sh = len(spec_p._sph_harm_lm)

    q_norm=np.array([np.linalg.norm(Q[:,i]) for i in range(num_p*num_sh)])
    print("Eigenvectors normalized : ", np.allclose(q_norm,np.ones_like(q_norm)))

    fun_sol = np.array([y_init[i] * Q[:,i] for i in range(num_p*num_sh)])
    #print(fun_sol)
    fun_sol = np.transpose(fun_sol)

    #M0=np.array([utils.moment_zero_f(spec_p,fun_sol[:,i],maxwellian,None,None,None,1) for i in range(num_p*num_sh)])
    #M2=np.array([utils.compute_avg_temp(collisions.MASS_ELECTRON,spec_p,fun_sol[:,i],maxwellian,None,None,None,1) for i in range(num_p*num_sh)])
    
    print("Scaled for the initial condition (t=0) : " ,np.allclose(fun_sol.sum(axis=1),h_init))
    
    dt_tau = 1/params.PLASMA_FREQUENCY
    print("==========================================================================")
    VTH_t = float(MVTH)
    
    mass_ts=np.zeros_like(ts)
    temp_ts=np.zeros_like(ts)

    for t_i, t in enumerate(ts):
        c_t = np.exp(W*t)
        #print(c_t)
        h_t = np.dot(fun_sol,c_t)
        #print(" imag part close to zero : ", np.allclose(np.imag(h_t),np.zeros(h_t.shape)))
        assert np.allclose(np.imag(h_t),np.zeros(h_t.shape)) , "imaginary part is not close to zero"
        maxwellian_t = lambda x: (collisions.MAXWELLIAN_N / ((VTH_t * np.sqrt(np.pi))**3) ) * np.exp(-x**2)

        h_t   = np.real(h_t)
        #h_t   = h_t*(maxwellian_t(0)/maxwellian(0))
        m0   = utils.moment_zero_f(spec_p,h_t,maxwellian,MVTH,None,None,None,1)
        temp = utils.compute_avg_temp(collisions.MASS_ELECTRON,spec_p,h_t,maxwellian,MVTH,None,None,None,m0,1)
        VTH_t = collisions.thermal_velocity(temp)
        mass_ts[t_i] = (MNE-m0)/MNE
        temp_ts[t_i] = temp
        #print(temp/collisions.MAXWELLIAN_N)
        #mt=np.dot(M0,c_t)
        print(f"N_e(t={t:.2E})={m0:.10E} maxwellian N_e={MNE:.2E} relative error: {(MNE-m0)/MNE:.8E} t/dt={t/dt_tau:.2E} temperature(K)={temp:.5E} maxwellian temp(K)={MTEMP:.5E}")

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(ts,mass_ts,'b-o')
    axs[0].set_title("n(0)= %.2E, dt=%.2E" %(MNE,dt_tau))
    axs[0].set_xlabel('time (s)')
    axs[0].set_ylabel('mass loss (n(t)-n(0))/n(0)')

    axs[1].plot(ts,temp_ts,'b-o')
    axs[1].set_title("T(0)= %.2E K, dt=%.2E" %(MTEMP,dt_tau))
    axs[1].set_xlabel('time (s)')
    axs[1].set_ylabel('Temperature (K) ')
    plt.tight_layout()
    plt.show()
    plt.close()
    

params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER = 8
params.BEVelocitySpace.SPH_HARM_LM = [[0,0],[1,0]]
params.BEVelocitySpace.NUM_Q_VR  = 51
params.BEVelocitySpace.NUM_Q_VT  = 10
params.BEVelocitySpace.NUM_Q_VP  = 10
params.BEVelocitySpace.NUM_Q_CHI = 64
params.BEVelocitySpace.NUM_Q_PHI = 10
colOpSp.SPEC_SPHERICAL  = sp.SpectralExpansionSpherical(params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER,basis.Maxwell(),params.BEVelocitySpace.SPH_HARM_LM) 

# instance of the collision operator
cf    = colOpSp.CollisionOpSP(params.BEVelocitySpace.VELOCITY_SPACE_DIM,params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER)
spec  = colOpSp.SPEC_SPHERICAL


print("""===========================Parameters ======================""")
print("\tMAXWELLIAN_N : ", collisions.MAXWELLIAN_N)
print("\tELECTRON_THEMAL_VEL : ", collisions.ELECTRON_THEMAL_VEL," ms^-1")
print("\tDT : ", params.BEVelocitySpace().VELOCITY_SPACE_DT, " s")
print("""============================================================""")
params.print_parameters()

#h_vec    = np.zeros(spec.get_num_coefficients())
#h_vec[0] = 2*np.sqrt(np.pi)
maxwellian = lambda x: (collisions.MAXWELLIAN_N / ((collisions.ELECTRON_THEMAL_VEL * np.sqrt(np.pi))**3) ) * np.exp(-x**2)
hv         = lambda v,vt,vp : np.ones_like(v)
h_vec      = utils.compute_coefficients(spec,hv,maxwellian,None,None,None)
#print(h_vec)



col_g0 = collisions.eAr_G0()
col_g0_no_E_loss = collisions.eAr_G0_NoEnergyLoss()
#col_g1 = collisions.eAr_G1()
#col_g2 = collisions.eAr_G2()

t_M.start()
M  = spec.compute_maxwellian_mm(maxwellian,collisions.ELECTRON_THEMAL_VEL)
t_M.stop()
print("Mass assembly time (s): ", t_M.seconds)

t_L.start()
L_g0  = cf.assemble_mat(col_g0,maxwellian)
L_g0p  = cf.assemble_mat(col_g0_no_E_loss,maxwellian)
#L_g1  = cf.assemble_mat(col_g1,maxwellian)
#L_g2  = cf.assemble_mat(col_g2,maxwellian)
#L     = L_g0p
t_L.stop()
print("Collision Op time (s): ", t_L.seconds)

invMLg0 = np.matmul(np.linalg.inv(M) , L_g0)
invMLg0p = np.matmul(np.linalg.inv(M) , L_g0p)
#print("M^-1 L : \n")
#print(invML)
ode_first_order_linear(spec,invMLg0,h_vec, maxwellian, np.linspace(0,1e-6,20))
ode_first_order_linear(spec,invMLg0p,h_vec, maxwellian, np.linspace(0,1e-3,20))




