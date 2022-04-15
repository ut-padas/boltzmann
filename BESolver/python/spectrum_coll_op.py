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

collisions.AR_NEUTRAL_N=3.22e22
collisions.MAXWELLIAN_N=1e18
collisions.AR_IONIZED_N=collisions.MAXWELLIAN_N
L_MAX=1

q_mode = sp.QuadMode.GMX
r_mode = basis.BasisType.MAXWELLIAN_POLY
params.BEVelocitySpace.NUM_Q_VR  = 300

params.BEVelocitySpace.NUM_Q_VT  = 4
params.BEVelocitySpace.NUM_Q_VP  = 4
params.BEVelocitySpace.NUM_Q_CHI = 2
params.BEVelocitySpace.NUM_Q_PHI = 2
params.BEVelocitySpace.SPH_HARM_LM = [[i,j] for i in range(L_MAX) for j in range(0,i+1)]
print(params.BEVelocitySpace.SPH_HARM_LM)


collisions.MAXWELLIAN_TEMP_K   = collisions.TEMP_K_1EV
collisions.ELECTRON_THEMAL_VEL = collisions.electron_thermal_velocity(collisions.MAXWELLIAN_TEMP_K) 
VTH   = collisions.ELECTRON_THEMAL_VEL
maxwellian = BEUtils.get_maxwellian_3d(VTH,collisions.MAXWELLIAN_N)

Nr=[32,64,128]
coll_mats=list()
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(6, 8)) #(figsize=(6, 6), dpi=300)

for i, nr in enumerate(Nr):
    r_mode = basis.BasisType.MAXWELLIAN_POLY
    params.BEVelocitySpace.NUM_Q_VR  = 300
    params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER=nr
    params.print_parameters()
    
    cf    = colOpSp.CollisionOpSP(params.BEVelocitySpace.VELOCITY_SPACE_DIM, nr,q_mode,r_mode)
    spec  = cf._spec
    mm    = spec.compute_mass_matrix()
    mm    = np.linalg.inv(mm)

    g0  = collisions.eAr_G0()
    g0.reset_scattering_direction_sp_mat()
    t1=time()
    FOp = cf.assemble_mat(g0,maxwellian,VTH)
    t2=time()
    FOp = np.matmul(mm, FOp)
    #print("Assembled the collision op. for Vth : ", VTH)
    #print("Collision Operator assembly time (s): ",(t2-t1))
    #coll_mats.append(FOp)
    u, s, v = np.linalg.svd(FOp)
    # pt_c=1
    # for k in range(Nr[0]):
    #     plt.subplot(Nr[0], 2, pt_c)
    #     plt.plot(u[:,k],label="L Nr=%d"%(nr))
    #     #plt.legend()

    #     plt.subplot(Nr[0], 2, pt_c+1)
    #     plt.plot(v[k,:],label="R Nr=%d"%(nr))
    #     #plt.legend()

    #     pt_c+=2

    plt.plot(s,label="Nr=%d"%(nr))

plt.legend()
plt.tight_layout()
plt.show()

plt.close()

SPLINE_ORDER=2
basis.BSPLINE_BASIS_ORDER=SPLINE_ORDER
basis.XLBSPLINE_NUM_Q_PTS_PER_KNOT=11

for i, nr in enumerate(Nr):
    r_mode = basis.BasisType.SPLINES
    params.BEVelocitySpace.NUM_Q_VR  = basis.BSpline.get_num_q_pts(nr, SPLINE_ORDER, basis.XLBSPLINE_NUM_Q_PTS_PER_KNOT)
    
    params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER=nr
    params.print_parameters()

    cf    = colOpSp.CollisionOpSP(params.BEVelocitySpace.VELOCITY_SPACE_DIM,nr,q_mode,r_mode)
    spec  = cf._spec
    mm    = spec.compute_mass_matrix()
    mm    = np.linalg.inv(mm)

    g0  = collisions.eAr_G0()
    g0.reset_scattering_direction_sp_mat()
    t1=time()
    FOp = cf.assemble_mat(g0,maxwellian,VTH)
    t2=time()
    FOp = np.matmul(mm, FOp)
    # r_mode = basis.BasisType.MAXWELLIAN_POLY
    # params.BEVelocitySpace.NUM_Q_VR  = 300
    u, s, v = np.linalg.svd(FOp)
    # pt_c=1
    # for k in range(Nr[0]):
    #     plt.subplot(Nr[0], 2, pt_c)
    #     plt.plot(u[:,k],label="L Nr=%d"%(nr))
    #     #plt.legend()

    #     plt.subplot(Nr[0], 2, pt_c+1)
    #     plt.plot(v[k,:],label="R Nr=%d"%(nr))
    #     #plt.legend()

    #     pt_c+=2

    plt.plot(s,label="Nr=%d"%(nr))

plt.legend()
plt.tight_layout()
plt.show()




# import matplotlib.pyplot as plt
# for i,c_mat in enumerate(coll_mats):
#     print("Nr=%d condition number=%.8E"%(Nr[i],np.linalg.cond(c_mat)))
#     u, s , v = np.linalg.svd(3e22*c_mat)
#     plt.plot(s,label="Nr=%d"%Nr[i])

# plt.yscale('log')
# plt.legend()
# plt.show()

# plt.close()
# for i,c_mat in enumerate(coll_mats):
#     print("Nr=%d condition number=%.8E"%(Nr[i],np.linalg.cond(c_mat)))
#     u, s , v = np.linalg.svd(c_mat)
#     plt.plot(v[:,0],label="Nr=%d"%Nr[i])

# plt.legend()
# plt.show()

# c_mat = coll_mats[2]
# u, s , v = np.linalg.svd(c_mat)
# k_off    = 10
# c_mat_p = np.matmul(u[:,0:k_off] , np.matmul(np.diag(s[0:k_off]), np.transpose(v[:,0:k_off])))  
# print(np.linalg.norm(c_mat-c_mat_p))
# np.linalg.cond(c_mat_p)




