"""
@package Boltzmann collision operator solver. 
"""

import basis
import spec_spherical as sp
import numpy as np
import math
import collision_operator_spherical as colOpSp
import collisions 
import parameters as params
import visualize_utils
import ets 
import os
import matplotlib.pyplot as plt
# instance of the collision operator
cf    = colOpSp.CollisionOpSP(params.BEVelocitySpace.VELOCITY_SPACE_DIM,params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER)
spec  = colOpSp.SPEC_SPHERICAL

# # solution coefficients
#h_vec = spec.create_vec()
#h_vec = np.random.rand(len(h_vec))

# intially set to Maxwellian
# h_vec  = np.ones(spec.get_num_coefficients())
# h_vec  = h_vec/1e2 
h_vec  = np.zeros(spec.get_num_coefficients())
h_vec[0] = 1
plot_domain = np.array([[-3,3],[-3,3]])

#maxwellian = basis.Maxwell().Wx()
print("""===========================Parameters ======================""")
print("\tMAXWELLIAN_N : ", collisions.MAXWELLIAN_N)
print("\tELECTRON_THEMAL_VEL : ", collisions.ELECTRON_THEMAL_VEL," ms^-1")
print("\tDT : ", params.BEVelocitySpace().VELOCITY_SPACE_DT, " s")
print("""============================================================""")

maxwellian = lambda x: (collisions.MAXWELLIAN_N / ((collisions.ELECTRON_THEMAL_VEL * np.sqrt(np.pi))**3) ) * np.exp(-x**2)
#maxwellian = lambda x: (1/ ((np.sqrt(np.pi))**3) ) * np.exp(-x**2)
col_g0 = collisions.eAr_G0()
col_g0_no_E_loss = collisions.eAr_G0_NoEnergyLoss()
col_g1 = collisions.eAr_G1()
col_g2 = collisions.eAr_G2()

M  = spec.compute_maxwellian_mm(maxwellian,collisions.ELECTRON_THEMAL_VEL)

print("Mass matrix w.r.t. Maxwellian\n")
print(M)

L_g0  = cf.assemble_mat(col_g0,maxwellian)
#L_g0p  = cf.assemble_mat(col_g0_no_E_loss,maxwellian)
L_g1  = cf.assemble_mat(col_g1,maxwellian)
#L_g2  = cf.assemble_mat(col_g2,maxwellian)

L     = L_g0 + L_g1# + L_g2 #L_g0 #L_g0p #L_g0 #+ L_g2 #L_g0 + L_g1 + L_g2
print("Collision Op: \n")
print(L)

invML = np.matmul(np.linalg.inv(M) , L)
print("M^-1 L : \n")
print(invML)


if not os.path.exists('plots'):
    print("creating folder `plots`, output will be written into it")
    os.makedirs('plots')


def col_op_rhs(u,t):
    return np.dot(invML,u)

ts = ets.ExplicitODEIntegrator(ets.TSType.RK4)
ts.set_ts_size(params.BEVelocitySpace.VELOCITY_SPACE_DT)
ts.set_rhs_func(col_op_rhs)

X = np.linspace(-3,3,80)
Y = np.linspace(-3,3,80)
T = 5e-9 

h_vec_t=list()
im_count=0
while ts.current_ts()[0] < T:
    ts_info = ts.current_ts()
    if ts_info[1] % params.BEVelocitySpace.IO_STEP_FREQ == 0:
        print("time stepper current time ",ts_info[0])
        print(h_vec)
        fname = params.BEVelocitySpace.IO_FILE_NAME_PREFIX % im_count
        h_vec_t.append(h_vec)
        #visualize_utils.plot_f_z_slice(h_vec,maxwellian,spec,X,Y,fname,0.0,"z=0 , t = %ss"%format(ts_info[0], ".6E"))
        # plt.title("T=%.2f"%ts_info[0])
        # plt.imshow(u[:,:,z_slice_index,0])
        # plt.colorbar()
        # fname = IO_FILE_NAME_PREFIX %ts_info[1]
        # #print(fname)
        # plt.savefig(fname)
        # plt.close()
        im_count+=1
        
    
    #v= np.zeros(u.shape)
    
    h_vec = ts.evolve(h_vec)

h_vec_t = np.transpose(np.array(h_vec_t))
ts      = np.linspace(0,T,h_vec_t.shape[1])
#print(h_vec_t)
#print(ts)
num_pr   = params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER+1
num_sh   = len(params.BEVelocitySpace.SPH_HARM_LM)
sh_modes = params.BEVelocitySpace.SPH_HARM_LM
fig, axs = plt.subplots(num_pr, num_sh)
fig.set_size_inches(4.5*num_sh,4*num_pr,)
for pk in range(params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER+1):
    for lm_i,lm in enumerate(sh_modes):
        axs[pk, lm_i].plot(ts, h_vec_t[pk*num_sh + lm_i],'g-o')
        axs[pk, lm_i].set_title("P_%d Ylm[%d, %d]" %(pk,lm[0],lm[1]))

#plt.show()
fig.savefig("coll_op_g0_g1_g2")
np.savetxt("coll_op_g0_g1_g2.dat",h_vec_t)


# axs[0, 0].plot(x, y)

# axs[0, 1].plot(x, y, 'tab:orange')
# axs[0, 1].set_title('Axis [0, 1]')
# axs[1, 0].plot(x, -y, 'tab:green')
# axs[1, 0].set_title('Axis [1, 0]')
# axs[1, 1].plot(x, -y, 'tab:red')
# axs[1, 1].set_title('Axis [1, 1]')

# for ax in axs.flat:
#     ax.set(xlabel='x-label', ylabel='y-label')

# # Hide x labels and tick labels for top plots and y ticks for right plots.
# for ax in axs.flat:
#     ax.label_outer()











