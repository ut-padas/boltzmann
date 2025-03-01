"""
@package Boltzmann collision operator solver. 
"""

import basis
import spec as sp
import numpy as np
import math
import collision_operator
import collisions
import parameters as params
import visualize_utils
import ets,os

# get the Boltzmann parameters. 
# instance of the collision operator
cf    = collision_operator.CollisionOp3D(params.BEVelocitySpace.VELOCITY_SPACE_DIM, params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER)
spec  = collision_operator.SPEC_HERMITE_E

# # solution coefficients
# h_vec = np.ones(spec.get_num_coefficients())
# h_vec  = h_vec/1e2 
h_vec = np.zeros(spec.get_num_coefficients())
# intially set to Maxwellian
h_vec[0]=1



#maxwellian = basis.Maxwell().Wx()
print("""===========================Parameters ======================""")
print("\tMAXWELLIAN_N : ", collisions.MAXWELLIAN_N)
print("\tELECTRON_THEMAL_VEL : ", collisions.ELECTRON_THEMAL_VEL," ms^-1")
print("""============================================================""")

maxwellian = lambda x: (collisions.MAXWELLIAN_N / ((collisions.ELECTRON_THEMAL_VEL * np.sqrt(np.pi))**3) ) * np.exp(-x**2)
#maxwellian = lambda x: (1/ ((np.sqrt(np.pi))**3) ) * np.exp(-x**2)
col_g0 = collisions.eAr_G0()
col_g1 = collisions.eAr_G1()
col_g2 = collisions.eAr_G2()

M  = spec.compute_maxwellian_mm(maxwellian, collisions.ELECTRON_THEMAL_VEL)

print("Mass matrix w.r.t. Maxwellian\n")
print(M)

L_g0  = cf.assemble_mat(col_g0,maxwellian)
L_g1  = cf.assemble_mat(col_g1,maxwellian)
L_g2  = cf.assemble_mat(col_g2,maxwellian)

L     = L_g0 + L_g1 + L_g2
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

#X= np.linspace(-4,4,80)
#Y= np.linspace(-4,4,80)
plot_domain = np.array([[-3,3],[-3,3]])
im_count=0
while ts.current_ts()[0] < 1000:
    ts_info = ts.current_ts()
    if ts_info[1] % params.BEVelocitySpace.IO_STEP_FREQ == 0:
        print("time stepper current time ",ts_info[0])
        print(h_vec)
        fname = params.BEVelocitySpace.IO_FILE_NAME_PREFIX %im_count
        visualize_utils.plot_density_distribution_z_slice(spec,h_vec,plot_domain,80,0.0,maxwellian,fname)
        #visualize_utils.plot_f_z_slice(h_vec,maxwellian,spec,X,Y,fname,0.0)
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








