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

# instance of the collision operator
cf    = colOpSp.CollisionOpSP(params.BEVelocitySpace.VELOCITY_SPACE_DIM,params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER)
spec  = colOpSp.SPEC_SPHERICAL

# # solution coefficients
h_vec = spec.create_vec()
#h_vec = np.random.rand(len(h_vec))
h_vec  = np.ones_like(h_vec)
plot_domain = np.array([[-5,5],[-5,5]])

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

M  = spec.compute_maxwellian_mm(maxwellian)

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

ts = ets.ExplicitODEIntegrator(ets.TSType.FORWARD_EULER)
ts.set_ts_size(params.BEVelocitySpace.VELOCITY_SPACE_DT)
ts.set_rhs_func(col_op_rhs)

while ts.current_ts()[0] < 1000:
    ts_info = ts.current_ts()
    if ts_info[1] % params.BEVelocitySpace.IO_STEP_FREQ == 0:
        print("time stepper current time ",ts_info[0])
        print(h_vec)
        fname = params.BEVelocitySpace.IO_FILE_NAME_PREFIX %ts_info[1]
        visualize_utils.plot_f_zslice(h_vec,maxwellian,spec,5,1e-1,1e-1,fname)
        # plt.title("T=%.2f"%ts_info[0])
        # plt.imshow(u[:,:,z_slice_index,0])
        # plt.colorbar()
        # fname = IO_FILE_NAME_PREFIX %ts_info[1]
        # #print(fname)
        # plt.savefig(fname)
        # plt.close()
        
    
    #v= np.zeros(u.shape)
    
    h_vec = ts.evolve(h_vec)





#print(mm)

# orthogonality of the hermite ensures the mass matrix is diagonal
# mm_diag = spec.compute_mass_matrix(is_diagonal=True)
# # Collision operator discretization
# Lij = spec.create_mat()
# # compute the spectral PG approximation of the collision integral
# Lij = cf.assemble_collision_mat(binary_collisions.ElectronNeutralCollisionElastic_X0D_V3D(),maxwellian=maxwellian)
# dt  = PARS.TIME_STEP_SIZE
# [rows, cols] = Lij.shape
# Lij = dt * Lij
# for r in range(rows):
#     Lij[r,r] = Lij[r,r]/mm_diag[r]

# Lij = np.eye(rows) - Lij
# Lij_inv = np.linalg.inv(Lij)

# import os
# if not os.path.exists('plots'):
#     print("creating folder `plots`, output will be written into it")
#     os.makedirs('plots')

# for t_step in range(100):
#     t = t_step*dt
#     print("time %f" %(t))
#     # output_fname = f"plots/fv_step_%04d"%t_step
#     # cf_name      = f"plots/fv_c_step_%04d"%t_step
#     # visualize_utils.plot_spec_coefficients(h_vec,file_name=cf_name)
#     #print(output_fname)
#     print(h_vec)
#     # visualize_utils.plot_density_distribution_z_slice(spec,h_vec,plot_domain,50,z_val=0.1,weight_func=boltzmann_parameters.maxwellian_normalized,file_name=output_fname)
#     #h_vec = np.matmul(Lij_inv,np.multiply(mm_diag,h_vec))
#     h_vec = np.matmul(Lij_inv,h_vec)






