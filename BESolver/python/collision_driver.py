"""
@package Boltzmann collision operator solver. 
"""

import basis
import spec as sp
import numpy as np
import math
import collision_operator
import binary_collisions 
import boltzmann_parameters
import visualize_utils

# get the Boltzmann parameters. 
PARS      = boltzmann_parameters.BoltzmannEquationParameters()

# create a basis for the computations. 
hermite_e = basis.HermiteE()

# instance of the collision operator
cf    = collision_operator.CollisionOpElectronNeutral3D(3,hermite_e)
spec  = cf.get_spectral_structure()

# # solution coefficients
h_vec = spec.create_vec()
h_vec = np.ones((h_vec.shape))
plot_domain = np.array([[-5,5],[-5,5]])


# orthogonality of the hermite ensures the mass matrix is diagonal
mm_diag = spec.compute_mass_matrix(is_diagonal=True)
# Collision operator discretization
Lij = spec.create_mat()
# compute the spectral PG approximation of the collision integral
Lij = cf.assemble_collision_mat(binary_collisions.ElectronNeutralCollisionElastic_X0D_V3D())
dt  = PARS.TIME_STEP_SIZE

Lij = -dt * Lij
np.fill_diagonal(Lij, Lij.diagonal() + mm_diag)
Lij_inv = np.linalg.inv(Lij)

import os
if not os.path.exists('plots'):
    print("creating folder `plots`, output will be written into it")
    os.makedirs('plots')

for t_step in range(10):
    t = t_step*dt
    print("time %f" %(t))
    output_fname = f"plots/fv_step_%04d"%t_step
    #print(output_fname)
    visualize_utils.plot_density_distribution_z_slice(spec,h_vec,plot_domain,80,z_val=0.1,weight_func=boltzmann_parameters.maxwellian_normalized,file_name=output_fname)
    h_vec = np.matmul(Lij_inv,h_vec)







