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

# get the Boltzmann parameters. 
PARS      = boltzmann_parameters.BoltzmannEquationParameters()

# create a basis for the computations. 
hermite_e = basis.HermiteE()

# instance of the collision operator
cf    = collision_operator.CollisionOpElectronNeutral3D(2,hermite_e)
spec  = cf.get_spectral_structure()

# orthogonality of the hermite ensures the mass matrix is diagonal
mm_diag = spec.compute_mass_matrix(is_diagonal=True)

Vi  = spec.create_vec()
Lij = spec.create_mat()
M   = spec.create_mat()

# solution coefficients
Hi = spec.create_vec()

np.fill_diagonal(M,mm_diag)
dt =0.01

# compute the spectral PG approximation of the collision integral
[Vi,Lij] = cf.assemble_collision_mat(binary_collisions.ElectronNeutralCollisionElastic_X0D_V3D())

G = M + dt * Lij
G_inv = np.linalg.inv(G)

for t in np.arange(0,0.06,dt):
    rhs = np.dot(np.transpose(mm_diag),Hi) + dt *Vi
    print(rhs)
    Hi  = np.matmul(G_inv,rhs)
    print(Hi)






