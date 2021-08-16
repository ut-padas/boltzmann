"""
simple cases to test the collision operator, 
compute the collision operator, for known expression 
for cross-section and numerically validate the result
"""
import basis
import numpy as np
import math
import collision_operator_spherical as colOpSp
import collision_operator as colOp
import collisions 
import parameters as params
import visualize_utils
import ets 
import os

cf_sp    = colOpSp.CollisionOpSP(params.BEVelocitySpace.VELOCITY_SPACE_DIM,params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER)
spec_sp  = colOpSp.SPEC_SPHERICAL

cf_ct    = colOp.CollisionOp3D(params.BEVelocitySpace.VELOCITY_SPACE_DIM,params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER)
spec_ct  = colOp.SPEC_HERMITE_E

maxwellian = lambda x: (collisions.MAXWELLIAN_N / ((collisions.ELECTRON_THEMAL_VEL * np.sqrt(np.pi))**3) ) * np.exp(-x**2)
g=collisions.eAr_TestCollision()

L_sp   = cf_sp.assemble_mat(g,maxwellian)
L_cart = cf_ct.assemble_mat(g,maxwellian)

print(L_sp)