import basis
import spec as sp
import numpy as np
import math

x = np.linspace(-1,1,10)
y = np.linspace(-1,1,10)
z = np.linspace(-1,1,10)


#S=sp.SpectralExpansion(3,3,basis.hermite_e)
#print(S.basis_eval3d(x,y,z,(0,0,0)))
#print(S.basis_eval2d(x,y,z,(1,1,2)))

hermiteE=basis.HermiteE()

p_order=3
[gh_x,gh_w] = hermiteE.Gauss_Pn(p_order+1)

## 1d mass matrix
M = np.zeros((p_order+1,p_order+1))
for i in range(p_order+1):
    hi = hermiteE.Pn(i)
    for j in range(p_order+1):
        hj = hermiteE.Pn(j)
        for q,x in enumerate(gh_x):
            M[i,j]+= hi(x) * hj(x) *gh_w[q]
        if ( i==j):
            M[i,j]/= np.sqrt(2*np.pi) * math.factorial(i)

print(M)

## 3D example. 
h_poly=list()
M = np.zeros(((p_order+1)**3,(p_order+1)**3))
for i in range(p_order+1):
    h_poly.append(hermiteE.Pn(i))

for ek in range(p_order+1):
    for ej in range(p_order+1):
        for ei in range(p_order+1):
            row = ek*(p_order+1)*(p_order+1) + ej*(p_order+1) + ei
            for wz,z in enumerate(gh_x):
                for wy,y in enumerate(gh_x):
                    for wx,x in enumerate(gh_x):
                        M[row,row] += gh_w[wx]*gh_w[wy]*gh_w[wz] * h_poly[ei](x) * h_poly[ej](y)  * h_poly[ek](z) * h_poly[ei](x) * h_poly[ej](y)  * h_poly[ek](z)
            M[row,row]/=((np.sqrt(2*np.pi)**3)*math.factorial(ei)*math.factorial(ej)*math.factorial(ek))

print(M)

#1d
# se= sp.SpectralExpansion(1,1,hermiteE)
# M=se.compute_mass_matrix(is_diagonal=True)
# print(M)

# M=se.compute_mass_matrix(is_diagonal=False)
# print(M)

# se= sp.SpectralExpansion(2,1,hermiteE)
# M=se.compute_mass_matrix(is_diagonal=True)
# print(M)

# M=se.compute_mass_matrix(is_diagonal=False)
# print(M)

# se= sp.SpectralExpansion(3,1,hermiteE)
# M=se.compute_mass_matrix(is_diagonal=True)
# print(M)

# M=se.compute_mass_matrix(is_diagonal=False)
# print(M)


import collision_operator
import binary_collisions 

electron_ar_elastic = binary_collisions.ElectronNeutralCollisionElastic_X0D_V3D()
electron_ar_inelastic = binary_collisions.ElectronNeutralCollisionKind1_X0D_V3D()

cOp = collision_operator.CollisionOpElectronNeutral3D(1,hermiteE)

cOp.assemble_collision_mat(electron_ar_elastic)
cOp.assemble_collision_mat(electron_ar_inelastic)
