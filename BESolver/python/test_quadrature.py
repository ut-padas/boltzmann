import basis
import spec as sp
import numpy as np
import math
from scipy.special import sph_harm
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


def _sph_harm_real(l, m, theta, phi):
        # in python's sph_harm phi and theta are swapped
        Y = sph_harm(abs(m), l, phi, theta)
        if m < 0:
            Y = np.sqrt(2) * (-1)**m * Y.imag
        elif m > 0:
            Y = np.sqrt(2) * (-1)**m * Y.real
        else:
            Y=Y.real

        return Y 

num_q_pts_on_sphere=100
legendre = basis.Legendre()
[glx,glw] = legendre.Gauss_Pn(num_q_pts_on_sphere)
theta_q = np.arccos(glx)
phi_q = np.linspace(0,2*np.pi,2*(num_q_pts_on_sphere))
spherical_quadrature_fac = (np.pi/num_q_pts_on_sphere)

lm_modes = [[0,0],[1,0],[1,-1],[1,1]]
num_lm_modes = len(lm_modes)
q_mat  = np.zeros((num_lm_modes,num_lm_modes))

for v_theta_i, v_theta in enumerate(theta_q):
    for v_phi in phi_q:
        for lm1_i, lm1 in enumerate(lm_modes):
            for lm2_j, lm2 in enumerate(lm_modes):
                q_mat[lm1_i , lm2_j] += glw[v_theta_i]*spherical_quadrature_fac* _sph_harm_real(lm1[0],lm1[1],v_theta,v_phi) * _sph_harm_real(lm2[0],lm2[1],v_theta,v_phi)

print(q_mat)



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


# import collision_operator
# import binary_collisions 
# import boltzmann_parameters
# import visualize_utils

# electron_ar_elastic = binary_collisions.ElectronNeutralCollisionElastic_X0D_V3D()
# #electron_ar_inelastic = binary_collisions.ElectronNeutralCollisionKind1_X0D_V3D()

# cOp = collision_operator.CollisionOpElectronNeutral3D(3,hermiteE)

# #cOp.assemble_collision_mat(electron_ar_elastic)
# #cOp.assemble_collision_mat(electron_ar_inelastic)

# spec = cOp.get_spectral_structure()
# maxwelian1 = lambda x : boltzmann_parameters.gaussian(x,mu=None,sigma=1)
# c_vec = spec.compute_coefficients(maxwelian1, mm_diag=None)
# print(c_vec)
# plot_domain = np.array([[-5,5],[-5,5]])
# visualize_utils.plot_density_distribution_z_slice(spec,c_vec,plot_domain,50,z_val=0.0,weight_func=hermiteE.Wx(),file_name=None)