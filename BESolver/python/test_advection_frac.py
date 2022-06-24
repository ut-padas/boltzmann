import basis as bs
import spec_spherical as sp
import scipy.integrate
import numpy as np
from maxpoly_frac import *
import matplotlib.pyplot as plt
# import profiler
from utils import *
from mpl_toolkits import mplot3d
from scipy.special import eval_genlaguerre 
from advection_operator_spherical_polys import *

from time import perf_counter as time

l_all = [3,4,5,6]
Nr_all = [4,8,16,30]

error_linf_2d_max_l = np.zeros(len(l_all))

error_linf_2d_max_Nr = np.zeros(len(Nr_all))

t_end = 0.1
# nsteps = 400000
nsteps = 1000
dt = t_end/nsteps

x = np.linspace(-2,2,100)
z = np.linspace(-2,2,100)
quad_grid = np.meshgrid(x,z,indexing='ij')

y = np.zeros_like(quad_grid[0])

sph_coord_init = cartesian_to_spherical(quad_grid[0],y,quad_grid[1])
sph_coord_end = cartesian_to_spherical(quad_grid[0],y,quad_grid[1]-t_end)

theta = 0.5*np.pi - np.sign(x)*0.5*np.pi

f_initial = np.zeros(len(x))
f_exact = np.zeros(len(x))

f_num_max_l = np.zeros([len(l_all), len(x)])
f_num_max_Nr = np.zeros([len(Nr_all), len(x)])

f_initial_2d = np.zeros([len(x), len(z)])
f_exact_2d = np.zeros([len(x), len(z)])

f_num_2d_max_l = np.zeros([len(l_all), len(x), len(z)])
f_num_2d_max_Nr = np.zeros([len(l_all), len(x), len(z)])

# sample initial conditions and exact solution
Nr = Nr_all[0]
lmax = l_all[0]

lm_all = []

for l in range(lmax+1):
    # for m in range(-l,l+1):
    for m in range(1):
        lm_idx = l**2+1+l+m
        lm_all.append([l,m])

num_sph = len(lm_all)
Ntotal = (Nr+1)*num_sph

coeffs = np.zeros(Ntotal)
coeffs[0] = 1

maxpolybasis = bs.MaxwellEnergy()
sph_basis = sp.SpectralExpansionSpherical(Nr, maxpolybasis, lm_all)

for k in range(Nr+1):
    for lm_idx, lm in enumerate(lm_all):           
        f_initial += coeffs[k*num_sph+lm_idx]*sph_basis.basis_eval_full(np.abs(x), 0.5*np.pi - np.sign(x)*0.5*np.pi, 0, k, lm[0], lm[1])*np.exp(-(x)**4)
        f_exact   += coeffs[k*num_sph+lm_idx]*sph_basis.basis_eval_full((x-t_end), 0.5*np.pi - np.sign(x-t_end)*0.5*np.pi, 0, k, lm[0], lm[1])*np.exp(-(x-t_end)**4)
        
        f_initial_2d += coeffs[k*num_sph+lm_idx]*sph_basis.basis_eval_full(sph_coord_end[0], sph_coord_init[1], sph_coord_init[2], k, lm[0], lm[1])*np.exp(-sph_coord_init[0]**4)
        f_exact_2d   += coeffs[k*num_sph+lm_idx]*sph_basis.basis_eval_full(sph_coord_end[0], sph_coord_end[1], sph_coord_end[2], k, lm[0], lm[1])*np.exp(-sph_coord_end[0]**4)

for num_dofs_idx, num_dofs in enumerate(l_all):

    Nr = Nr_all[-1]
    lmax = num_dofs

    lm_all = []

    for l in range(lmax+1):
        # for m in range(-l,l+1):
        for m in range(1):
            lm_idx = l**2+1+l+m
            lm_all.append([l,m])

    num_sph = len(lm_all)
    Ntotal = (Nr+1)*num_sph

    coeffs = np.zeros(Ntotal)
    coeffs[0] = 1

    t1=time()
    advmat = assemble_advection_matix_lp_max_energy(Nr, lm_all)
    t2=time()
    print("Advection Operator assembly time (s): ",(t2-t1))
    
    func = lambda t,a: -np.matmul(advmat,a)
    sol = scipy.integrate.solve_ivp(func, (0,t_end), coeffs, max_step=dt, method='RK45')
    coeffs_num_max = sol.y[:,-1]

    maxpolybasis = bs.MaxwellEnergy()
    sph_basis = sp.SpectralExpansionSpherical(Nr, maxpolybasis, lm_all)

    for k in range(Nr+1):
        for lm_idx, lm in enumerate(lm_all):           
            f_num_max_l[num_dofs_idx,:] += coeffs_num_max[k*num_sph+lm_idx]*sph_basis.basis_eval_full(np.abs(x), 0.5*np.pi - np.sign(x)*0.5*np.pi, 0, k, lm[0], lm[1])*np.exp(-x**4)
 
            f_num_2d_max_l[num_dofs_idx,:] += coeffs_num_max[k*num_sph+lm_idx]*sph_basis.basis_eval_full(sph_coord_init[0], sph_coord_init[1], sph_coord_init[2], k, lm[0], lm[1])*np.exp(-sph_coord_init[0]**4)


    error_linf_2d_max_l[num_dofs_idx] = np.max(abs(f_num_2d_max_l[num_dofs_idx,:]-f_exact_2d))


for num_dofs_idx, num_dofs in enumerate(Nr_all):

    Nr = num_dofs
    lmax = l_all[-1]

    lm_all = []

    for l in range(lmax+1):
        # for m in range(-l,l+1):
        for m in range(1):
            lm_idx = l**2+1+l+m
            lm_all.append([l,m])

    num_sph = len(lm_all)
    Ntotal = (Nr+1)*num_sph

    coeffs = np.zeros(Ntotal)
    coeffs[0] = 1

    t1=time()
    advmat = assemble_advection_matix_lp_max_energy(Nr, lm_all)
    t2=time()
    print("Advection Operator assembly time (s): ",(t2-t1))
    
    func = lambda t,a: -np.matmul(advmat,a)
    sol = scipy.integrate.solve_ivp(func, (0,t_end), coeffs, max_step=dt, method='RK45')
    coeffs_num_max = sol.y[:,-1]

    maxpolybasis = bs.MaxwellEnergy()
    sph_basis = sp.SpectralExpansionSpherical(Nr, maxpolybasis, lm_all)

    for k in range(Nr+1):
        for lm_idx, lm in enumerate(lm_all):           
            f_num_max_Nr[num_dofs_idx,:] += coeffs_num_max[k*num_sph+lm_idx]*sph_basis.basis_eval_full(np.abs(x), 0.5*np.pi - np.sign(x)*0.5*np.pi, 0, k, lm[0], lm[1])*np.exp(-x**4)
 
            f_num_2d_max_Nr[num_dofs_idx,:] += coeffs_num_max[k*num_sph+lm_idx]*sph_basis.basis_eval_full(sph_coord_init[0], sph_coord_init[1], sph_coord_init[2], k, lm[0], lm[1])*np.exp(-sph_coord_init[0]**4)


    error_linf_2d_max_Nr[num_dofs_idx] = np.max(abs(f_num_2d_max_Nr[num_dofs_idx,:]-f_exact_2d))

# plt.semilogy(abs(coeffs_num_lag),'.-')
# plt.show()
# plt.subplot(1,5,1)
# plt.plot(x, f_initial)
# plt.plot(x, f_exact)

# # for num_dofs_idx,Nr in enumerate(num_dofs_all):
# #     plt.plot(x, f_num_lag[num_dofs_idx,:], '--')
# #     plt.plot(x, f_num_max[num_dofs_idx,:], ':')
# plt.plot(x, f_num_lag_l[-1,:], '--')
# plt.plot(x, f_num_max_l[-1,:], ':')

# plt.grid()
# plt.legend(['Initial', 'Exact', 'Laguerre', 'Maxwell'])
# # plt.legend(['Initial Conditions', 'Exact', '$N_r = 32, l_{max} = 2$', '$N_r = 32, l_{max} = 4$', '$N_r = 32, l_{max} = 8$', '$N_r = 32, l_{max} = 16$'])
# # plt.legend(['Initial Conditions', 'Exact', '$N_r = 8, l_{max} = 8$', '$N_r = 16, l_{max} = 8$', '$N_r = 32, l_{max} = 8$', '$N_r = 64, l_{max} = 8$'])
# plt.ylabel('Distribution function')
# plt.xlabel('$v_z$')

plt.subplot(1,5,2)
plt.semilogy(x, f_initial)
plt.plot(x, f_exact)

for num_dofs_idx,Nr in enumerate(Nr_all):
    # plt.plot(x, f_num_lag_l[num_dofs_idx,:], '--')
    plt.plot(x, f_num_max_Nr[num_dofs_idx,:], ':')
# plt.plot(x, f_num_max_l[-1,:], ':')

plt.grid()
plt.legend(['Initial', 'Exact', 'Maxwell'])
# plt.legend(['Initial Conditions', 'Exact', '$N_r = 32, l_{max} = 2$', '$N_r = 32, l_{max} = 4$', '$N_r = 32, l_{max} = 8$', '$N_r = 32, l_{max} = 16$'])
# plt.legend(['Initial Conditions', 'Exact', '$N_r = 8, l_{max} = 8$', '$N_r = 16, l_{max} = 8$', '$N_r = 32, l_{max} = 8$', '$N_r = 64, l_{max} = 8$'])
plt.ylabel('Distribution function')
plt.xlabel('$v_z$')

plt.subplot(1,5,3)
plt.contour(quad_grid[0], quad_grid[1], f_initial_2d, linestyles='solid', colors='grey', linewidths=1)
plt.contour(quad_grid[0], quad_grid[1], f_exact_2d, linestyles='dashed', colors='red', linewidths=2)
ax = plt.contour(quad_grid[0], quad_grid[1], f_num_2d_max_l[-1,:,:], linestyles='dotted', colors='blue', linewidths=2)
# ax.plot_surface(quad_grid[0], quad_grid[1], f_initial2[2,:,:], rstride=1, cstride=1,
#                 cmap='viridis', edgecolor='none')
# ax.set_title('surface');
plt.gca().set_aspect('equal')
plt.legend(['Initial', 'Exact', 'Laguerre', 'Maxwell'])

plt.subplot(1,5,4)
plt.semilogy(Nr_all, error_linf_2d_max_Nr, '--o')
plt.ylabel('Error')
plt.xlabel('$N_r$')
plt.grid()
plt.legend(['Laguerre', 'Maxwell'])

plt.subplot(1,5,5)
plt.semilogy(l_all, error_linf_2d_max_l, '--o')
plt.ylabel('Error')
plt.xlabel('$l_{\max}$')
plt.grid()
plt.legend(['Laguerre', 'Maxwell'])


# plt.subplot(1,5,6)

# for num_dofs_idx,Nr in enumerate(l_all):
#     plt.semilogy(x, abs(f_num_lag_l[num_dofs_idx,:]-f_exact), '--')
#     plt.semilogy(x, abs(f_num_max_l[num_dofs_idx,:]-f_exact), ':')
    

# plt.grid()
# # plt.legend(['Initial Conditions', 'Exact', 'Numerical'])
# # plt.legend(['$N_r = 32, l_{max} = 2$', '$N_r = 32, l_{max} = 4$', '$N_r = 32, l_{max} = 8$', '$N_r = 32, l_{max} = 16$'])
# # plt.legend(['Initial Conditions', 'Exact', '$N_r = 8, l_{max} = 8$', '$N_r = 16, l_{max} = 8$', '$N_r = 32, l_{max} = 8$', '$N_r = 64, l_{max} = 8$'])
# plt.ylabel('Error in distribution function')
# plt.xlabel('$v_z$')


plt.show()

# plt.semilogy(x,np.abs(f_in-f))
# # plt.semilogy(x,np.abs(f_ex-f))
# plt.grid()
# plt.legend(['Initial Conditions', 'Exact', 'Nr=5, l_max=10'])
# plt.show()

# vx = np.linspace(-5,5,100)
# vy = np.linspace(-5,5,100)

# f = np.zeros([100, 100])
# f_sl = np.zeros([100, 100])
# f_ex = np.zeros([100, 100])

# for k in range(Nr+1):
#     for lm_idx, lm in enumerate(lm_all):