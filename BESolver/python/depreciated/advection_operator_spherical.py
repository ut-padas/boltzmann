import basis as bs
import spec_spherical as sp
import scipy.integrate
import numpy as np
from maxpoly import *
import matplotlib.pyplot as plt
#import profiler
from utils import *
from mpl_toolkits import mplot3d

import math

#t_adv_mat = profiler.profile_t("v_adv")
phimat = np.genfromtxt('sph_harm_del/phimat16.dat',delimiter=',')
psimat = np.genfromtxt('sph_harm_del/psimat16.dat',delimiter=',')

def sph_harm_norm(l,m):
    if m == 0:
        return np.sqrt((2.*l+1)/(4.*np.pi))
    else:
        return (-1)**m * np.sqrt((2.*l+1)/(2.*np.pi)*math.factorial(l-abs(m))/math.factorial(l+abs(m)))

AM = lambda l,m: (l+abs(m))/(2.*l+1.)*sph_harm_norm(l,m)/sph_harm_norm(l-1,m)
BM = lambda l,m: (l-abs(m)+1.)/(2.*l+1.)*sph_harm_norm(l,m)/sph_harm_norm(l+1,m)

AD = lambda l,m: (l+abs(m))*(l+1.)/(2.*l+1.)*sph_harm_norm(l,m)/sph_harm_norm(l-1,m)
BD = lambda l,m: -l*(l-abs(m)+1.)/(2.*l+1.)*sph_harm_norm(l,m)/sph_harm_norm(l+1,m)

CpsiDphiP = np.load('polynomials/maxpoly_CpsiDphiP.npy')
CpsiDphiM = np.load('polynomials/maxpoly_CpsiDphiM.npy')
CphiDpsiP = np.load('polynomials/maxpoly_CphiDpsiP.npy')
CphiDpsiM = np.load('polynomials/maxpoly_CphiDpsiM.npy')

def assemble_advection_matix_lp(Nr, sph_harm_lm):

    num_sh = len(sph_harm_lm)
    num_total = (Nr+1)*num_sh
    adv_mat = np.zeros([num_total, num_total])

    for qs_idx,qs in enumerate(sph_harm_lm):

        lm = [qs[0]+1,qs[1]]
        if lm in sph_harm_lm:
            lm_idx = sph_harm_lm.index(lm)

            for p in range(Nr+1):
                for k in range(Nr+1):
                    klm = k*num_sh + lm_idx
                    pqs = p*num_sh + qs_idx

                    adv_mat[pqs,klm] += (AM(lm[0],lm[1]) - .5*AD(lm[0],lm[1]))*CpsiDphiP[qs[0],p,k] - .5*AD(lm[0],lm[1])*CphiDpsiP[qs[0],p,k]
        
        lm = [qs[0]-1,qs[1]]
        if lm in sph_harm_lm:
            lm_idx = sph_harm_lm.index(lm)

            for p in range(Nr+1):
                for k in range(Nr+1):
                    klm = k*num_sh + lm_idx
                    pqs = p*num_sh + qs_idx

                    adv_mat[pqs,klm] += (BM(lm[0],lm[1]) - .5*BD(lm[0],lm[1]))*CpsiDphiM[qs[0],p,k] - .5*BD(lm[0],lm[1])*CphiDpsiM[qs[0],p,k]

    return adv_mat

# def assemble_advection_matrix(Nr,sph_harm_lm):
#     num_sh = len(sph_harm_lm)
#     num_total = (Nr+1)*num_sh

#     l_max = sph_harm_lm[-1][0]

#     diff_mat = maxpoly.diff_matrix(Nr)
#     lift_mat = maxpoly.lift_matrix(Nr)

#     g_mat = diff_mat - 2*lift_mat
    
#     gg_pk  = np.matmul(g_mat, g_mat)
#     dg_pk  = np.matmul(np.transpose(diff_mat),g_mat)

#     psimat_local= np.transpose(psimat[0:(l_max+1)**2, 0:(l_max+1)**2])
#     phimat_local= np.transpose(phimat[0:(l_max+1)**2, 0:(l_max+1)**2])

#     adv_mat= np.kron(g_mat,psimat_local).reshape(num_total,num_total) + 0.5*(np.kron((gg_pk+dg_pk),phimat_local).reshape(num_total,num_total)) 
#     return adv_mat

# num_dofs_all = [2, 4, 8, 16]
# num_dofs_all = [8, 16, 32, 64]
# num_dofs_all = [2, 4,8]
# num_dofs_all = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
# num_dofs_all = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
num_dofs_all = [16]

error_linf = np.zeros(len(num_dofs_all))
error_l2 = np.zeros(len(num_dofs_all))
error_linf_2d = np.zeros(len(num_dofs_all))
error_l2_2d = np.zeros(len(num_dofs_all))

t_end = 0.5
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
f_num = np.zeros([len(num_dofs_all), len(x)])
f_initial = np.zeros([len(num_dofs_all), len(x)])
f_exact = np.zeros([len(num_dofs_all), len(x)])

f_num_2d = np.zeros([len(num_dofs_all), len(x), len(z)])
f_initial_2d = np.zeros([len(num_dofs_all), len(x), len(z)])
f_exact_2d = np.zeros([len(num_dofs_all), len(x), len(z)])

print(np.shape(sph_coord_init))


for num_dofs_idx, num_dofs in enumerate(num_dofs_all):

    # Nr = 16
    # lmax = num_dofs

    # Nr = num_dofs
    # lmax = 16

    Nr = num_dofs
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
    coeffs[1] = 1
    coeffs[2] = 1
    coeffs[3] = 1
    coeffs[4] = 1
    coeffs[5] = 1
    coeffs[6] = 1
    coeffs[7] = 1

    #t_adv_mat.start()
    advmat   = assemble_advection_matix_lp(Nr, lm_all)
    # advmat    = assemble_advection_matrix(Nr,lm_all)
    #t_adv_mat.stop()
    #print("time for advection op assembly: ",t_adv_mat.seconds)
    #print(np.linalg.norm(advmat-advmat1))
    

    func = lambda t,a: -np.matmul(advmat,a)
    sol = scipy.integrate.solve_ivp(func, (0,t_end), coeffs, max_step=dt, method='RK45')
    # sol = scipy.integrate.solve_ivp(func, (0,t_end), coeffs, max_step=dt, method='BDF')
    coeffs_num = sol.y[:,-1]

    # backward Euler
    # solver = np.linalg.inv(np.eye(Ntotal) + dt*advmat)
    # coeffs_num = coeffs

    # for step in range(nsteps):
        # coeffs_num = np.matmul(solver, coeffs_num)

    # coeff_decomp = np.zeros([(lmax+1)**2, Nr+1])

    # for k in range(Nr+1):
    #     for lm_idx in range(num_sph):
    #         klm = k*num_sph + lm_idx
    #         coeff_decomp[lm_idx, k] = coeffs_num[klm]

    # for lm_idx,lm in enumerate(lm_all):
    #     plt.subplot(lmax+1, 2*lmax+1, lm[0]*(2*lmax+1)+lm[1]+1+lm[0]+lmax-lm[0])
    #     plt.semilogy(np.abs(coeff_decomp[lm_idx,:]))
    #     plt.title("(" + str(lm[0]) + ", " + str(lm[1]) + ")")

    # plt.show()
    # sp._sph_harm_real(lm[0], lm[1], theta, 0)

    maxpolybasis = bs.Maxwell()
    sph_basis = sp.SpectralExpansionSpherical(Nr, maxpolybasis, lm_all)

    for k in range(Nr+1):
        for lm_idx, lm in enumerate(lm_all):
            f_num[num_dofs_idx,:]     += coeffs_num[k*num_sph+lm_idx]*sph_basis.basis_eval_spherical(theta, 0, lm[0], lm[1])*maxpolyeval(2*lm[0]+2,np.abs(x),k)*np.exp(-x**2)*(np.abs(x)**lm[0])
            f_initial[num_dofs_idx,:] += coeffs[k*num_sph+lm_idx]*sph_basis.basis_eval_spherical(theta, 0, lm[0], lm[1])*maxpolyeval(2*lm[0]+2,np.abs(x),k)*np.exp(-(x)**2)*(np.abs(x)**lm[0])
            f_exact[num_dofs_idx,:]   += coeffs[k*num_sph+lm_idx]*sph_basis.basis_eval_spherical(theta, 0, lm[0], lm[1])*maxpolyeval(2*lm[0]+2,x-t_end,k)*np.exp(-(x-t_end)**2)*(np.abs(x)**lm[0])
            
            f_num_2d[num_dofs_idx,:]     += coeffs_num[k*num_sph+lm_idx]*sph_basis.basis_eval_spherical(sph_coord_init[1], sph_coord_init[2], lm[0], lm[1])*maxpolyeval(2*lm[0]+2,sph_coord_init[0],k)*np.exp(-sph_coord_init[0]**2)*(sph_coord_init[0]**lm[0])
            f_initial_2d[num_dofs_idx,:] += coeffs[k*num_sph+lm_idx]*sph_basis.basis_eval_spherical(sph_coord_init[1], sph_coord_init[2], lm[0], lm[1])*maxpolyeval(2*lm[0]+2,sph_coord_init[0],k)*np.exp(-sph_coord_init[0]**2)*(sph_coord_init[0]**lm[0])
            f_exact_2d[num_dofs_idx,:]   += coeffs[k*num_sph+lm_idx]*sph_basis.basis_eval_spherical(sph_coord_end[1], sph_coord_end[2], lm[0], lm[1])*maxpolyeval(2*lm[0]+2,sph_coord_end[0],k)*np.exp(-sph_coord_end[0]**2)*(sph_coord_end[0]**lm[0])

    error_linf[num_dofs_idx] = np.max(abs(f_num[num_dofs_idx,:]-f_exact[0,:]))
    error_l2[num_dofs_idx] = np.linalg.norm(f_num[num_dofs_idx,:]-f_exact[0,:])

    error_linf_2d[num_dofs_idx] = np.max(abs(f_num_2d[num_dofs_idx,:]-f_exact_2d[0,:]))
    error_l2_2d[num_dofs_idx] = np.linalg.norm(f_num_2d[num_dofs_idx,:]-f_exact_2d[0,:])

    v = np.linspace(0,5,100)
    radial_part = np.zeros([num_sph, len(v)])
    for lm_idx, lm in enumerate(lm_all):
        for k in range(Nr+1):
            radial_part[lm_idx,:] += coeffs_num[k*num_sph+lm_idx]*maxpolyeval(2*lm[0]+2,np.abs(x),k)*np.exp(-v**2)*(np.abs(v)**lm[0])

        plt.subplot(int(np.round(np.sqrt(num_sph))), int(np.ceil(num_sph/int(np.round(np.sqrt(num_sph))))),lm_idx+1)
        plt.plot(v, radial_part[lm_idx,:],'.-')

    plt.show()



plt.semilogy(abs(coeffs_num),'.-')
plt.show()
plt.subplot(2,3,1)
plt.plot(x, f_initial[0,:])
plt.plot(x, f_exact[0,:])

for num_dofs_idx,Nr in enumerate(num_dofs_all):
    plt.plot(x, f_num[num_dofs_idx,:], '--')

plt.grid()
plt.legend(['Initial Conditions', 'Exact', 'Numerical'])
# plt.legend(['Initial Conditions', 'Exact', '$N_r = 32, l_{max} = 2$', '$N_r = 32, l_{max} = 4$', '$N_r = 32, l_{max} = 8$', '$N_r = 32, l_{max} = 16$'])
# plt.legend(['Initial Conditions', 'Exact', '$N_r = 8, l_{max} = 8$', '$N_r = 16, l_{max} = 8$', '$N_r = 32, l_{max} = 8$', '$N_r = 64, l_{max} = 8$'])
plt.ylabel('Distribution function')
plt.xlabel('$v_z$')

plt.subplot(2,3,2)
plt.semilogy(x, f_initial[0,:])
plt.plot(x, f_exact[0,:])

for num_dofs_idx,Nr in enumerate(num_dofs_all):
    plt.plot(x, f_num[num_dofs_idx,:], '--')

plt.grid()
plt.legend(['Initial Conditions', 'Exact', 'Numerical'])
# plt.legend(['Initial Conditions', 'Exact', '$N_r = 32, l_{max} = 2$', '$N_r = 32, l_{max} = 4$', '$N_r = 32, l_{max} = 8$', '$N_r = 32, l_{max} = 16$'])
# plt.legend(['Initial Conditions', 'Exact', '$N_r = 8, l_{max} = 8$', '$N_r = 16, l_{max} = 8$', '$N_r = 32, l_{max} = 8$', '$N_r = 64, l_{max} = 8$'])
plt.ylabel('Distribution function')
plt.xlabel('$v_z$')

plt.subplot(2,3,4)

for num_dofs_idx,Nr in enumerate(num_dofs_all):
    plt.semilogy(x, abs(f_num[num_dofs_idx,:]-f_exact[0,:]), '--')

plt.grid()
# plt.legend(['Initial Conditions', 'Exact', 'Numerical'])
# plt.legend(['$N_r = 32, l_{max} = 2$', '$N_r = 32, l_{max} = 4$', '$N_r = 32, l_{max} = 8$', '$N_r = 32, l_{max} = 16$'])
# plt.legend(['Initial Conditions', 'Exact', '$N_r = 8, l_{max} = 8$', '$N_r = 16, l_{max} = 8$', '$N_r = 32, l_{max} = 8$', '$N_r = 64, l_{max} = 8$'])
plt.ylabel('Error in distribution function')
plt.xlabel('$v_z$')

plt.subplot(2,3,5)
plt.semilogy(num_dofs_all, error_linf_2d, '-o')
plt.semilogy(num_dofs_all, error_l2_2d, '-*')
plt.ylabel('Error')
plt.xlabel('$l_{\max}$')
plt.grid()
plt.legend(['$L_\inf$', '$L_2$'])

plt.subplot(1,3,3)
plt.contour(quad_grid[0], quad_grid[1], f_initial_2d[-1,:,:], linestyles='solid', colors='grey', linewidths=1)
plt.contour(quad_grid[0], quad_grid[1], f_exact_2d[-1,:,:], linestyles='dashed', colors='red', linewidths=2)
ax = plt.contour(quad_grid[0], quad_grid[1], f_num_2d[-1,:,:], linestyles='dotted', colors='blue', linewidths=2)
# ax.plot_surface(quad_grid[0], quad_grid[1], f_initial2[2,:,:], rstride=1, cstride=1,
#                 cmap='viridis', edgecolor='none')
# ax.set_title('surface');
plt.gca().set_aspect('equal')


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