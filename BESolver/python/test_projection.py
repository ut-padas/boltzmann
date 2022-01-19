import numpy as np
import matplotlib.pyplot as plt
from numpy.ma.core import concatenate
from maxpoly import *

# max dofs used for represenation
max_dofs = 50

# change in thermal velocity
# eps_all = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
# eps_all = np.linspace(-0.45, 0.55, 21)
eps_all = [.25]
eps_all = np.linspace(0.05, 0.55, 11)
# eps_all = np.arange(-0.15,0.201,0.01)
# eps_all = np.arange(-0.15,0.001,0.01)
# eps_all = np.arange(-0.15,0.2,0.05)
eps_all = [-0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15, 0.20]
# eps_all = [-0.015, -0.010, -0.005, 0, 0.005, 0.010, 0.015, 0.020]
# eps_all = [0.0015]

# use Gaussian quadratures or trapezoidal integration for projection
use_gauss_for_proj = False

# order of Gaussian quadrature used for projection
gauss_q_order_proj = 50

# number of trapezoidal points used for projection
trapz_num_q_points_proj = 100000

# tolerance for determining unstable growth
tol = 1.e-13

# get gaussian quadratures
[xg, wg] = maxpolygauss(gauss_q_order_proj)

N_crit = np.zeros(np.shape(eps_all))
maxeig_fwd = np.zeros([len(eps_all), max_dofs])
maxeig_bwd = np.zeros([len(eps_all), max_dofs])
maxeig_cyc = np.zeros([len(eps_all), max_dofs])

a_exp = np.ones(max_dofs)*np.exp(-np.linspace(0, 100, max_dofs))*0
a_noise = np.random.rand(max_dofs)*1.e-14
a_exp[0] = 1

# for i in range(max_dofs):
#     if abs(a[i]) < 1.e-15:
#         a[i] = 0

proj_mat_fwd = np.zeros([max_dofs, max_dofs])
proj_mat_fwd_pen = np.zeros([max_dofs, max_dofs])
proj_mat_fwd_over = np.zeros([max_dofs, max_dofs])
proj_mat_fwd_back = np.zeros([max_dofs, max_dofs])
proj_mat_bwd = np.zeros([max_dofs, max_dofs])
proj_mat_bwd2 = np.zeros([max_dofs, max_dofs])

eigs_fwd = np.zeros([len(eps_all), max_dofs])
eigs_fwd2 = np.zeros([len(eps_all), max_dofs])
eigs_bwd = np.zeros([len(eps_all), max_dofs])
eigs_bwd2 = np.zeros([len(eps_all), max_dofs])

b_exp = np.zeros([len(eps_all), max_dofs])
b_noise = np.zeros([len(eps_all), max_dofs])

c_exp_noise = np.zeros([len(eps_all), max_dofs])

f_exp = np.zeros([len(eps_all), max_dofs])
f_noise = np.zeros([len(eps_all), max_dofs])

overstep = 5
pen = 1.e-14

for eps_idx, eps in enumerate(eps_all):
    eps_fwd = eps
    eps_bwd = -eps/(1.+eps)

    for i in range(max_dofs):
        norm = np.sum(maxpolyeval(2, xg, i)**2*wg)
        for j in range(max_dofs):
            proj_mat_fwd[i, j] = np.sum(maxpolyeval(2, xg, j)*maxpolyeval(2, xg/(1.+eps_fwd), i)*wg)/norm
            proj_mat_fwd_over[i, j] = np.sum(maxpolyeval(2, xg, j)*maxpolyeval(2, xg/(1.+overstep*eps_fwd), i)*wg)/norm
            proj_mat_fwd_back[i, j] = np.sum(maxpolyeval(2, xg, j)*maxpolyeval(2, xg/(1.+eps_fwd)*(1.+overstep*eps_fwd), i)*wg)/norm
            proj_mat_bwd[i, j] = np.sum(maxpolyeval(2, xg, j)*maxpolyeval(2, xg/(1.+eps_bwd), i)*wg)/norm
            proj_mat_bwd2[i, j] = np.sum(maxpolyeval(2, xg, j)*maxpolyeval(2, xg/(1.+2.*eps_bwd), i)*wg)/norm

    proj_mat_fwd_pen = np.matmul(np.transpose(proj_mat_bwd),proj_mat_bwd) + pen*np.eye(max_dofs,max_dofs)
    proj_mat_fwd_pen = np.matmul(np.linalg.inv(proj_mat_fwd_pen),np.transpose(proj_mat_bwd))
    proj_mat_fwd = np.linalg.inv(proj_mat_bwd)
    proj_mat_fwd_over = np.matmul(proj_mat_fwd_back, proj_mat_fwd_over)
    proj_mat_bwd2 = np.matmul(proj_mat_fwd, proj_mat_bwd2)
    eigs_fwd[eps_idx,:] = np.reshape(abs(np.linalg.eigvals(proj_mat_fwd)), [1,max_dofs])
    eigs_fwd2[eps_idx,:] = np.reshape(abs(np.linalg.eigvals(proj_mat_fwd_over)), [1,max_dofs])
    eigs_bwd[eps_idx,:] = np.reshape(abs(np.linalg.eigvals(proj_mat_bwd)), [1,max_dofs])
    eigs_bwd2[eps_idx,:] = np.reshape(abs(np.linalg.eigvals(proj_mat_bwd2)), [1,max_dofs])

    # w, v = np.linalg.eig(proj_mat_bwd)
    # c_noise = np.linalg.solve(v, a_exp)
    # c_noise2 = np.linalg.solve(v, a_noise+a_exp)
    # plt.semilogy(abs(c_noise-c_noise2)/abs(c_noise),'-o')
    # plt.semilogy(abs(np.real(c_noise)),'-o')
    # plt.semilogy(abs(np.imag(c_noise)),'-o')
    # plt.semilogy(abs(np.real(c_noise2)),'-*')
    # plt.semilogy(abs(np.imag(c_noise2)),'-*')
    # plt.show()
    # if eps > 0:
    #     c_noise[0:5] = 0
    # else:
    #     c_noise[-5:] = 0
    # c_noise = np.matmul(v, c_noise)

    if abs(eps) < 1.e-15:
        b_exp[eps_idx,:] = a_exp
        b_noise[eps_idx,:] = a_noise
        # b_noise[eps_idx,:] = c_noise
        eps_all[eps_idx] = 0
    else:
        # b_exp[eps_idx,:] = np.matmul(proj_mat_fwd, a_exp)
        # b_noise[eps_idx,:] = np.matmul(proj_mat_fwd, a_noise)
        # b_exp[eps_idx,:] = np.matmul(proj_mat_fwd, a_exp)
        # b_noise[eps_idx,:] = np.matmul(proj_mat_fwd, a_noise)
        b_exp[eps_idx,:] = np.matmul(proj_mat_fwd_pen, a_exp)
        b_noise[eps_idx,:] = np.matmul(proj_mat_fwd_pen, a_noise)
        # b_noise[eps_idx,:] += a_noise
        # b_exp[eps_idx,:] = np.matmul(proj_mat_bwd, b_exp[eps_idx,:])
        # b_noise[eps_idx,:] = np.matmul(proj_mat_bwd, b_noise[eps_idx,:])
        # b_noise[eps_idx,:] = np.matmul(proj_mat_fwd, c_noise)
        # b_exp[eps_idx,:] = a_exp
        # b_noise[eps_idx,:] = np.real(c_noise)

# plt.semilogy(np.transpose(eigs_fwd), ':')
# plt.semilogy(np.transpose(eigs_fwd2))
# plt.ylabel('Magnitude')
# plt.xlabel('Eigenvalue no.')
# plt.grid()
# plt.legend(['$\epsilon=$'+str(eps) for eps in eps_all])
# plt.show()

eps0 = 0.001
overstep = 10

# for eps_idx, eps in enumerate(eps_all):

#     c_exp_noise[eps_idx,:] = a_exp+a_noise

#     if eps > 0:
#         n = int(np.ceil(np.log(1.+eps)/np.log(1.+eps0)))
#     else:
#         n = int(np.ceil(np.log(1.+eps)/np.log(1.-eps0)))

#     if n > 0:
#         eps_fwd = (1.+eps)**(1/n) - 1.

#         for i in range(max_dofs):
#             norm = np.sum(maxpolyeval(2, xg, i)**2*wg)
#             for j in range(max_dofs):
#                 proj_mat_fwd[i, j] = np.sum(maxpolyeval(2, xg, j)*maxpolyeval(2, xg/(1.+overstep*eps_fwd), i)*wg)/norm
#                 proj_mat_fwd_back[i, j] = np.sum(maxpolyeval(2, xg, j)*maxpolyeval(2, xg*(1.+overstep*eps_fwd), i)*wg)/norm

#         for i in range(n):
#             # for j in range(max_dofs):
#             #     if abs(c_exp_noise[eps_idx,j]) < 1.e-14:
#             #         c_exp_noise[eps_idx,j] = 0
#             c_exp_noise[eps_idx,:] = np.matmul(proj_mat_fwd, c_exp_noise[eps_idx,:])
#             c_exp_noise[eps_idx,:] = np.matmul(proj_mat_fwd_back, c_exp_noise[eps_idx,:])

for eps_idx, eps in enumerate(eps_all):

    c_exp_noise[eps_idx,:] = a_exp+a_noise

    if eps > 0:
        n = int(np.ceil(np.log(1.+eps)/np.log(1.+eps0)))
    else:
        n = int(np.ceil(np.log(1.+eps)/np.log(1.-eps0)))

    if n > 0:
        eps_fwd = (1.+eps)**(1/n) - 1.

        for i in range(max_dofs):
            norm = np.sum(maxpolyeval(2, xg, i)**2*wg)
            for j in range(max_dofs):
                proj_mat_fwd[i, j] = np.sum(maxpolyeval(2, xg, j)*maxpolyeval(2, xg/(1.+eps_fwd), i)*wg)/norm

        for i in range(n):
            for j in range(max_dofs):
                if abs(c_exp_noise[eps_idx,j]) < 1.e-14:
                    c_exp_noise[eps_idx,j] = 0
            c_exp_noise[eps_idx,:] = np.matmul(proj_mat_fwd, c_exp_noise[eps_idx,:])

    # for i in range(max_dofs):
    #     f_exp[eps_idx,:] += b_exp[i]*maxpolyeval(2, x, i)*np.exp(-x**2)*x**2
    #     f_noise[eps_idx,:] += b_noise[i]*maxpolyeval(2, x, i)*np.exp(-x**2)*x**2

plt.subplot(1,4,1)
plt.semilogy(np.transpose(eigs_fwd))
plt.ylabel('Magnitude')
plt.xlabel('Eigenvalue no.')
plt.grid()
plt.legend(['$\epsilon=$'+str(eps) for eps in eps_all])

plt.subplot(1,4,2)
plt.semilogy(abs(np.transpose(b_exp)))
plt.ylabel('Magnitude')
plt.xlabel('Coefficient no.')
plt.grid()
plt.legend(['$\epsilon=$'+str(eps) for eps in eps_all])

plt.subplot(1,4,3)
plt.semilogy(abs(np.transpose(b_exp+b_noise)))
plt.ylabel('Magnitude')
plt.xlabel('Coefficient no.')
plt.grid()
plt.legend(['$\epsilon=$'+str(eps) for eps in eps_all])

plt.subplot(1,4,4)
plt.semilogy(abs(np.transpose(c_exp_noise)))
plt.gca().set_prop_cycle(None)
plt.semilogy(abs(np.transpose(b_exp+b_noise)),':')
plt.ylabel('Magnitude')
plt.xlabel('Coefficient no.')
plt.grid()
plt.legend(['$\epsilon=$'+str(eps) for eps in eps_all])

plt.show()

x = np.linspace(0, 5, 200)

eps_all = [-.1, .1]

for eps_idx, eps in enumerate(eps_all):
    eps_fwd = eps
    eps_bwd = -eps/(1.+eps)

    for i in range(max_dofs):
        norm = np.sum(maxpolyeval(2, xg, i)**2*wg)
        for j in range(max_dofs):
            proj_mat_fwd[i, j] = np.sum(maxpolyeval(2, xg, j)*maxpolyeval(2, xg/(1.+eps_fwd), i)*wg)/norm
            proj_mat_bwd[i, j] = np.sum(maxpolyeval(2, xg, j)*maxpolyeval(2, xg/(1.-eps_bwd), i)*wg)/norm

    w, v = np.linalg.eig(proj_mat_fwd)

    for i in range(max_dofs):
        plt.subplot(6, 9, i+1)
        plt.plot(np.real(v[:,i]))
        plt.plot(np.imag(v[:,i]))
        plt.title('$\lambda=$'+str(abs(w[i])))
        plt.grid()
    plt.show()

    for i in range(max_dofs):
        plt.subplot(6, 9, i+1)
        eigviz_re = np.zeros(np.shape(x))
        eigviz_im = np.zeros(np.shape(x))
        for j in range(max_dofs):
            eigviz_re += np.real(v[j,i])*maxpolyeval(2, x, j)*np.exp(-x**2)*x**2
            eigviz_im += np.imag(v[j,i])*maxpolyeval(2, x, j)*np.exp(-x**2)*x**2
        plt.plot(eigviz_re)
        plt.plot(eigviz_im)
        plt.title('$\lambda=$'+str(abs(w[i])))
        plt.grid()
    plt.show()

w, v = np.linalg.eig(proj_mat_bwd)

for i in range(max_dofs):
    plt.subplot(6, 9, i+1)
    plt.plot(np.real(v[:,i]))
    plt.plot(np.imag(v[:,i]))
plt.show()


coeffsum = np.zeros(len(eps_all))

eps_step = 0.001

for i in range(max_dofs):
    norm = np.sum(maxpolyeval(2, xg, i)**2*wg)
    for j in range(max_dofs):
        proj_mat_fwd[i, j] = np.sum(maxpolyeval(2, xg, j)*maxpolyeval(2, xg/(1.+eps_step), i)*wg)/norm
        proj_mat_bwd[i, j] = np.sum(maxpolyeval(2, xg, j)*maxpolyeval(2, xg/(1.-eps_step), i)*wg)/norm

for eps_idx, eps in enumerate(eps_all):

    x = np.linspace(0, 5, 200)
    b = a

    for j in range(int(eps/eps_step)):
        for i in range(max_dofs):
            if abs(b[i]) < 1.e-15:
                b[i] = 0
        b = np.matmul(proj_mat_fwd, b)

    for j in range(-int(eps/eps_step)):
        for i in range(max_dofs):
            if abs(b[i]) < 1.e-15:
                b[i] = 0
        b = np.matmul(proj_mat_bwd, b)

    fb = np.zeros(np.shape(x))

    for i in range(max_dofs):
        fb += b[i]*maxpolyeval(2, x/(1+eps), i)*np.exp(-(x/(1+eps))**2)/(1+eps)**3*x**2
    
    # plt.subplot(1,2,1)
    # plt.plot(x,(fa))
    # plt.plot(x,(fb), ':*')
    # plt.plot(x,(fc), ':o')

    # plt.subplot(1,2,2)
    # plt.semilogy(abs(a))
    # plt.semilogy(abs(b), ':*')
    # plt.semilogy(abs(c), ':o')
    # plt.show()

    if abs(eps) < 0.0001:
        plt.subplot(1,3,1)
        plt.plot(x,(fb), ':*')
        plt.subplot(1,3,2)
        plt.semilogy(abs(b), ':*')
    else:
        plt.subplot(1,3,1)
        plt.plot(x,(fb), ':o')
        plt.subplot(1,3,2)
        plt.semilogy(abs(b), ':o')

    coeffsum[eps_idx] = np.sum(abs(b))

plt.subplot(1,3,3)
plt.plot(eps_all, coeffsum,'-o')

plt.show()


for eps_idx, eps in enumerate(eps_all):
    # forward and backward change in thermal velocity
    eps_fwd = eps
    eps_bwd = -eps/(1.+eps)

    proj_mat_fwd = np.zeros([max_dofs, max_dofs])
    proj_mat_bwd = np.zeros([max_dofs, max_dofs])

    for i in range(max_dofs):
        norm = np.sum(maxpolyeval(2, xg, i)**2*wg)
        for j in range(max_dofs):
            proj_mat_fwd[i, j] = np.sum(maxpolyeval(2, xg, j)*maxpolyeval(2, xg/(1.+eps_fwd), i)*wg)/norm
            proj_mat_bwd[i, j] = np.sum(maxpolyeval(2, xg, j)*maxpolyeval(2, xg/(1.+eps_bwd), i)*wg)/norm

    x = np.linspace(0, 5, 200)
    b = np.matmul(proj_mat_fwd, a)
    c = np.matmul(proj_mat_bwd, a)

    fa = np.zeros(np.shape(x))
    fb = np.zeros(np.shape(x))
    fc = np.zeros(np.shape(x))

    for i in range(max_dofs):
        fa += a[i]*maxpolyeval(2, x, i)*np.exp(-x**2)*x**2
        fb += b[i]*maxpolyeval(2, x/(1+eps_fwd), i)*np.exp(-(x/(1+eps_fwd))**2)/(1+eps_fwd)**3*x**2
        fc += c[i]*maxpolyeval(2, x/(1+eps_bwd), i)*np.exp(-(x/(1+eps_bwd))**2)/(1+eps_bwd)**3*x**2
    
    # plt.subplot(1,2,1)
    # plt.plot(x,(fa))
    # plt.plot(x,(fb), ':*')
    # plt.plot(x,(fc), ':o')

    # plt.subplot(1,2,2)
    # plt.semilogy(abs(a))
    # plt.semilogy(abs(b), ':*')
    # plt.semilogy(abs(c), ':o')
    # plt.show()

    if abs(eps) < 0.0001:
        plt.subplot(1,3,1)
        plt.plot(x,(fb), ':*')
        plt.subplot(1,3,2)
        plt.semilogy(abs(b), ':*')
    else:
        plt.subplot(1,3,1)
        plt.plot(x,(fb), ':o')
        plt.subplot(1,3,2)
        plt.semilogy(abs(b), ':o')

    coeffsum[eps_idx] = np.sum(abs(b))

    for i in range(max_dofs):
        maxeig_fwd[eps_idx, i] = max(abs(np.linalg.eigvals(proj_mat_fwd[:i+1, :i+1])))
        maxeig_bwd[eps_idx, i] = max(abs(np.linalg.eigvals(proj_mat_bwd[:i+1, :i+1])))
        cycle_mat = np.matmul(proj_mat_bwd[:i+1, :i+1], proj_mat_fwd[:i+1, :i+1])
        maxeig_cyc[eps_idx, i] = max(abs(np.linalg.eigvals(cycle_mat)))

plt.subplot(1,3,3)
plt.plot(eps_all, coeffsum,'-o')

plt.show()

a = b


for i in range(max_dofs):
    if abs(a[i]) < 1.e-15:
        a[i] = 1.e-15

for eps_idx, eps in enumerate(eps_all):
    # forward and backward change in thermal velocity
    eps_fwd = eps
    eps_bwd = -eps/(1.+eps)

    proj_mat_fwd = np.zeros([max_dofs, max_dofs])
    proj_mat_bwd = np.zeros([max_dofs, max_dofs])

    for i in range(max_dofs):
        norm = np.sum(maxpolyeval(2, xg, i)**2*wg)
        for j in range(max_dofs):
            proj_mat_fwd[i, j] = np.sum(maxpolyeval(2, xg, j)*maxpolyeval(2, xg/(1.+eps_fwd), i)*wg)/norm
            proj_mat_bwd[i, j] = np.sum(maxpolyeval(2, xg, j)*maxpolyeval(2, xg/(1.+eps_bwd), i)*wg)/norm

    # proj_mat_fwd = np.linalg.inv(proj_mat_bwd)
    
    x = np.linspace(0, 5, 200)
    b = np.matmul(proj_mat_fwd, a)
    c = np.matmul(proj_mat_bwd, a)



    fa = np.zeros(np.shape(x))
    fb = np.zeros(np.shape(x))
    fc = np.zeros(np.shape(x))

    for i in range(max_dofs):
        fa += a[i]*maxpolyeval(2, x, i)*np.exp(-x**2)*x**2
        fb += b[i]*maxpolyeval(2, x/(1+eps_fwd), i)*np.exp(-(x/(1+eps_fwd))**2)/(1+eps_fwd)**3*x**2
        fc += c[i]*maxpolyeval(2, x/(1+eps_bwd), i)*np.exp(-(x/(1+eps_bwd))**2)/(1+eps_bwd)**3*x**2
    
    # plt.subplot(1,2,1)
    # plt.plot(x,(fa))
    # plt.plot(x,(fb), ':*')
    # plt.plot(x,(fc), ':o')

    # plt.subplot(1,2,2)
    # plt.semilogy(abs(a))
    # plt.semilogy(abs(b), ':*')
    # plt.semilogy(abs(c), ':o')
    # plt.show()

    if abs(eps) < 0.0001:
        plt.subplot(1,3,1)
        plt.plot(x,(fb), ':*')
        plt.subplot(1,3,2)
        plt.semilogy(abs(b), ':*')
    else:
        plt.subplot(1,3,1)
        plt.plot(x,(fb), ':o')
        plt.subplot(1,3,2)
        plt.semilogy(abs(b), ':o')

    coeffsum[eps_idx] = np.sum(abs(b))

    for i in range(max_dofs):
        maxeig_fwd[eps_idx, i] = max(abs(np.linalg.eigvals(proj_mat_fwd[:i+1, :i+1])))
        maxeig_bwd[eps_idx, i] = max(abs(np.linalg.eigvals(proj_mat_bwd[:i+1, :i+1])))
        cycle_mat = np.matmul(proj_mat_bwd[:i+1, :i+1], proj_mat_fwd[:i+1, :i+1])
        maxeig_cyc[eps_idx, i] = max(abs(np.linalg.eigvals(cycle_mat)))

plt.subplot(1,3,3)
plt.plot(eps_all, coeffsum,'-o')

plt.show()

plt.semilogy(eps_all, maxeig_fwd)
plt.semilogy(eps_all, np.ones(np.shape(eps_all)), ':k')
plt.xlabel('Relative change in temperature')
plt.ylabel('Largest eigenvalue')
plt.grid()
plt.show()

print(maxeig_fwd)
plt.subplot(1,2,1)
plt.semilogy(eps_all, maxeig_fwd)
plt.semilogy(eps_all, np.ones(np.shape(eps_all)), ':k')
plt.subplot(1,2,2)
plt.semilogy(eps_all, maxeig_cyc)
plt.semilogy(eps_all, np.ones(np.shape(eps_all)), ':k')
plt.show()

# plt.grid()
# plt.xlabel('Number of polynomials')
# plt.ylabel('Largest eigenvalue')
# plt.legend()
# plt.show()

# plt.plot(eps_all, N_crit, 'o-')
# plt.grid()
# plt.xlabel('Relative change in temperature')
# plt.ylabel('Critical number of polynomials')
# plt.show()


# print(cycle_mat)
print(abs(np.linalg.eigvals(cycle_mat)))
print(np.linalg.eigvals(proj_mat_bwd))
print(np.linalg.eigvals(proj_mat_fwd))

x = np.linspace(0, 5, 200)
a = np.random.rand(max_dofs)*np.exp(-np.linspace(0, 10, max_dofs))
b = np.matmul(proj_mat_fwd, a)
c = np.matmul(proj_mat_bwd, a)

# for i in range(40):
#     c = np.matmul(proj_mat_fwd, c)
#     plt.semilogy(abs(c), '*')
# plt.show()

# for i in range(10):
#     b = np.matmul(proj_mat_fwd, c)
#     c = np.matmul(proj_mat_bwd, b)

fa = np.zeros(np.shape(x))
fb = np.zeros(np.shape(x))
fc = np.zeros(np.shape(x))

for i in range(max_dofs):
    fa += a[i]*maxpolyeval(2, x, i)*np.exp(-x**2)*x**2
    fb += b[i]*maxpolyeval(2, x*(1+eps_fwd), i)*np.exp(-(x*(1+eps_fwd))**2)*x**2
    fc += c[i]*maxpolyeval(2, x*(1+eps_bwd), i)*np.exp(-(x*(1+eps_bwd))**2)*x**2

plt.plot(x,(fa))
plt.plot(x,(fb))
plt.plot(x,(fc),'o')
# plt.semilogy(abs(a), 'o-')
# plt.semilogy(abs(b), 'o-')
# plt.semilogy(abs(c), '*')
# plt.show()
