import numpy as np
import matplotlib.pyplot as plt
from numpy.ma.core import concatenate
from maxpoly import *

# max dofs used for represenation
max_dofs = 129

# change in thermal velocity
# eps_all = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
# eps_all = np.linspace(-0.45, 0.55, 21)
eps_all = [0, -0.02, -0.04, -0.06]
# eps_all = np.linspace(0.05, 0.55, 11)
# eps_all = np.arange(-0.15,0.201,0.01)
# eps_all = np.arange(-0.15,0.001,0.01)
# eps_all = np.arange(-0.15,0.2,0.05)
# eps_all = [-0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15, 0.20]
# eps_all = [-0.015, -0.010, -0.005, 0, 0.005, 0.010, 0.015, 0.020]
# eps_all = [0.0015]

# use Gaussian quadratures or trapezoidal integration for projection
use_gauss_for_proj = True

# order of Gaussian quadrature used for projection
gauss_q_order_proj = 256

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

a_exp = [ 2.3597304924146956e+00, -6.6838750506123203e-08, -1.1126522946307354e-07,\
 -7.7276925238944848e-08, -3.1123748682675316e-08, -1.0056393044375676e-08,\
  6.4521892147112488e-09,  8.6711909412693096e-09,  1.1472264184607000e-09,\
 -3.3595322440861861e-09, -2.2907787091943262e-09,  1.4429143792550318e-09, \
  1.4430550032057249e-09, -4.7546758027356785e-10, -9.6360593417524337e-10,\
  2.5261723603719437e-10,  5.8176705491748456e-10, -1.7559924192098384e-10,\
 -3.4689810489367663e-10,  1.4662654535522383e-10,  2.0313226560667125e-10,\
 -1.3554190716327607e-10, -1.0010827662628211e-10,  1.1256505484104861e-10,\
  3.5205515486372264e-11, -8.5833001096712454e-11,  5.1638569950075070e-12,\
  5.5890617870491442e-11, -2.4983281633397713e-11, -2.7672054848020177e-11,\
  2.8677843075480124e-11,  6.2874840157305164e-12, -2.2782525288482723e-11,\
  6.8048132981226189e-12,  1.2302433656037817e-11, -1.1083916951136016e-11,\
 -2.6865349122388899e-12,  9.2962363355062527e-12, -3.4289400958592088e-12,\
 -4.5345996399649997e-12,  5.1224383047177458e-12,  1.0006270325492275e-13,\
 -3.7283588887295711e-12,  2.3303706844126646e-12,  1.1337098554568222e-12,\
 -2.4141976565382252e-12,  8.1087191043020433e-13,  1.1859020806785862e-12,\
 -1.4166733156750774e-12,  1.4982570900455221e-13,  9.1568355644082894e-13,\
 -7.7104409711060567e-13, -1.0555850904681058e-13,  6.5351772739431712e-13,\
 -4.3147296586141505e-13, -1.3784575199855533e-13,  4.1934005769861135e-13,\
 -2.2995244018351793e-13, -1.2082754185620414e-13,  2.6131542539649000e-13,\
 -1.2032789776365474e-13, -9.1247262550718102e-14,  1.5581699408763097e-13,\
 -5.0409193576213120e-14, -8.4095541004268840e-14,  1.1424333467148804e-13,\
 -3.4086941327232193e-14, -6.2312248624988356e-14,  8.8859711442967216e-14,\
 -3.7801781975073455e-14, -3.1372948481493638e-14,  5.9390095317292000e-14,\
 -3.7445916763516878e-14, -4.1613245378526665e-15,  2.7019554633221179e-14,\
 -2.0063127731435236e-14,  3.0254766108076447e-16,  1.1508104739665659e-14,\
 -8.2266791305876477e-15, -1.9496112575948632e-15,  5.4927021491350455e-15,\
  2.1245461312530409e-16, -9.0942462344704539e-15,  1.1957566241120077e-14,\
 -5.6256351564177753e-15, -4.4942505117103214e-15,  1.1698756039424209e-14,\
 -1.2232926085628828e-14,  5.6411619211175453e-15,  2.7983160147155399e-15,\
 -9.0174361936453183e-15,  9.6700411561034029e-15, -7.1659206089073994e-15,\
  1.3763795155794840e-15,  3.6213212470027735e-15, -7.6302841810364361e-15,\
  8.9850240075308049e-15, -7.4415778212995924e-15,  1.7576379589260231e-15,\
  5.5497065223666954e-15, -1.1928414326173245e-14,  1.4800696102035887e-14,\
 -1.1342909767790092e-14,  3.3013091500766786e-15,  7.6401015158366637e-15,\
 -1.5300462710912575e-14,  1.7262238083028678e-14, -1.2093079706849550e-14,\
  2.4492979953370197e-15,  7.9370460817364775e-15, -1.5959421645990318e-14,\
  1.7453910545558523e-14, -1.4282873879554496e-14,  6.1178255054608702e-15,\
  2.6878943464544919e-15, -9.4061453957501547e-15,  1.2443835163696696e-14,\
 -1.2020803316678398e-14,  7.1587348573569044e-15, -2.3700827928987660e-15,\
 -3.0330037430708051e-15,  6.4714848330109353e-15, -7.2792825432442092e-15,\
  6.0580042761956570e-15, -2.4281839350281631e-15, -1.8503932264799231e-15,\
  4.2215561895535527e-15, -6.5182848321163693e-15,  6.2668189179640925e-15]

# for i in range(max_dofs):
#     if abs(a_exp[i]) < 1.e-13:
#         a_exp[i] = 0

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
            # proj_mat_fwd_over[i, j] = np.sum(maxpolyeval(2, xg, j)*maxpolyeval(2, xg/(1.+overstep*eps_fwd), i)*wg)/norm
            # proj_mat_fwd_back[i, j] = np.sum(maxpolyeval(2, xg, j)*maxpolyeval(2, xg/(1.+eps_fwd)*(1.+overstep*eps_fwd), i)*wg)/norm
            # proj_mat_bwd[i, j] = np.sum(maxpolyeval(2, xg, j)*maxpolyeval(2, xg/(1.+eps_bwd), i)*wg)/norm
            # proj_mat_bwd2[i, j] = np.sum(maxpolyeval(2, xg, j)*maxpolyeval(2, xg/(1.+2.*eps_bwd), i)*wg)/norm

    # proj_mat_fwd_pen = np.matmul(np.transpose(proj_mat_bwd),proj_mat_bwd) + pen*np.eye(max_dofs,max_dofs)
    # proj_mat_fwd_pen = np.matmul(np.linalg.inv(proj_mat_fwd_pen),np.transpose(proj_mat_bwd))
    # proj_mat_fwd = np.linalg.inv(proj_mat_bwd)
    # proj_mat_fwd_over = np.matmul(proj_mat_fwd_back, proj_mat_fwd_over)
    # proj_mat_bwd2 = np.matmul(proj_mat_fwd, proj_mat_bwd2)
    eigs_fwd[eps_idx,:] = np.reshape(abs(np.linalg.eigvals(proj_mat_fwd)), [1,max_dofs])
    # eigs_fwd2[eps_idx,:] = np.reshape(abs(np.linalg.eigvals(proj_mat_fwd_over)), [1,max_dofs])
    # eigs_bwd[eps_idx,:] = np.reshape(abs(np.linalg.eigvals(proj_mat_bwd)), [1,max_dofs])
    # eigs_bwd2[eps_idx,:] = np.reshape(abs(np.linalg.eigvals(proj_mat_bwd2)), [1,max_dofs])

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
        b_exp[eps_idx,:] = np.matmul(proj_mat_fwd, a_exp)
        b_noise[eps_idx,:] = np.matmul(proj_mat_fwd, a_noise)
        # b_exp[eps_idx,:] = np.matmul(proj_mat_fwd_pen, a_exp)
        # b_noise[eps_idx,:] = np.matmul(proj_mat_fwd_pen, a_noise)
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

    n = 0

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

# plt.subplot(1,4,2)
plt.subplot(1,1,1)
plt.semilogy(abs(np.transpose(b_exp)))
plt.ylabel('Magnitude')
plt.xlabel('Coefficient no.')
plt.grid()
plt.legend(['$\epsilon=$'+str(eps) for eps in eps_all])

plt.show()

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
