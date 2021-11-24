import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
from numpy.ma.core import concatenate
from maxpoly import *

# max dofs used for represenation
max_dofs = 20

# function and exact derivative
f = lambda x: np.sin(x)
fd = lambda x: np.cos(x)
xf = lambda x: x*np.sin(x)
# f = lambda x: x*x
# fd = lambda x: 0
# xf = lambda x: x*x*x

# project onto maxwell polynomials
f_coeffs = np.zeros(max_dofs)
fd_coeffs_exact = np.zeros(max_dofs)
fd_coeffs_app = np.zeros(max_dofs)
xf_coeffs_exact = np.zeros(max_dofs)
xf_coeffs_app = np.zeros(max_dofs)
gauss_q_order_proj = 100

[xg, wg] = maxpolygauss(gauss_q_order_proj)

for i in range(max_dofs):
    f_coeffs[i] = np.sum(maxpolyeval(xg, i)*f(xg)*wg)
    fd_coeffs_exact[i] = np.sum(maxpolyeval(xg, i)*fd(xg)*wg)
    xf_coeffs_exact[i] = np.sum(maxpolyeval(xg, i)*xf(xg)*wg)

# assemble matrix
D = diff_matrix(max_dofs-1)
fd_coeffs_app = np.matmul(D, f_coeffs)

L = lift_matrix(max_dofs-1)
xf_coeffs_app = np.matmul(L, f_coeffs)

xplot = linspace(0,5,100)
f_app = np.zeros(len(xplot))
fd_app = np.zeros(len(xplot))
xf_app = np.zeros(len(xplot))

for i in range(max_dofs):
    f_app += f_coeffs[i]*maxpolyeval(xplot, i)
    fd_app += fd_coeffs_app[i]*maxpolyeval(xplot, i)
    xf_app += xf_coeffs_app[i]*maxpolyeval(xplot, i)

plt.subplot(1,2,1)
plt.plot(xplot, xf(xplot),'-')
plt.plot(xplot, xf_app,'*')
plt.plot(xplot, fd(xplot),'-')
plt.plot(xplot, fd_app,'*-')

plt.subplot(1,2,2)
# plt.semilogy(abs(xf_coeffs_exact),'-')
# plt.semilogy(abs(xf_coeffs_app),'*')
# plt.semilogy(abs(fd_coeffs_exact),'-')
# plt.semilogy(abs(fd_coeffs_app),'*')
plt.semilogy(abs(xf_coeffs_exact-xf_coeffs_app),'o-')
plt.semilogy(abs(fd_coeffs_exact-fd_coeffs_app),'*-')

plt.show()