import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
from numpy.ma.core import concatenate
from maxpoly import *

# max dofs used for represenation
max_dofs = 64

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
gauss_q_order_proj = 200

G = 0

[xg, wg] = maxpolygauss(gauss_q_order_proj, G)

for i in range(max_dofs):
    f_coeffs[i] = np.sum(maxpolyeval(G, xg, i)*f(xg)*wg)
    fd_coeffs_exact[i] = np.sum(maxpolyeval(G, xg, i)*fd(xg)*wg)
    xf_coeffs_exact[i] = np.sum(maxpolyeval(G, xg, i)*xf(xg)*wg)

# assemble matrix
D = diff_matrix(G, max_dofs-1)
print(D[:4,:4])
fd_coeffs_app = np.matmul(D, f_coeffs)

L = lift_matrix(G, max_dofs-1)
xf_coeffs_app = np.matmul(L, f_coeffs)

# for i in range(max_dofs):
#     fd_coeffs_app[i] = sum(D[i,:]*f_coeffs)
#     xf_coeffs_app[i] = sum(L[i,:]*f_coeffs)

xplot = linspace(0,10,100)
f_app = np.zeros(len(xplot))
fd_app = np.zeros(len(xplot))
xf_app = np.zeros(len(xplot))

# for i in range(max_dofs):
#     f_app += f_coeffs[i]*maxpolyeval(G, xplot, i)
#     fd_app += fd_coeffs_app[i]*maxpolyeval(G, xplot, i)
#     xf_app += xf_coeffs_app[i]*maxpolyeval(G, xplot, i)


f_app = maxpolyserieseval(G, xplot, f_coeffs)
fd_app = maxpolyserieseval(G, xplot, fd_coeffs_app)
xf_app = maxpolyserieseval(G, xplot, xf_coeffs_app)

plt.subplot(1,2,1)
# plt.plot(xplot, xf(xplot)*xplot**G*np.exp(-xplot**2),'-')
# plt.plot(xplot, xf_app*xplot**G*np.exp(-xplot**2),'*')
plt.plot(xplot, fd(xplot)*np.exp(-xplot**2),'-')
plt.plot(xplot, fd_app*np.exp(-xplot**2),'o')
plt.plot(xplot, f(xplot)*np.exp(-xplot**2),'-')
plt.plot(xplot, f_app*np.exp(-xplot**2),'^')

plt.subplot(1,2,2)
# plt.semilogy(abs(xf_coeffs_exact),'-')
# plt.semilogy(abs(xf_coeffs_app),'*')
plt.semilogy(abs(fd_coeffs_exact),'-')
plt.semilogy(abs(fd_coeffs_app),'*')
# plt.semilogy(abs(xf_coeffs_exact-xf_coeffs_app),'o-')
# plt.semilogy(abs(fd_coeffs_exact-fd_coeffs_app),'*-')

plt.show()