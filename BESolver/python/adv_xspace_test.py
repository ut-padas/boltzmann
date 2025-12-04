import numpy as xp
import mesh
import scipy.interpolate
import matplotlib.pyplot as plt
import sys
sys.path.append("./plot_scripts")
import plot_utils


def fd_coefficients(x, order):
    I = xp.eye(x.shape[0])
    L = [scipy.interpolate.lagrange(x, I[i]) for i in range(len(x))]

    Vdx = xp.array([L[i].deriv(order)(x) for i in range(len(L))]).T
    return Vdx

def upwinded_dx(x, dir):
    Np = len(x)
    Dx = xp.zeros((Np, Np))
    
    if (dir == "LtoR"):
        Dx[0, 0:2] = fd_coefficients(x[0:2], 1)[0]
    
        for i in range(1, Np):
            Dx[i, (i-1):(i+1)] = fd_coefficients(x[i-1:i+1], 1)[1]
    
    else:
        assert dir == "RtoL"

        for i in range(0, Np-1):
            Dx[i, i:(i+2)] = fd_coefficients(x[i:i+2], 1)[0]

        Dx[-1, -2:] = fd_coefficients(x[-2:], 1)[1]

    return Dx

def xmvt_linear_inp(x, c, dt):
    xx = xp.concatenate((xp.array([x[0] - abs(c) * dt]), x, xp.array([x[-1] + abs(c) * dt])))
    yy = xp.concatenate((xp.array([xx[0]]), x - c * dt, xp.array([xx[-1]])))
    return scipy.interpolate.interp1d(xx, xp.eye(len(xx)), axis=0, kind="linear")(yy)
    
def adv_solve(c, xx, u0, steps, dt, method):

    y               = xp.copy(u0)
    Imat            = xp.eye(len(op.xp))
    uwDxL           = upwinded_dx(xx, "LtoR")
    uwDxR           = upwinded_dx(xx, "RtoL")
    uwDxL[0 , :]    = 0.0
    uwDxR[-1, :]    = 0.0

    cbDxL           = xp.zeros_like(op.Dp)
    cbDxR           = xp.zeros_like(op.Dp)

    cbDxL[1: , :]    = op.Dp[1: ,:]
    cbDxR[:-1, :]    = op.Dp[:-1,:]

    if (c >=0):
        uwDx = uwDxL
        cbDx = cbDxL
    else:
        uwDx = uwDxR
        cbDx = cbDxR
    
    if method==0:
        P    = xp.linalg.inv(Imat + c * dt * uwDx)
        #print(P)
        print(P[1])
        print(P[2])
        print(P[3])
        print(P[10])
    elif method == 1:
        P    = xp.linalg.inv(Imat + c * dt * cbDx)
    elif method == 2:
        P    = xmvt_linear_inp(xx, c, dt)
        y    = xp.concatenate((xp.array([0]), u0, xp.array([0])))
    else:
        raise NotImplementedError
    
    
    if (c>0):
        for i in range(steps):
            y[0] = 0.0
            y    = P @ y
    else:
        for i in range(steps):
            y[-1] = 0.0
            y     = P @ y
    
    if (method == 2):
        return y[1:-1]
    
    return y
        

c     = 1.0
#Nps   = [512, 1024, 2048]
Nps   = [32]

# 0 : FD with upwinding
# 1 : Chebyshev 
# 2 : linear interp 

sol     = {0:list(), 1:list(), 2:list()}
xx_grid = plot_utils.op(Nps[-1], use_tab_data=0).xp 
dt_all  = list()
for Np in Nps:
    op    = plot_utils.op(Np, use_tab_data=0) #mesh.collocation_op.cheb_collocation_1d(Np)
    xx    = op.xp
    dt    = 0.25 * xp.min(xx[1:]-xx[0:-1])
    steps = int(0.001/dt)
    dt_all.append(dt)

    # single advection step
    g0    = -xx**2 + 1.2
    u0    = adv_solve(c, xx, g0, steps, dt, 0)
    u1    = adv_solve(c, xx, g0, steps, dt, 1)
    u2    = adv_solve(c, xx, g0, steps, dt, 2)

    P     = op.interp_op_galerkin(xx_grid)

    sol[0].append(P @ u0)
    sol[1].append(P @ u1)
    sol[2].append(P @ u2)


lbl=[r"Nx=%d dt=%.4E"%(Nps[i], dt_all[i]) for i in range(len(Nps))]

plt.subplot(2, 2, 1)
plt.title(r"BE + 1st order FD (upwinded)")
for i in range(len(Nps)):
    plt.plot(xx_grid, sol[0][i], label=lbl[i])
    plt.plot(xx_grid, sol[1][i], label=lbl[i])
plt.grid(visible=True)
plt.legend()
plt.xlabel(r"x")
plt.ylabel(r"u(x)")

plt.subplot(2, 2, 2)
plt.title(r"BE + Chebyshev")
for i in range(len(Nps)):
    plt.plot(xx_grid, sol[1][i], label=lbl[i])
plt.grid(visible=True)
plt.legend()
plt.xlabel(r"x")
plt.ylabel(r"u(x)")

plt.subplot(2, 2, 3)
plt.title(r"f(x-vt) linear interp")
for i in range(len(Nps)):
    plt.plot(xx_grid, sol[2][i], label=lbl[i])

plt.grid(visible=True)
plt.legend()
plt.xlabel(r"x")
plt.ylabel(r"u(x)")

plt.subplot(2, 2, 4)
plt.title(r"relative errors")
r0     = xp.array([xp.linalg.norm(sol[0][i] - sol[0][-1])/xp.linalg.norm(sol[0][-1]) for i in range(len(Nps)-1)])
r1     = xp.array([xp.linalg.norm(sol[1][i] - sol[1][-1])/xp.linalg.norm(sol[1][-1]) for i in range(len(Nps)-1)])
r2     = xp.array([xp.linalg.norm(sol[2][i] - sol[2][-1])/xp.linalg.norm(sol[2][-1]) for i in range(len(Nps)-1)])
r3     = xp.array([r0[0]/(2**i) for i in range(len(Nps)-1)])
run_id = ["r%d"%(i) for i in range(len(Nps)-1)]

plt.semilogy(r0, "x-", label=r"BE + upwinded FD")
plt.semilogy(r1, "o-", label=r"BE + Chebyshev")
plt.semilogy(r2, "^-", label=r"$f_0(x-vt) w. linear interp$")
plt.semilogy(r3, "--", label=r"")
plt.xticks(xp.arange(len(run_id)), run_id)

plt.xlabel(r"run id")
plt.ylabel(r"relative error $||u - u_h||/||u_h||$")

plt.grid(visible=True)
plt.legend()



plt.show()
plt.close()



# plt.subplot(1, 3, 1)
# plt.plot(xx[-1], g0_all[-1], label=r"IC")
# plt.plot(xx, u0, label=r"BE + upwinded FD (cheb-grid)")
# plt.plot(xx, u1, label=r"BE + cheb-collocation")
# plt.plot(xx, u2, label=r"analytical (linear-inp)")

# plt.subplot(1, 3, 2)
# plt.semilogy(xx, g0, label=r"IC")
# plt.semilogy(xx, u0, label=r"BE + upwinded FD (cheb-grid)")
# plt.semilogy(xx, u1, label=r"BE + cheb-collocation")
# plt.semilogy(xx, u2, label=r"analytical (linear-inp)")

# plt.subplot(1, 3, 3)


# plt.legend()
# plt.grid(visible=True)
# plt.show()


