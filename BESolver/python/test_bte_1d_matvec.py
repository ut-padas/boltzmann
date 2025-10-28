import numpy as np

xp     = np
num_p  = 15
num_sh = 2 
num_x  = 13 
num_vt = 10 
Dx     = np.random.rand(num_x, num_x)
Ex     = np.random.rand(num_x)
tau    = 0.1#1/13.56e6

xp_vt_l = np.array([i * num_vt + np.arange(0, 2) for i in range(num_p)])
xp_vt_r = np.array([i * num_vt + np.arange(num_vt-2, num_vt) for i in range(num_p)])

Cop    = np.random.rand(num_p * num_sh, num_p * num_sh)
Av     = np.random.rand(num_p * num_sh, num_p * num_sh)
Ax     = np.random.rand(num_p * num_vt, num_p * num_vt)
Ps     = np.random.rand(num_p * num_sh, num_p * num_vt)
Po     = np.random.rand(num_p * num_vt, num_p * num_sh)

def rhs_vx(x):
    x  = x.reshape((num_p * num_vt, num_x))
    xs = np.dot(Ps, x)
    y  = tau * np.dot(Po, np.dot(Cop, xs) + Ex * np.dot(Av, xs)) -np.dot(Ax, np.dot(x, Dx.T))

    y[xp_vt_l, 0 ]  = 0.0
    y[xp_vt_r, -1 ] = 0.0
    return y.reshape((-1))

def rhs_vx_transpose(x):
    x  = x.reshape((num_p * num_vt, num_x))
    x[xp_vt_l, 0 ]  = 0.0
    x[xp_vt_r, -1 ] = 0.0
    
    xs = np.dot(Po.T, x)
    y  = tau * (np.dot(Ps.T, np.dot(Cop.T, xs) + Ex * np.dot(Av.T, xs))) - np.dot(np.dot(Ax.T, x) , Dx)
    y  = y.reshape((-1))
    return y


x    = xp.random.rand(num_p * num_vt * num_x)

y1   = Ex * x.reshape((num_p * num_vt, num_x))
y2   = xp.kron(np.eye(num_p * num_vt), xp.diag(Ex)) @ x

print("%.2E"%(xp.linalg.norm(y2-y1.reshape((-1)))/xp.linalg.norm(y1.reshape((-1)))))

Ndof = num_p * num_vt * num_x
#L1   = np.zeros((Ndof, Ndof))
#L2   = np.zeros((Ndof, Ndof))

# for i in range(num_p * num_vt):
#     L1[i * num_x : (i+1) * num_x, i * num_x : (i+1) * num_x] = Dx

# for i in range(num_p * num_vt):
#     for j in range(num_p * num_vt):
#         for k in range(num_x):
#             L2 [ i * num_x + k, j * num_x +  k ] = Ax[i, j]


L1 = xp.kron(xp.eye(num_p * num_vt), Dx)
L2 = xp.kron(Ax, xp.eye(num_x))


Cop_o = Po @ Cop @ Ps
Av_o  = Po @ Av @ Ps

# Lv    = np.zeros((Ndof, Ndof))
# for i in range(num_p * num_vt):
#     for j in range(num_p * num_vt):
#         for k in range(num_x):
#             Lv[i * num_x + k, j * num_x +  k] = tau * (Cop_o[i,j] + Ex[k] * Av_o[i,j])

Lv = tau * (xp.kron(Cop_o, xp.eye(num_x)) + xp.kron(Av_o, xp.diag(Ex))) 
#print(Lv.shape)

Lx  = -xp.dot(L2, L1)
L   = Lv + Lx

L[xp_vt_l * num_x + 0, :]        = 0.0
L[xp_vt_r * num_x + num_x-1, : ] = 0.0

y1  = rhs_vx(x)
y2  = xp.dot(L , x)


y3  = rhs_vx_transpose(x)
y4  = xp.dot(L.T , x)
print(xp.linalg.norm(y1 - y2)/xp.linalg.norm(y1))
print(xp.linalg.norm(y3 - y4)/xp.linalg.norm(y4))
        




    


