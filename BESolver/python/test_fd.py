"""
@package simple fd test to check uniform grid creation and stencil application. 
"""

import numpy as np
from numpy.core.fromnumeric import shape
import unigrid 
import fd_derivs
import matplotlib.pyplot as plt

f1d = lambda x : np.sin(2*np.pi*x[0])
f2d = lambda x : np.sin(2*np.pi*x[0]) * np.sin(2*np.pi*x[1])
f3d = lambda x : np.sin(2*np.pi*x[0]) * np.sin(2*np.pi*x[1]) * np.sin(2*np.pi*x[2])

dxf1d = lambda x : np.cos(2*np.pi*x[0]) * 2*np.pi
dxf2d = lambda x : np.cos(2*np.pi*x[0]) * np.sin(2*np.pi*x[1]) * 2*np.pi
dxf3d = lambda x : np.cos(2*np.pi*x[0]) * np.sin(2*np.pi*x[1]) * np.sin(2*np.pi*x[2]) * 2*np.pi

grid_min = np.array([ -0.5 ,-0.5])
grid_max = np.array([  0.5 , 0.5])
grid_res = np.array([  0.01 , 0.01])
mesh2d = unigrid.UCartiesianGrid(2,grid_min,grid_max,grid_res)

fv = unigrid.init_vec(mesh2d,f2d)
fv = fv[:,:,0]

print("2d created vec shape : ", fv.shape)
plt.imshow(fv, interpolation='none')
plt.colorbar()
plt.savefig("f2d.png")
plt.close()


grid_min = np.array([ -0.5 ,-0.5, -0.5])
grid_max = np.array([  0.5 , 0.5,  0.5])
grid_res = np.array([  0.01 ,0.01, 0.01])
mesh3d = unigrid.UCartiesianGrid(3,grid_min,grid_max,grid_res)

fv  = unigrid.init_vec(mesh3d,f3d)
dxfv = unigrid.init_vec(mesh3d,dxf3d)

dxfv_s = fd_derivs.deriv42(fv,grid_res[0],0)
#dyfv_s = fd_derivs.deriv42(fv,grid_res[0],1)
#dzfv_s = fd_derivs.deriv42(fv,grid_res[0],2)
#print("norm l2 |dfv - Dfv| = ", np.linalg.norm(dfv-dfv_s,2))

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
#plt.imshow(fv[:,:,20,0],axis=ax1)
z_index = 1
ax1.set_title("f(x)")
p1 = ax1.imshow(fv[:,:,z_index,0])
plt.colorbar(p1,ax=ax1)

ax2.set_title("df(x)")
p2 = ax2.imshow(dxfv[:,:,z_index,0])
plt.colorbar(p2,ax=ax2)

ax3.set_title("Df(x)")
p3 = ax3.imshow(dxfv_s[:,:,z_index,0])
plt.colorbar(p3,ax=ax3)

diff_v = dxfv - dxfv_s 
ax4.set_title("|df(x) - Df(x)|")
p4 = ax4.imshow(diff_v[:,:,z_index,0])
plt.colorbar(p4,ax=ax4)

#plt.imshow(fv, interpolation='none')
#plt.show()
plt.savefig("f3d.png")
plt.close()




