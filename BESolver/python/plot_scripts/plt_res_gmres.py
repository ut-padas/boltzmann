import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

Nr  = 257
Nvt =  32
Nl  =   3
Np  = 200 

res   = np.load("../res1.npy")
rr    = np.load("../rr.npy")
xp_vt = np.load("../vt.npy")
 
res = res.reshape((Nr ,  Nvt, Np))

vmin = 1e-10#np.min(np.abs(res))
vmax = 1e2#np.max(np.abs(res))


xp_vt_l       = np.array([i * Nvt + j for i in range(Nr) for j in list(np.where(xp_vt <= 0.5 * np.pi)[0])])
xp_vt_r       = np.array([i * Nvt + j for i in range(Nr) for j in list(np.where(xp_vt > 0.5 * np.pi)[0])])

plt.figure(figsize=(20, 8), dpi=300)

for theta_idx in range(32):
    plt.subplot(4, 8, theta_idx+1)
    
    plt.imshow(np.abs(res[:, theta_idx, :]), norm=LogNorm(vmin, vmax))
    
    plt.colorbar()

plt.tight_layout()    
plt.savefig("res1.png")
plt.close()