import numpy as np
import h5py 
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import cupy as cp

import plot_utils

def load_run_args(fname):
    args   = dict()
    
    f  = open(fname)
    st = f.read().strip()
    st = st.split(":")[1].strip()
    st = st.split("(")[1]
    st = st.split(")")[0].strip()
    st = st.split(",")

    for s in st:
        kv = s.split("=")
        if len(kv)!=2 or kv[0].strip()=="collisions":
            continue
        args[kv[0].strip()]=kv[1].strip().replace("'", "")

    return args

# folder_name = "../1dbte_rom/E0.00_vx_ord"
# ff          = h5py.File("%s/1d_bte_ic_T1500_32x8x32_dt_1.0E-03.h5"%(folder_name), 'r')
folder_name = "../1dbte_rom/E10.00_vx_ord"
ff          = h5py.File("%s/1d_bte_ic_T2000_32x8x32_dt_1.0E-03.h5"%(folder_name), 'r')

np0     = 8e16
args    = load_run_args("%s/1d_bte_args.txt"%(folder_name))
mass_op = np.load("%s/1d_bte_bte_mass_op.npy"%(folder_name))
temp_op = np.load("%s/1d_bte_bte_temp_op.npy"%(folder_name))
Ps      = np.load("%s/1d_bte_bte_po2sh.npy"  %(folder_name))
Po      = np.load("%s/1d_bte_bte_psh2o.npy"  %(folder_name))
Lop     = np.load("%s/1d_bte_Lop_32x8x32.npy"%(folder_name))

F       = np.array(ff["F"][()])
ts      = np.array(ff["time[T]"][()])
xx      = np.array(ff["x[-1,1]"][()])

Fs      = np.einsum("vl,tilx->tivx", Ps,       F)
ne      = np.einsum("l,tilx->tix"  , mass_op, Fs)
Te      = np.einsum("l,tilx->tix"  , temp_op, np.einsum("tilx,tix->tilx", Fs, (1/ne)))

cols    = 5
rows    = 3
num_ic  = F.shape[1]

plt.figure(figsize=(6 * cols + 2 ,  3 * cols + 2), dpi=300)
plt_idx = 1
for tidx in range(0, len(ts), len(ts) // (rows * cols) + 1):
    #print(ts[tidx])
    plt.subplot(rows, cols, plt_idx)
    for ic_idx in range(F.shape[1]):
        plt.plot(xx, ne[tidx, ic_idx] * np0, label=r"ic-%d"%(ic_idx))
    
    plt.xlabel(r"x")
    plt.ylabel(r"$n_e[m^{-3}]$")
    plt.grid(visible=True)
    plt.title("time = %.2f T"%(ts[tidx]))
    if(tidx == 0):
        plt.legend()

    plt_idx+=1

plt.tight_layout()
plt.savefig("%s/ne.png"%(folder_name))
plt.close()


plt.figure(figsize=(6 * cols + 2 ,  3 * cols + 2), dpi=300)
plt_idx = 1
for tidx in range(0, len(ts), len(ts) // (rows * cols) + 1):
    #print(ts[tidx])
    plt.subplot(rows, cols, plt_idx)
    for ic_idx in range(F.shape[1]):
        plt.plot(xx, Te[tidx, ic_idx], label=r"ic-%d"%(ic_idx))
    
    plt.xlabel(r"x")
    plt.ylabel(r"$T_e[eV]$")
    plt.grid(visible=True)
    plt.title("time = %.2f T"%(ts[tidx]))
    if(tidx == 0):
        plt.legend()

    plt_idx+=1

plt.tight_layout()
plt.savefig("%s/Te.png"%(folder_name))
plt.close()

plt.figure(figsize=(8, 8), dpi=300)

plt.subplot(2, 2, 1)
Ndof     = Lop.shape[0]
svd_split= Ndof - Ndof // 10
u, s, vt = cp.linalg.svd(cp.asarray(Lop))
u, s, vt = cp.asnumpy(u), cp.asnumpy(s), cp.asnumpy(vt)
v        = vt.T

F        = F.reshape((len(ts), num_ic, -1))
FH       = F @ v[:, 0:svd_split]
FL       = F @ v[:, svd_split: ]

idx = np.arange(s.shape[0])
plt.semilogy(idx[0:svd_split], s[0:svd_split]/s[0], label="most-dominant", color="blue")
plt.semilogy(idx[svd_split: ], s[svd_split: ]/s[0], label="lest-dominant", color="red")
plt.xlabel(r"$i$")
plt.ylabel(r"singular value ($\sigma_i$)")
plt.grid(visible=True)
plt.legend()

plt.subplot(2, 2, 2)
m, z     = np.linalg.eig(Lop)
#print(m, np.real(m), np.imag(m))
idx      = np.argsort(np.real(m))
Im       = np.eye(z.shape[0])
nIm      = np.linalg.norm(Im)
zinv     = cp.asnumpy(cp.linalg.inv(cp.asarray(z)))
print("||I-zz^{-1}|| = %.8E ||I-z^{-1}z||=%.8E"%(np.linalg.norm(Im- z @ zinv)/nIm, np.linalg.norm(Im - zinv @ z)/nIm))

m_sort    = m[idx]
z_sort    = z[:, idx]
zinv_sort = zinv[:, idx]

eigen_split = int(Lop.shape[0] * 0.1)

plt.scatter(np.real(m_sort[0:eigen_split]), np.imag(m_sort[0:eigen_split]), s=0.6, facecolors='none', edgecolors='b', label="fast")
plt.scatter(np.real(m_sort[eigen_split: ]), np.imag(m_sort[eigen_split: ]), s=0.6, facecolors='none', edgecolors='r', label="slow")
plt.xlabel(r"$Re(\lambda_i)$")
plt.ylabel(r"$Im(\lambda_i)$")
plt.legend()
plt.grid(visible=True)

plt.subplot(2, 2, 3)
color = iter(cm.rainbow(np.linspace(0, 1, num_ic)))
for ic_idx in range(num_ic):
    c = next(color)
    plt.semilogy(ts, np.linalg.norm(FH[:, ic_idx, :], axis=1), '-'  ,label="most-dominant  $(IC=%d)"%(ic_idx), color=c)
    plt.semilogy(ts, np.linalg.norm(FL[:, ic_idx, :], axis=1), '--' ,label="least-dominant $(IC=%d)"%(ic_idx), color=c)
plt.xlabel(r"time [T]")
plt.ylabel(r"$L = U \Sigma V^T \text{ and } ||V^T F||$")
plt.legend(fontsize=6, loc='upper right')
plt.grid(visible=True)
plt.title(r"SVD modes")

plt.subplot(2, 2, 4)
print(np.min(np.real(m_sort[0:eigen_split])), np.max(np.real(m_sort[0:eigen_split])))
print(np.min(np.real(m_sort[eigen_split: ])), np.max(np.real(m_sort[eigen_split: ])))
FH       = np.real(F @ zinv_sort[:, 0:eigen_split])
FL       = np.real(F @ zinv_sort[:, eigen_split:])

color = iter(cm.rainbow(np.linspace(0, 1, num_ic)))
for ic_idx in range(num_ic):
    c = next(color)
    plt.semilogy(ts, np.linalg.norm(FH[:, ic_idx, :], axis=1), '-'   ,label="fast (IC=%d)"%(ic_idx), color=c)
    plt.semilogy(ts, np.linalg.norm(FL[:, ic_idx, :], axis=1), '--'  ,label="slow (IC=%d)"%(ic_idx), color=c)

plt.xlabel(r"time [T]")
plt.ylabel(r"$L = U \Lambda U^{-1} \text{ and } ||Re(U^{-1} F)||$")
plt.legend(fontsize=6, loc='upper right')
plt.grid(visible=True)
plt.title(r"Eigen modes")


plt.tight_layout()
plt.savefig("%s/sol_svd_eig.png"%(folder_name))
plt.close()
#d, z     = np.linalg.eig(Lop)









