import numpy as np
import matplotlib.pyplot as plt


def glowfluid():
    ut         = np.load("ut.npy")
    tt         = np.linspace(0, 1, ut.shape[-1])
    ut         = ut.reshape((glow_1d.Np, glow_1d.Nv, ut.shape[-1]))
    
    plt.figure(figsize=(10, 8), dpi=300)
    for i in range(40, 161, 20):
        plt.plot(tt, ut[i, 0, :]     ,      label=r"$n_e$, x=%.4E"%(glow_1d.xp[i]))
        plt.plot(tt, ut[i, 1, :],'--',      label=r"$n_i$, x=%.4E"%(glow_1d.xp[i]))
        plt.grid(visible=True)
        plt.xlabel(r"time")
        plt.legend()
    
    plt.savefig("test1.png")
    plt.close()
    
    plt.figure(figsize=(10, 8), dpi=300)
    for i in range(0, 40, 5):
        plt.semilogy(tt, ut[i, 0, :],      label=r"$n_e$, x=%.4E"%(glow_1d.xp[i]))
        plt.semilogy(tt, ut[i, 1, :],'--', label=r"$n_i$, x=%.4E"%(glow_1d.xp[i]))
        plt.grid(visible=True)
        plt.xlabel(r"time")
        plt.legend()
        
    plt.savefig("test2.png")
    plt.close()
    
    
    plt.figure(figsize=(10, 8), dpi=300)
    for i in range(40, 161, 20):
        plt.plot(tt, ut[i, 2, :] / ut[i, 0, :],      label=r"$T_e$, x=%.4E"%(glow_1d.xp[i]))
        plt.grid(visible=True)
        plt.xlabel(r"time")
        plt.legend()
        
    plt.savefig("test3.png")
    plt.close()
    
    plt.figure(figsize=(10, 8), dpi=300)
    for i in range(0, 40, 5):
        plt.plot(tt, ut[i, 2, :] / ut[i, 0, :],      label=r"$T_e$, x=%.4E"%(glow_1d.xp[i]))
        plt.grid(visible=True)
        plt.xlabel(r"time")
        plt.legend()
        
    plt.savefig("test4.png")
    plt.close()
    
    from matplotlib.colors import LogNorm
    freq    = np.fft.fftfreq(ut.shape[-1], d=1e-3 * glow_1d.param.tau)
    idx     = freq.argsort()
    ele_fft = np.fft.fft(ut[:, 0])[:, idx]          #np.array([np.fft.fft(ut[i,0])[idx] for i in range(glow_1d.Np)]).reshape((glow_1d.Np, -1))
    ion_fft = np.fft.fft(ut[:, 1])[:, idx]          #np.array([np.fft.fft(ut[i,1])[idx] for i in range(glow_1d.Np)]).reshape((glow_1d.Np, -1))
    Te_fft  = np.fft.fft(ut[:, 2]/ut[:, 0])[:, idx] #np.array([np.fft.fft(ut[i,2]/ut[i,0])[idx] for i in range(glow_1d.Np)]).reshape((glow_1d.Np, -1))
    
    plt.figure(figsize=(12,12), dpi=100)
    plt.subplot(3, 1, 1)
    plt.imshow(np.abs(ele_fft), norm=LogNorm(vmin=1e-6, vmax=1e3), cmap='jet', aspect='auto', interpolation='none', extent=[np.min(freq),np.max(freq), -1,1])
    plt.colorbar()
    plt.ylabel(r"x")
    plt.xlabel(r"freq")
    plt.title(r"$n_e$")
    
    plt.subplot(3, 1, 2)
    plt.imshow(np.abs(ion_fft), norm=LogNorm(vmin=1e-6, vmax=1e3), cmap='jet', aspect='auto', interpolation='none', extent=[np.min(freq),np.max(freq), -1,1])
    plt.colorbar()
    plt.ylabel(r"x")
    plt.xlabel(r"freq")
    plt.title(r"$n_i$")
    
    plt.subplot(3, 1, 3)
    plt.imshow(np.abs(Te_fft), norm=LogNorm(vmin=1e-6, vmax=1e3), cmap='jet', aspect='auto', interpolation='none', extent=[np.min(freq),np.max(freq), -1,1])
    plt.colorbar()
    plt.ylabel(r"x")
    plt.xlabel(r"freq")
    plt.title(r"$T_e$")
    plt.tight_layout()
    plt.savefig("test5.png")
    plt.close()
    
    xx=glow_1d.xp
    for i in range(0, 4):
        plt.semilogy(freq[idx], np.abs(ele_fft[i]), label=r"x=%.4E"%(xx[i]))
        
    plt.xlabel(r"freq")
    plt.ylabel(r"mag")
    plt.legend()
    plt.savefig("test.png")
    plt.close()