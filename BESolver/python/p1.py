import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#plt.rcParams['text.usetex'] = True
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "font.size": 16,
    "lines.linewidth":1.2,
    "lines.markersize" : 4
})



def plot_pde_vs_bolsig():

    # df          = pd.read_csv (r'pde_vs_bolsig_ee_collisions_lmax1/pde_vs_bolsig_02_04_2023_coulomb_nr64_lmax1.csv')
    # fout_name   = 'pde_vs_bolsig_ee_collisions_lmax1/pde_vs_bolsig_with_coulomb'

    df1         = pd.read_csv(r'pde_vs_bolsig_approx_collop/pde_vs_bolsig_09_14_2022_collop_approx.csv')

    df          = pd.read_csv (r'pde_vs_bolsig_ee_collisions_lmax1_03_31_2023/pde_vs_bolsig_03_31_2023_11:44:46.csv')
    fout_name   = 'pde_vs_bolsig_ee_collisions_lmax1_03_31_2023/pde_vs_bolsig_with_coulomb_collision'

    rows = 2
    cols = 2

    rr_data1   = df1[df1['Nr']==128.0]
    fig        = plt.figure(figsize=(21, 9), dpi=300)
    color_list = ["black", "red","blue", "orange"]
    for idx, ion_deg in enumerate([0,1e-3,1e-2,1e-1]):
        if ion_deg==0:
            rr_data=rr_data1
        else:
            rr_data = df[df['Nr']==64.0][df['ion_deg']==ion_deg]
        e_field = np.array(rr_data['E/N(Td)'])
        color   = color_list[idx]
        
        plt.subplot(rows, cols,   1)
        #plt.plot(e_field, np.array(rr_data['bolsig_g0']), '.', color=color, label="ne/N=%.1E bolsig"%(ion_deg))
        plt.plot(e_field, np.array(rr_data['g0']), 'x'       , color=color, label="ne/N=%.1E"%(ion_deg))
        plt.xlabel(r"E/N (Td)")
        plt.ylabel(r"reaction rate ($m^3s^{-1}$)")
        plt.xscale('log')
        plt.yscale('log')
        plt.title(r"Elastic ($e + Ar \rightarrow e + Ar$)")
        plt.legend()
        plt.grid(visible=True)

        plt.subplot(rows, cols,   2)
        #plt.plot(e_field, np.array(rr_data['bolsig_g2']), '.', color=color, label="ne/N=%.1E bolsig"%(ion_deg))
        plt.plot(e_field, np.array(rr_data['g2']), 'x'       , color=color, label="ne/N=%.1E"%(ion_deg))
        plt.xlabel(r"E/N (Td)")
        plt.ylabel(r"reaction rate ($m^3s^{-1}$)")
        plt.xscale('log')
        plt.yscale('log')
        plt.title(r"Ionization ($e + Ar \rightarrow 2e + Ar^{+}$)")
        plt.grid(visible=True)

        plt.subplot(rows, cols,   3)
        #plt.plot(e_field, np.array(rr_data['bolsig_energy']), '.', color=color, label="ne/N=%.1E bolsig"%(ion_deg))
        plt.plot(e_field, np.array(rr_data['energy']), 'x'       , color=color, label="ne/N=%.1E"%(ion_deg))
        plt.xlabel(r"E/N (Td)")
        plt.ylabel(r"energy (eV)")
        plt.xscale('log')
        plt.yscale('log')
        plt.title(r"steady-state energy")
        plt.grid(visible=True)

        plt.subplot(rows, cols,   4)
        #plt.plot(e_field, np.array(rr_data['bolsig_mobility']), '.', color=color , label="ne/N=%.1E bolsig"%(ion_deg))
        plt.plot(e_field, np.array(rr_data['mobility']), 'x'       , color=color , label="ne/N=%.1E"%(ion_deg))
        plt.xlabel(r"E/N (Td)")
        plt.ylabel(r"mobility (N (1/m/V/s))")
        plt.xscale('log')
        plt.yscale('log')
        plt.title(r"mobility")
        plt.grid(visible=True)
        
    
    


    #fig.suptitle("Gas temperature =%.2EK, with Nr=%d"%(rr_data.iloc[0]['Tg'], rr_data.iloc[0]['Nr'])) # or plt.suptitle('Main title')
    plt.tight_layout()
    plt.savefig("%s.svg"%(fout_name))
    plt.close()

    rows = 3 
    cols = 2
    fig = plt.figure(figsize=(21, 15), dpi=300)
    color_list = ["red","blue", "orange"]
    for idx, ion_deg in enumerate([1e-1,1e-2,1e-3]):
        rr_data = df[df['Nr']==64.0][df['ion_deg']==ion_deg]
        e_field = np.array(rr_data['E/N(Td)'])
        color   = color_list[idx]

        plt.subplot(rows, cols,   1)
        w0        = np.array(rr_data['bolsig_g0'])
        w1        = np.array(rr_data['g0'])
        rel_error = np.abs(1-w1/w0)
        plt.plot(e_field, rel_error, 'x--', color=color ,  label="ne/n0=%.1E"%(ion_deg))
        plt.xlabel(r"E/N (Td)")
        plt.ylabel(r"relative error ")
        plt.xscale('log')
        plt.yscale('log')
        plt.title(r"Elastic ($e + Ar \rightarrow e + Ar$)")
        plt.legend()
        plt.grid()

        plt.subplot(rows, cols,   2)
        w0        = np.array(rr_data['bolsig_g2'])
        w1        = np.array(rr_data['g2'])
        rel_error = np.abs(1-w1/w0)
        plt.plot(e_field, rel_error, 'x--', color=color, label="ne/n0=%.1E"%(ion_deg))
        plt.xlabel(r"E/N (Td)")
        plt.ylabel(r"relative error ")
        plt.xscale('log')
        plt.yscale('log')
        #plt.legend()
        plt.title(r"Ionization ($e + Ar \rightarrow 2e + Ar^{+}$)")
        plt.grid()

        plt.subplot(rows, cols,   3)
        w0        = np.array(rr_data['bolsig_energy'])
        w1        = np.array(rr_data['energy'])
        rel_error = np.abs(1-w1/w0)
        plt.plot(e_field, rel_error, 'x--', color=color , label="ne/n0=%.1E"%(ion_deg))
        plt.xlabel(r"E/N (Td)")
        plt.ylabel(r"relative error ")
        plt.xscale('log')
        plt.yscale('log')
        #plt.legend()
        plt.title(r"steady-state energy")
        plt.grid()

        plt.subplot(rows, cols,   4)
        w0        = np.array(rr_data['bolsig_mobility'])
        w1        = np.array(rr_data['mobility'])
        rel_error = np.abs(1-w1/w0)
        plt.plot(e_field, rel_error, 'x--', color=color , label="ne/n0=%.1E"%(ion_deg))
        plt.xlabel(r"E/N (Td)")
        plt.ylabel(r"relative error ")
        plt.xscale('log')
        plt.yscale('log')
        #plt.legend()
        plt.title(r"mobility")
        plt.grid()

        plt.subplot(rows, cols,   5)
        rel_error = np.array(rr_data['l2_f0'])
        plt.plot(e_field, rel_error, 'x--', color=color , label="ne/n0=%.1E"%(ion_deg))
        plt.xlabel(r"E/N (Td)")
        plt.ylabel(r"relative error ")
        plt.xscale('log')
        plt.yscale('log')
        #plt.legend()
        plt.title(r"f0")
        plt.grid()

        plt.subplot(rows, cols,   6)
        rel_error = np.array(rr_data['l2_f1'])
        plt.plot(e_field, rel_error, 'x--', color=color , label="ne/n0=%.1E"%(ion_deg))
        plt.xlabel(r"E/N (Td)")
        plt.ylabel(r"relative error ")
        plt.xscale('log')
        plt.yscale('log')
        #plt.legend()
        plt.title(r"f1")
        plt.grid()

    plt.suptitle("Gas temperature =%.2EK with Nr=%d"%(rr_data.iloc[0]['Tg'], rr_data.iloc[0]['Nr'])) # or plt.suptitle('Main title')
    plt.tight_layout()
    plt.savefig("%s_error.svg"%(fout_name))
    plt.close()

def plot_lmax1_vs_lmax3():
    dir_1 = "pde_vs_bolsig_ee_collisions_lmax1"
    dir_2 = "pde_vs_bolsig_ee_collisions_lmax3"

    f1    = open('%s/eedf_02_04_2023_coulomb_nr64_lmax1.npy'%(dir_1), 'rb')
    f2    = open('%s/eedf_02_05_2023_coulomb_nr64_lmax3.npy'%(dir_2), 'rb')


    EE     = np.load(f1)
    eedf_1 = list()
    for i in range(len(EE)):
        for ion_deg in [1e-1,1e-2,1e-3]:
            eedf_1.append({'E': EE[i] , 'ev': np.load(f1), 'f0': np.load(f1), 'f1': np.load(f1), 'bolsig_f0' : np.load(f1),'bolsig_f1': np.load(f1),'ion_deg':ion_deg}) 

    EE     = np.load(f2)
    eedf_2 = list()
    for i in range(len(EE)):
        for ion_deg in [1e-1,1e-2,1e-3]:
            eedf_2.append({'E': EE[i] , 'ev': np.load(f2), 'f0': np.load(f2), 'f1': np.load(f2), 'bolsig_f0' : np.load(f2),'bolsig_f1': np.load(f2),'ion_deg':ion_deg}) 
    
    import os
    path    = "%s_vs_%s"%(dir_1,dir_2)
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    
    for ee_idx, ee in enumerate(EE):
        for ion_deg_idx, ion_deg in enumerate([1e-1,1e-2,1e-3]):
            print("E=", ee, "ion_deg", ion_deg, "ev ", np.min(eedf_1[3 * ee_idx + ion_deg_idx]['ev']) , np.max(eedf_1[3 * ee_idx + ion_deg_idx]['ev']))
            fig = plt.figure(figsize=(12, 12), dpi=100)
            plt.subplot(2, 2, 1)
            plt.semilogy(eedf_1[3 * ee_idx + ion_deg_idx]['ev'], np.abs(eedf_1[3 * ee_idx + ion_deg_idx]['bolsig_f0']), label='bolsig+')
            plt.semilogy(eedf_1[3 * ee_idx + ion_deg_idx]['ev'], np.abs(eedf_1[3 * ee_idx + ion_deg_idx]['f0']), label='lmax=1')
            plt.semilogy(eedf_2[3 * ee_idx + ion_deg_idx]['ev'], np.abs(eedf_2[3 * ee_idx + ion_deg_idx]['f0']), label='lmax=3')
            plt.xlabel("energy (ev)")
            plt.legend()
            plt.grid()

            plt.subplot(2, 2, 2)
            plt.semilogy(eedf_1[3 * ee_idx + ion_deg_idx]['ev'], np.abs(eedf_1[3 * ee_idx + ion_deg_idx]['bolsig_f1']), label='bolsig+')
            plt.semilogy(eedf_1[3 * ee_idx + ion_deg_idx]['ev'], np.abs(eedf_1[3 * ee_idx + ion_deg_idx]['f1']), label='lmax=1')
            plt.semilogy(eedf_2[3 * ee_idx + ion_deg_idx]['ev'], np.abs(eedf_2[3 * ee_idx + ion_deg_idx]['f1']), label='lmax=3')
            plt.xlabel("energy (ev)")
            plt.legend()
            plt.grid()

            plt.subplot(2, 2, 3)
            plt.semilogy(eedf_1[3 * ee_idx + ion_deg_idx]['ev'], np.abs(eedf_1[3 * ee_idx + ion_deg_idx]['bolsig_f0'] - eedf_1[3 * ee_idx + ion_deg_idx]['f0'])/ np.abs(eedf_1[3 * ee_idx + ion_deg_idx]['bolsig_f0']), label='bolsig+ vs. lmax=1')
            plt.semilogy(eedf_2[3 * ee_idx + ion_deg_idx]['ev'], np.abs(eedf_1[3 * ee_idx + ion_deg_idx]['bolsig_f0'] - eedf_2[3 * ee_idx + ion_deg_idx]['f0'])/ np.abs(eedf_1[3 * ee_idx + ion_deg_idx]['bolsig_f0']), label='bolsig+ vs. lmax=3')
            plt.xlabel("energy (ev)")
            plt.ylabel("relative error")
            plt.legend()
            plt.grid()

            plt.subplot(2, 2, 4)
            plt.semilogy(eedf_1[3 * ee_idx + ion_deg_idx]['ev'], np.abs(eedf_1[3 * ee_idx + ion_deg_idx]['bolsig_f1'] - eedf_1[3 * ee_idx + ion_deg_idx]['f1'])/np.abs(eedf_1[3 * ee_idx + ion_deg_idx]['bolsig_f1']), label='bolsig+ vs. lmax=1')
            plt.semilogy(eedf_2[3 * ee_idx + ion_deg_idx]['ev'], np.abs(eedf_1[3 * ee_idx + ion_deg_idx]['bolsig_f1'] - eedf_2[3 * ee_idx + ion_deg_idx]['f1'])/np.abs(eedf_1[3 * ee_idx + ion_deg_idx]['bolsig_f1']), label='bolsig+ vs. lmax=3')
            plt.ylabel("relative error")
            plt.legend()
            plt.grid()

            plt.tight_layout()
            fig.suptitle("E = %.3E ionization degree = %.3E with nr=64"%(ee, ion_deg))
            fig.savefig("%s_vs_%s/p_%04d.png"%(dir_1,dir_2, 3 * ee_idx + ion_deg_idx))
            fig.clear()
            plt.close()
            
#plot_lmax1_vs_lmax3()
plot_pde_vs_bolsig()


    

