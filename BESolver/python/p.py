import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#plt.rcParams['text.usetex'] = True
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "lines.linewidth":1.2
})

# df     = pd.read_csv (r'pde_vs_bolsig_08_25_2022.csv')
# fout_name = 'pde_vs_bolsig_full_collop.png'

df        = pd.read_csv (r'pde_vs_bolsig_09_14_2022_collop_approx.csv')
fout_name = 'pde_vs_bolsig_collop_approx_eedf.png'
#df_pic    = pd.read_csv (r'PIC_runs.csv')


r_32  = df[df['Nr']==32]
r_64  = df[df['Nr']==64]
r_128 = df[df['Nr']==128]

rr    = [r_32, r_64, r_128]
nr    = ['32', '64', '128']


# for r in rr: 
#     r.loc[:,'error_g0']=abs(r['g0']/r['bolsig_g0']-1)
#     r.loc[:,'error_g2']=abs(r['g2']/r['bolsig_g2']-1)


rows = 2
cols = 3
e_field = np.array(r_128['E/N(Td)'])
#e_field_pic = np.array(df_pic['E/N(Td)'])
r=r_128
fig = plt.figure(figsize=(21, 9), dpi=300)
plt.subplot(rows, cols,   1)
#plt.plot(e_field_pic, np.array(df_pic['PIC G0']), 'g*', label="PIC g0")
plt.plot(e_field, np.array(r['bolsig_g0']), 'b*', label="bolsig")
plt.plot(e_field, np.array(r['g0']), 'rx', label="PDE")
plt.xlabel(r"E/N (Td)")
plt.ylabel(r"reaction rate ($m^3s^{-1}$)")
plt.xscale('log')
plt.yscale('log')
plt.title(r"Elastic ($e + Ar \rightarrow e + Ar$)")
plt.legend()
plt.grid()

plt.subplot(rows, cols,   2)
#plt.plot(e_field_pic, np.array(df_pic['PIC G2']), 'g*', label="PIC g2")
plt.plot(e_field, np.array(r['bolsig_g2']), 'b*', label="bolsig")
plt.plot(e_field, np.array(r['g2']), 'rx', label="PDE")
plt.xlabel(r"E/N (Td)")
plt.ylabel(r"reaction rate ($m^3s^{-1}$)")
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.title(r"Ionization ($e + Ar \rightarrow 2e + Ar^{+}$)")
plt.grid()


plt.subplot(rows, cols,   3)
#plt.plot(e_field_pic, np.array(df_pic['PIC ENERGY']), 'g*', label="PIC")
plt.plot(e_field, np.array(r['bolsig_energy']), 'b*', label="bolsig")
plt.plot(e_field, np.array(r['energy']), 'rx', label="PDE")
plt.xlabel(r"E/N (Td)")
plt.ylabel(r"energy (eV)")
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.title(r"steady-state energy")
plt.grid()


plt.subplot(rows, cols,   4)
plt.plot(e_field, np.abs(np.array(r['g0'])/np.array(r['bolsig_g0']) -1), 'rx-')
#plt.plot(e_field_pic, np.abs(np.array(df_pic['PIC G0'])/np.array(df_pic['BOLSIG G0']) -1), 'b--', label="bolsig vs. PIC(g0)")
plt.xlabel(r"E/N (Td)")
plt.ylabel(r"relative error")
plt.xscale('log')
plt.yscale('log')
plt.title(r"Elastic ($e + Ar \rightarrow e + Ar$)")
plt.title("(d)")
plt.grid()

plt.subplot(rows, cols,   5)
plt.plot(e_field, np.abs(np.array(r['g2'])/np.array(r['bolsig_g2']) -1), 'rx-')
#plt.plot(e_field_pic, np.abs(np.array(df_pic['PIC G2'])/np.array(df_pic['BOLSIG G2']) -1), 'b--', label="bolsig vs. PIC(g2)")
plt.xlabel("E/N (Td)")
plt.ylabel("relative error")
plt.xscale('log')
plt.yscale('log')
plt.title(r"Ionization ($e + Ar \rightarrow 2e + Ar^{+}$)")
plt.grid()


plt.subplot(rows, cols,   6)
plt.plot(e_field, np.abs(np.array(r['energy'])/np.array(r['bolsig_energy'])-1), 'rx-')
#plt.plot(e_field_pic, np.abs(np.array(df_pic['PIC ENERGY'])/np.array(df_pic['BOLSIG ENERGY']) -1), 'b--', label="bolsig vs. PIC(energy)")
plt.xlabel(r"E/N (Td)")
plt.ylabel(r"relative error")
plt.xscale('log')
plt.yscale('log')
plt.title(r"steady-state energy")
plt.grid()


# plt.subplot(rows, cols,   7)
# plt.plot(e_field, np.array(r['l2_f0']), 'r-')
# plt.xlabel(r"E/N (Td)")
# plt.ylabel(r"relative error (l2)")
# plt.xscale('log')
# plt.yscale('log')
# plt.title(r"electron energy density function (EEDF)")
# plt.grid()

plt.tight_layout()
plt.savefig(fout_name)

    



def load_eedfs(fname):
    f    = open(fname, 'rb')
    EE   = np.load(f)
    eedf = list()
    
    for i in range(len(EE)):
        eedf.append({'E': EE[i] , 'ev': np.load(f), 'f0': np.load(f), 'f1': np.load(f), 'bolsig_f0' : np.load(f),'bolsig_f1': np.load(f)}) 

    f.close()
    return eedf


