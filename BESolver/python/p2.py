import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import collisions
#plt.rcParams['text.usetex'] = True
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "font.size": 12,
    "lines.linewidth":1.2
})


def read_eedfs(fname, n0=collisions.AR_NEUTRAL_N):
    runs_list   = list()
    with open(fname, 'rb') as f:
        try:
            while 1:
                num_p   = int(np.load(f))
                lmax    = int(np.load(f))
                sph_lm  = [(i,0) for i in range(lmax + 1)]
                e_field = float(np.load(f))
                EbN     = e_field / n0 / 1e-21
                ion_deg = float(np.load(f))
                Tg      = float(np.load(f))
                ev      = np.load(f)
                

                fl_eedf = list()

                for lm_idx, lm in enumerate(sph_lm):
                    fl_eedf.append(np.load(f))

                bolsig_f0 = np.load(f)
                bolsig_f1 = np.load(f)

                runs_list.append({"num_p": num_p, "lm": sph_lm, "E" : e_field, "E/N(Td)": EbN, "ion_deg": ion_deg, "Tg": Tg , "ev": ev, "fl": fl_eedf, "bolsig":[bolsig_f0, bolsig_f1]})

        except:
            pass
            #print("EoF")

    runs_list = pd.DataFrame(runs_list)
    return runs_list

fname1 = "eedf_02_28_2023_cc_nr_128_lmax1.npy"
fname2 = "eedf_03_01_2023_cc_nr_128_lmax6.npy"

df1    = pd.read_csv("pde_vs_bolsig_02_28_2023_cc_nr128_lmax1.csv")
df2    = pd.read_csv("pde_vs_bolsig_03_01_2023_cc_nr128_lmax6.csv")

print(df1)
print(df2)

run1   = read_eedfs(fname1)
run2   = read_eedfs(fname2)

EbN    = list(set(run1["E/N(Td)"].values.tolist()))
EbN.sort()

ion_deg_values = list(set(run1["ion_deg"].values.tolist()))
ion_deg_values.sort(reverse=True)

print(run1)

print(EbN, ion_deg_values)


lmax1  = run1.loc[:,"lm"][0][-1][0]
lmax2  = run2.loc[:,"lm"][0][-1][0]

lmax         = max(lmax1, lmax2)
assert lmax == lmax2, "fname2 must be the higher l mode run"

total_plots = lmax +1
rows        = 1
cols        = total_plots

sph_lm = [(i, 0) for i in range(lmax+1)]
print(lmax, total_plots)

for ebn_idx, ebn in enumerate(EbN):
    fig = plt.figure(figsize=(22, 6), dpi=300)
    
    for ion_deg_idx, ion_deg in enumerate(ion_deg_values):

        if ion_deg <=1e-4:
            continue

        row_id = ebn_idx * len(ion_deg_values) + ion_deg_idx
        r1     = run1.iloc[row_id]
        r2     = run2.iloc[row_id]

        ev1    = r1["ev"]
        ev2    = r2["ev"]

        bf     = r1["bolsig"]
        

        fl1    = r1["fl"]
        fl2    = r2["fl"]

        for lm_idx, lm in enumerate(sph_lm):
            plt.subplot(2, total_plots, lm_idx + 1)
            color = next(plt.gca()._get_lines.prop_cycler)['color']
            plt.semilogy(ev2, np.abs(fl2[lm_idx]), label="ne/n0=%.1E"%(ion_deg), color=color)
            plt.title("f_%d"%(lm[0]))
            plt.xlabel("energy (ev)")
            plt.grid(visible=True)

            # if lm_idx == 0:
            #     plt.semilogy(ev2, np.abs(bf0), label="ne/n0=%.1E"%(ion_deg), color=color)

            if lm_idx==0:
                plt.legend(fontsize="xx-small")


        for lm_idx, lm in enumerate([(0,0), (1,0)]):
            plt.subplot(2, total_plots, total_plots + lm_idx + 1)
            color = next(plt.gca()._get_lines.prop_cycler)['color']
            plt.semilogy(ev2, np.abs(fl1[lm_idx]), label="ne/n0=%.1E"%(ion_deg), color=color)
            plt.semilogy(ev2, np.abs(bf[lm_idx]) , '--', label="ne/n0=%.1E"%(ion_deg), color=color)
            plt.title("two term f_%d"%(lm[0]))
            plt.xlabel("energy (ev)")
            plt.grid(visible=True)

        for lm_idx, lm in enumerate([(0,0), (1,0)]):
            plt.subplot(2, total_plots, 2 + total_plots + lm_idx + 1)
            color = next(plt.gca()._get_lines.prop_cycler)['color']
            plt.semilogy(ev2, np.abs(fl1[lm_idx] - fl2[lm_idx]), label="ne/n0=%.1E"%(ion_deg), color=color)
            plt.title("lmax=1 vs. lmax=6 f_%d"%(lm[0]))
            plt.ylabel("absolute difference")
            plt.xlabel("energy (ev)")
            plt.grid(visible=True)

        
        plt.subplot(2, total_plots, 4 + total_plots + 0 + 1)
        color = next(plt.gca()._get_lines.prop_cycler)['color']
        plt.semilogy([129], np.abs(df1.iloc[row_id]["g0"]/df2.iloc[row_id]["g0"] - 1),"*", label="elastic ne/n0=%.1E"%(ion_deg), color=color)
        plt.title("lmax=1 vs. lmax=6 elastic coefficient")
        plt.ylabel("relative error")
        plt.xlabel("nr")
        plt.grid(visible=True)
        #plt.legend()
        
        plt.subplot(2, total_plots, 4 + total_plots + 1 + 1)
        color = next(plt.gca()._get_lines.prop_cycler)['color']
        if df2.iloc[row_id]["g2"]>0:
            plt.semilogy([129], np.abs(df1.iloc[row_id]["g2"]/df2.iloc[row_id]["g2"] - 1),"*", label="ne/n0=%.1E"%(ion_deg), color=color)
        plt.title("lmax=1 vs. lmax=6 ionization coefficient")
        plt.ylabel("relative error")
        plt.xlabel("nr")
        plt.grid(visible=True)

        plt.subplot(2, total_plots, 4 + total_plots + 2 + 1)
        color = next(plt.gca()._get_lines.prop_cycler)['color']
        plt.semilogy([129], np.abs(df1.iloc[row_id]["mobility"]/df2.iloc[row_id]["mobility"] - 1),"*", label="ne/n0=%.1E"%(ion_deg), color=color)
        plt.title("lmax=1 vs. lmax=6 mobility")
        plt.ylabel("relative error")
        plt.xlabel("nr")
        plt.grid(visible=True)

        #plt.legend()

    plt.tight_layout()
    fig.suptitle("E/N(Td)=%.2E gas temperature=%.2E, "%(ebn, r2["Tg"]))
    #plt.show()
    #plt.close()
    fig.savefig("EbN_Td_%2E.png"%(ebn))
    plt.close()















        