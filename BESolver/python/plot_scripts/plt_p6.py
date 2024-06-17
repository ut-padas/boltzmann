import numpy as np
import matplotlib.pyplot as plt
import plot_utils 


op200 = plot_utils.op(200)

# data=[plot_utils.load_data_bte("../1dglow/e8e4_dt_1e-3", [1], None, read_cycle_avg=False, use_ionization=1),
#       plot_utils.load_data_bte("../1dglow/e8e4_dt_5e-4", [1], None, read_cycle_avg=False, use_ionization=1),
#       plot_utils.load_data_bte("../1dglow/e8e4_dt_2e-4", [1], None, read_cycle_avg=False, use_ionization=1),
#       plot_utils.load_data_bte("../1dglow/e8e4_dt_1e-4", [1], None, read_cycle_avg=False, use_ionization=1),
#       plot_utils.load_data_bte("../1dglow/e8e4_dt_5e-5", [1], None, read_cycle_avg=False, use_ionization=1)]

#labels=[r"dt=1e-3T", r"dt=5e-4T",r"dt=2e-4T", r"dt=1e-4T", r"dt=5e-5T"]

for idx in range(0, 6, 1):
    # data=[
    #     plot_utils.load_data_bte("../1dglow/r1_v1_dt_5e-4"     , [idx], None, read_cycle_avg=False, use_ionization=1),
    #     plot_utils.load_data_bte("../1dglow/r1_v1_dt_2.5e-4"   , [idx], None, read_cycle_avg=False, use_ionization=1),
    #     plot_utils.load_data_bte("../1dglow/r1_v1_dt_1.25e-4"  , [idx], None, read_cycle_avg=False, use_ionization=1),
    #     plot_utils.load_data_bte("../1dglow/r1_v1_dt_5e-5"     , [idx], None, read_cycle_avg=False, use_ionization=1)
        
    #     ]

    # labels=[r"dt=5.0e-4T (v1)", r"dt=2.5e-4T (v1)", r"dt=1.25e-4T (v1)", r"dt=5e-5T (v1)"]
    # fprefix="cycle_v1"
    
    # data=[
    #     plot_utils.load_data_bte("../1dglow/r1_v2_dt_2.5e-4"   , [idx], None, read_cycle_avg=False, use_ionization=1),
    #     plot_utils.load_data_bte("../1dglow/r1_v2_dt_1.25e-4"  , [idx], None, read_cycle_avg=False, use_ionization=1),
    #     plot_utils.load_data_bte("../1dglow/r1_v1_dt_5e-5"     , [idx], None, read_cycle_avg=False, use_ionization=1)
    #     ]

    # labels=[r"dt=2.5e-4T (v2)", r"dt=1.25e-4T (v2)", r"dt=5e-5T (v1)"]
    # fprefix="cycle_v2"
    
    # data=[
    #     plot_utils.load_data_bte("../1dglow/r1_v2b_dt_2.5e-4"   , [idx], None, read_cycle_avg=False, use_ionization=1),
    #     plot_utils.load_data_bte("../1dglow/r1_v2b_dt_1.25e-4"  , [idx], None, read_cycle_avg=False, use_ionization=1),
    #     plot_utils.load_data_bte("../1dglow/r1_v1_dt_5e-5"     , [idx], None, read_cycle_avg=False, use_ionization=1)
    #     ]

    # labels=[r"dt=2.5e-4T (v2)", r"dt=1.25e-4T (v2)", r"dt=5e-5T (v1)"]
    
    # data=[ #plot_utils.load_data_bte("../1dglow/s1_dt_4e-4"   , [idx], None, read_cycle_avg=False, use_ionization=1),
    #        plot_utils.load_data_bte("../1dglow/s1_dt_2e-4"   , [idx], None, read_cycle_avg=False, use_ionization=1),
    #        plot_utils.load_data_bte("../1dglow/s0_dt_2e-4"   , [idx], None, read_cycle_avg=False, use_ionization=1),
    #        plot_utils.load_data_bte("../1dglow/s0_dt_1e-4"   , [idx], None, read_cycle_avg=False, use_ionization=1),
    #        plot_utils.load_data_bte("../1dglow/s0_dt_5e-5"   , [idx], None, read_cycle_avg=False, use_ionization=1)
    #        ]
    
    data=[ plot_utils.load_data_bte("../1dglow/c1_s1_dt_2e-4"   , [idx], None, read_cycle_avg=False, use_ionization=1),
           plot_utils.load_data_bte("../1dglow/c1_s1_dt_1e-4"   , [idx], None, read_cycle_avg=False, use_ionization=1)]
    
    labels=[r"dt=2e-4T (scheme-1)", r"dt=1e-4T (scheme-1)"]
    fprefix="tmp"
    
    ele_idx = 0
    ion_idx = 1
    Te_idx  = 2

    plt.figure(figsize=(16, 16), dpi=300)

    plt.subplot(2, 2, 1)
    for i in range(len(labels)):
        plt.plot(op200.xp, data[i][1][0, :, ele_idx] * op200.np0, label=labels[i])
        if (i==0):
            print(data[i][1][0, :, ele_idx])
    plt.xlabel(r"2x/L -1")
    plt.ylabel(r"$n_e [m^{-3}]$")
    plt.grid(visible=True)
    plt.legend()




    plt.subplot(2, 2, 2)
    for i in range(len(labels)):
        plt.semilogy(op200.xp, data[i][1][0, :, Te_idx], label=labels[i])

    plt.xlabel(r"2x/L -1")
    plt.ylabel(r"$T_e [eV]$")
    plt.grid(visible=True)
    plt.legend()

    plt.subplot(2, 2, 3)
    for i in range(len(labels)):
        ne  = data[i][1][0, :, ele_idx]
        ni  = data[i][1][0, :, ion_idx]
        
        phi = op200.solve_poisson(ne, ni, 0.0)
        E   = np.dot(op200.Dp, -phi) * (op200.V0/op200.L)
    
        plt.plot(op200.xp, E, label=labels[i])
    plt.xlabel(r"2x/L -1")
    plt.ylabel(r"$E [V/m]$")
    plt.grid(visible=True)
    plt.legend()
    
    plt.subplot(2, 2, 4)
    for i in range(len(labels)):
        ne  = data[i][1][0, :, ele_idx]
        ni  = data[i][1][0, :, ion_idx]
        
        phi = op200.solve_poisson(ne, ni, 0.0) * op200.V0
        
        plt.plot(op200.xp, phi, label=labels[i])
    plt.xlabel(r"2x/L -1")
    plt.ylabel(r"$\phi [V]$")
    plt.grid(visible=True)
    plt.legend()
    
    plt.suptitle(r"t=%.4E T"%(idx * 1e-3))
    plt.savefig("%s_%04d.png"%(fprefix, idx))
    plt.close()

