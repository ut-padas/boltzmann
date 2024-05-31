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


data=[plot_utils.load_data_bte("../1dglow/r1_v2_dt_2e-4", [1], None, read_cycle_avg=False, use_ionization=1),
      plot_utils.load_data_bte("../1dglow/r1_v2_dt_1e-4", [1], None, read_cycle_avg=False, use_ionization=1),
      plot_utils.load_data_bte("../1dglow/r1_v2_dt_5e-5", [1], None, read_cycle_avg=False, use_ionization=1),
      plot_utils.load_data_bte("../1dglow/r1_v1_dt_5e-5", [1], None, read_cycle_avg=False, use_ionization=1)]

labels=[r"dt=2e-4T", r"dt=1e-4T", r"dt=5e-5T", r"dt=5e-5T"]

ele_idx = 0
Te_idx  = 2

plt.figure(figsize=(16, 16), dpi=300)

plt.subplot(2, 2, 1)
for i in range(len(labels)):
    plt.plot(op200.xp, data[i][1][0, :, ele_idx] * op200.np0, label=labels[i])

plt.xlabel(r"2x/L -1")
plt.ylabel(r"$n_e [m^{-3}]$")
plt.grid(visible=True)
plt.legend()




plt.subplot(2, 2, 2)
for i in range(len(labels)):
    plt.plot(op200.xp, data[i][1][0, :, Te_idx], label=labels[i])

plt.xlabel(r"2x/L -1")
plt.ylabel(r"$T_e [eV]$")
plt.grid(visible=True)
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(op200.xp, op200.xp**7 * 8e4)
plt.xlabel(r"2x/L -1")
plt.ylabel(r"$E_0 [V/m]$")
plt.title(r"$E(x, t) = E_0 sin (\omega t)$")
plt.grid(visible=True)

plt.savefig("a.png")

