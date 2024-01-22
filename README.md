
# TPS - BTE integration
## 0D Batched BTE solver
* Single GPU
  * Solve for series of batched 0d Boltzmann on a single GPU (steady & transient cases) [Done]
* Extend Single GPU version to Multi-GPU and Parla Integration [Done]
* Finally we can move to spatial coupling if we decided to do so 
  
## TPS code
* Docker documentation: https://github.com/pecos/tps/blob/boltzmann-integration/docker/test-gpu/README.md

## SC24
* Abstract : MAR 26, 2024, paper due: APR 2, 2024
* 


# 1D-BTE
* Perform standalone comparison with PIC DSMC code, with enforced E-field, 
  * Time harmonic spatially homogenous E-field
  * Time harmonic E-field

# 1D-Glow discharge with BTE
## Fluid code
* For fluid approximation need to fix the electron density $n_e$ BC
* Ability to handle tabulated electron kinetic coefficients
* https://github.com/pecos/lxcat-review/blob/master/bolsig/Nominal_3Species.py

## BTE code
* Initial comparison with PIC DSMC with PDE code
* **Implement hybrid, fluid & BTE split scheme to reach the steady state faster**
* Further improvements for time integration
  * Fully-implicit scheme to eliminate operator split ? Matrix free RHS and Jacobian action

  


