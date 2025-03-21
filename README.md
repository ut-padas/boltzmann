# Boltzsim : Eulerian electron Boltzmann solver for low-temperature plasmas (LTPs)

This repository contains Python based Eulerian electron Boltzmann solver based on Galerkin discretization. We expand the electron distribution function (EDF) using spherical coordinates, with B-splines in radial, and spherical harmonics in angular directions. The framework supports the following collision mechanisms. 

* Electron-heavy binary collisions : These are collisions between electron and background heavy particles. We use LXCAT cross-section data base for collisional cross-sections. 
* Electron-electron Coulomb interactions: Coulomb interaction between electrons, modeled using Fokker-Plank collision operator. 


## How to run the solver

pip install sympy numpy scipy cupy-cuda12x multiprocess lxcat_data_parser matplotlib h5py toml

The main program that runs the comparison between Bolsig+ and the developed solver is, https://github.com/ut-padas/boltzmann/blob/main/BESolver/python/collision_driver_spherical_eulerian_dg.py and one can find the arguments for the file, https://github.com/ut-padas/boltzmann/blob/main/BESolver/python/collision_driver_spherical_eulerian_dg.py#L39

## Contact us

If you have any questions, or interested in collaboration please send an email to milinda at oden dot utexas dot edu
