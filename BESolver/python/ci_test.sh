#!/bin/bash

"""
-------------------------------------------------------------------------------------------------------------------------------------------------------------------
python3 collision_driver_spherical_eulerian_dg.py -l_max 1 -c lxcat_data/eAr_crs.6sp_Tg_0.5eV -sp_order 3 -spline_qpts 5 -steady 1 --sweep_values 63 127  -EbyN 2 1 1 -Tg 6000 -ion_deg 1e-2 -ns_by_n0 0.999 0.00033333333333333365 0.00033333333333333365 0.00033333333333333365

bolsig temp      = 8.47333333E-01
bolsig mobility  = 6.55200000E+24
bolsig diffusion = 5.55000000E+24
bolsig collision rates
C0 = 1.60500000E-14
C1 = 1.51500000E-21
C2 = 2.36100000E-21
C3 = 5.31700000E-23
C4 = 7.34800000E-23
C5 = 6.49800000E-16
C6 = 7.37000000E-16
C7 = 1.97000000E-14

target ev range : (0.0000E+00, 2.4888E+01) ----> knots domain : (0.0000E+00, 5.4196E+00)
singularity pts :  [1.64072387 2.15416819 2.18274111 3.69175459 3.70376135 3.90288279
 4.31266364] v/vth and [ 2.281    3.932    4.037   11.54835 11.62359 12.907   15.75961] eV


python3 bte_0d3v_batched_driver.py -threads 16 -solver_type steady-state -l_max 1 -sp_order 3 -spline_qpts 15 -Efreq 0.0 -dt 1e-3 -plot_data 1 -Nr 127 --ee_collisions 1 -use_gpu 1 -cycles 3 -out_fname batched_bte/ss_test_0 -verbose 1 -max_iter 10000 -atol 1e-10 -rtol 1e-10 -c lxcat_data/eAr_crs.6sp_Tg_0.5eV -ev_max 17.283333333333335 -n_pts 3 -Te 8.47333333E-01

n0,ne,ni,Tg,E,energy,mobility,diffusion,C0,C1,C2,C3,C4,C5,C6,C7
3.22e+22,3.22e+20,3.22e+20,6000.0,64.39999999999999,1.2727737506625219,6.601071615096679e+24,1.6868185007907134e+25,1.608588383780616e-14,1.54764824328749e-21,2.391123112684273e-21,5.387337725480827e-23,7.263669643398811e-23,6.557877082206871e-16,7.435222987797958e-16,1.983202052822877e-14

C0 = 1.608588383780616e-14
C1 = 1.54764824328749e-21
C2 = 2.391123112684273e-21
C3 = 5.387337725480827e-23
C4 = 7.263669643398811e-23
C5 = 6.557877082206871e-16
C6 = 7.435222987797958e-16
C7 = 1.983202052822877e-14

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

python3 collision_driver_spherical_eulerian_dg.py -l_max 1 -c lxcat_data/eAr_crs.6sp_Tg_0.5eV -sp_order 3 -spline_qpts 15 -steady 1 --sweep_values 63 127  -EbyN 2 1 1 -Tg 6000 -ion_deg 0 -ns_by_n0 0.999 0.00033333333333333365 0.00033333333333333365 0.00033333333333333365

bolsig temp      = 1.09133333E+00     
bolsig mobility  = 5.91800000E+24    
bolsig diffusion = 1.29800000E+25
bolsig collision rates
C0 = 2.11300000E-14
C1 = 2.42300000E-31
C2 = 1.76200000E-31
C3 = 6.55800000E-36                                                                                                                       
C4 = 0.00000000E+00                                                                                                                       
C5 = 1.39000000E-16                                                                                                                       
C6 = 1.82900000E-16                                                                                                                       
C7 = 1.72700000E-14                                                                                                                       
target ev range : (0.0000E+00, 1.1803E+01) ----> knots domain : (0.0000E+00, 3.2887E+00)                                  
singularity pts :  [1.44571915 1.89813914 1.92331609 3.25297899 3.26355872 3.43901401                                     
 3.80009123] v/vth and [ 2.281    3.932    4.037   11.54835 11.62359 12.907   15.75961] eV

python3 bte_0d3v_batched_driver.py -threads 16 -solver_type steady-state -l_max 1 -sp_order 3 -spline_qpts 15 -Efreq 0.0 -dt 1e-3 -plot_data 1 -Nr 127 --ee_collisions 0 -use_gpu 1 -cycles 3 -out_fname batched_bte/ss_test_1 -verbose 1 -max_iter 10000 -atol 1e-10 -rtol 1e-10 -c lxcat_data/eAr_crs.6sp_Tg_0.5eV -ev_max 8.196527777777778 -n_pts 3 -Te 1.09133333E+00

n0,ne,ni,Tg,E,energy,mobility,diffusion,C0,C1,C2,C3,C4,C5,C6,C7
3.22e+22,3.22e+20,3.22e+20,6000.0,64.39999999999999,1.6374333752787058,5.917012965837464e+24,1.2290748075758411e+25,2.1134120973416577e-14,1.1799605703764654e-29,4.274588289390594e-30,0.0,0.0,1.3903987273717992e-16,1.827887723061445e-16,1.7266313990616146e-14

C0 = 2.1134120973416577e-14
C1 = 1.1799605703764654e-29
C2 = 4.274588289390594e-30
C3 = 0.0
C4 = 0.0
C5 = 1.3903987273717992e-16
C6 = 1.827887723061445e-16
C7 = 1.7266313990616146e-14

"""




echo "running standalone 0D BTE"
python3 collision_driver_spherical_eulerian_dg.py -l_max 1 -c lxcat_data/eAr_crs.Biagi.3sp2r -sp_order 3 -spline_qpts 5 -steady 1 --sweep_values 31 63 127  -EbyN 1e-1 100 3 -Tg 0 -ion_deg 0
if [ $? != 0 ];
then
    echo "exit 1 : python3 collision_driver_spherical_eulerian_dg.py -l_max 1 -c lxcat_data/eAr_crs.Biagi.3sp2r -sp_order 3 -spline_qpts 5 -steady 1 --sweep_values 31 63 127  -EbyN 1e-1 100 3 -Tg 0 -ion_deg 0"
fi
echo "EXIT 0"

python3 collision_driver_spherical_eulerian_dg.py -l_max 1 -c lxcat_data/eAr_crs.Biagi.3sp2r -sp_order 3 -spline_qpts 5 -steady 1 --sweep_values 31 63 127  -EbyN 1e-1 100 3 -Tg 300 -ion_deg 0
if [ $? != 0 ];
then
    echo "exit 1 : python3 collision_driver_spherical_eulerian_dg.py -l_max 1 -c lxcat_data/eAr_crs.Biagi.3sp2r -sp_order 3 -spline_qpts 5 -steady 1 --sweep_values 31 63 127  -EbyN 1e-1 100 3 -Tg 300 -ion_deg 0"
fi
echo "EXIT 0"

python3 collision_driver_spherical_eulerian_dg.py -l_max 1 -c lxcat_data/eAr_crs.Biagi.3sp2r -sp_order 3 -spline_qpts 5 -steady 1 --sweep_values 31 63 127  -EbyN 1e-1 100 3 -Tg 300 -ion_deg 1e-3
if [ $? != 0 ];
then
    echo "exit 1 : python3 collision_driver_spherical_eulerian_dg.py -l_max 1 -c lxcat_data/eAr_crs.Biagi.3sp2r -sp_order 3 -spline_qpts 5 -steady 1 --sweep_values 31 63 127  -EbyN 1e-1 100 3 -Tg 300 -ion_deg 1e-3"
fi
echo "EXIT 0"


mkdir -p batched_bte

echo "running batched 0D BTE steady-state solver"
python3 bte_0d3v_batched_driver.py -threads 16 -solver_type steady-state -l_max 1 -sp_order 3 -spline_qpts 10 -n_pts 20 -Efreq 0.0 -dt 1e-4 -plot_data 1 -Nr 127 --ee_collisions 0 -ev_max 15.8 -Te 0.5 -use_gpu 1 -cycles 10 -out_fname batched_bte/ss -verbose 1 -max_iter 10000 -atol 1e-15 -rtol 1e-12 -c lxcat_data/eAr_crs.Biagi.3sp2r

if [ $? != 0 ];
then
    echo "exit 1 : python3 bte_0d3v_batched_driver.py -threads 64 -solver_type steady-state -l_max 1 -sp_order 3 -spline_qpts 10 -n_pts 18 -Efreq 0.0 -dt 1e-4 -plot_data 1 -Nr 127 --ee_collisions 0 -ev_max 15.8 -Te 0.5 -use_gpu 1 -cycles 10 -out_fname batched_bte/ss -verbose 1 -max_iter 10000 -atol 1e-15 -rtol 1e-12 -c lxcat_data/eAr_crs.Biagi.3sp2r"
fi
echo "EXIT 0"

echo "running batched 0D BTE time harmonic solver"
python3 bte_0d3v_batched_driver.py -threads 64 -solver_type transient -l_max 1 -sp_order 3 -spline_qpts 10 -n_pts 20 -Efreq 6e6 -dt 1e-4 -plot_data 1 -Nr 127 --ee_collisions 0 -ev_max 15.8 -Te 0.5 -use_gpu 1 -cycles 10 -out_fname batched_bte/ss -verbose 1 -max_iter 10000 -atol 1e-15 -rtol 1e-12 -c lxcat_data/eAr_crs.Biagi.3sp2r

if [ $? != 0 ];
then
    echo "exit 1 : python3 bte_0d3v_batched_driver.py -threads 64 -solver_type transient -l_max 1 -sp_order 3 -spline_qpts 10 -n_pts 18 -Efreq 0.0 -dt 1e-4 -plot_data 1 -Nr 127 --ee_collisions 0 -ev_max 15.8 -Te 0.5 -use_gpu 1 -cycles 10 -out_fname batched_bte/ss -verbose 1 -max_iter 10000 -atol 1e-15 -rtol 1e-12 -c lxcat_data/eAr_crs.Biagi.3sp2r"
fi
echo "EXIT 0"