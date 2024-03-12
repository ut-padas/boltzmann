#!/bin/bash

"""
-------------------------------------------------------------------------------------------------------------------------------------------------------------------
python3 collision_driver_spherical_eulerian_dg.py -l_max 1 -c lxcat_data/eAr_crs.6sp_Tg_0.5eV -sp_order 3 -spline_qpts 5 -steady 1 --sweep_values 63 127  -EbyN 2 1 1 -Tg 6000 -ion_deg 1e-2 -ns_by_n0 0.999 0.00033333333333333365 0.00033333333333333365 0.00033333333333333365 -store_csv 1

python3 bte_0d3v_batched_driver.py -threads 16 -solver_type steady-state -l_max 1 -sp_order 3 -spline_qpts 5 -Efreq 0.0 -dt 1e-3 -plot_data 1 -Nr 127 --ee_collisions 1 -use_gpu 1 -cycles 3 -out_fname batched_bte/ss_test_0 -verbose 1 -max_iter 10000 -atol 1e-10 -rtol 1e-10 -c lxcat_data/eAr_crs.6sp_Tg_0.5eV -ev_max 17.425 -n_pts 3 -Te 8.54666667E-01


number of total collisions = 8                                                                                                            
READCOLLISIONS                                                                                                                            
 Ar                                                           
crs_file.txt                                                                                                                              
 C1    Ar    Elastic                                                                                                                      
 C2    Ar    Excitation    11.55 eV                                                                                                       
 C3    Ar    Excitation    11.62 eV                              
 C4    Ar    Excitation    12.91 eV                              
 C5    Ar    Ionization    15.76 eV                              
 Ar*(1)  
crs_file.txt                           
 C6    Ar*(1)    Ionization    4.04 eV                           
 Ar*(2)                                                                                                                                   
crs_file.txt                                            
 C7    Ar*(2)    Ionization    3.93 eV                           
 Ar*(3)                          
crs_file.txt                    
 C8    Ar*(3)    Ionization    2.28 eV                           
RUN
R1    2.00 Td    1.28 eV
SAVERESULTS
argon.out
FINISHED
Found EbN =  2.0
Found mu =  1.282
Found mobility =  6.488e+24
Found diffusion =  5.544e+24
Found coulomb logarithm =  6.525
Found collision rate no. 1 =  1.624e-14
Found collision rate no. 2 =  1.708e-21
Found collision rate no. 3 =  2.673e-21
Found collision rate no. 4 =  6.109e-23
Found collision rate no. 5 =  8.675e-23
Found collision rate no. 6 =  6.311e-16
Found collision rate no. 7 =  7.717e-16
Found collision rate no. 8 =  1.938e-14
Adding vectors to lists for Etil = 2.000e+00...
bolsig temp      = 8.54666667E-01
bolsig mobility  = 6.48800000E+24
bolsig diffusion = 5.54400000E+24
bolsig collision rates
C0 = 1.62400000E-14
C1 = 1.70800000E-21
C2 = 2.67300000E-21
C3 = 6.10900000E-23
C4 = 8.67500000E-23
C5 = 6.31100000E-16
C6 = 7.71700000E-16
C7 = 1.93800000E-14
target ev range : (0.0000E+00, 2.5092E+01) ----> knots domain : (0.0000E+00, 5.4184E+00)
singularity pts :  [1.63366972 2.14490653 2.17335661 3.6758822  3.68783735 3.88610268

n0,ne,ni,Tg,E,energy,mobility,diffusion,C0,C1,C2,C3,C4,C5,C6,C7
3.22e+22,3.22e+20,3.22e+20,6000.0,64.39999999999999,1.2839072563970342,6.536644952404083e+24,1.6800965763177847e+25,1.6277537101055533e-14,1.7441450715096757e-21,2.70686793156887e-21,6.191387488208492e-23,8.625701094242665e-23,6.368977613583206e-16,7.78598298958007e-16,1.950828891778716e-14

1.6277537101055533e-14
1.7441450715096757e-21
2.70686793156887e-21
6.191387488208492e-23
8.625701094242665e-23
6.368977613583206e-16
7.78598298958007e-16
1.950828891778716e-14

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

python3 collision_driver_spherical_eulerian_dg.py -l_max 1 -c lxcat_data/eAr_crs.6sp_Tg_0.5eV -sp_order 3 -spline_qpts 5 -steady 1 --sweep_values 63 127  -EbyN 2 1 1 -Tg 6000 -ion_deg 0 -ns_by_n0 0.999 0.00033333333333333365 0.00033333333333333365 0.00033333333333333365 -store_csv 1

number of total collisions = 8                                
READCOLLISIONS                                                                                                                            
 Ar                                                                                                                                       
crs_file.txt                                                                                                                              
 C1    Ar    Elastic                                             
 C2    Ar    Excitation    11.55 eV                              
 C3    Ar    Excitation    11.62 eV                              
 C4    Ar    Excitation    12.91 eV                              
 C5    Ar    Ionization    15.76 eV                              
 Ar*(1)                                        
crs_file.txt
 C6    Ar*(1)    Ionization    4.04 eV                           
 Ar*(2)
crs_file.txt
 C7    Ar*(2)    Ionization    3.93 eV                           
 Ar*(3)
crs_file.txt
 C8    Ar*(3)    Ionization    2.28 eV                           
RUN
R1    2.00 Td    1.65 eV
SAVERESULTS
argon.out
FINISHED
Found EbN =  2.0
Found mu =  1.655
Found mobility =  5.813e+24
Found diffusion =  1.289e+25
Found collision rate no. 1 =  2.142e-14
Found collision rate no. 2 =  3.208e-31
Found collision rate no. 3 =  2.337e-31
Found collision rate no. 4 =  8.794e-36
Found collision rate no. 5 =  0.0
Found collision rate no. 6 =  1.286e-16
Found collision rate no. 7 =  2.003e-16
Found collision rate no. 8 =  1.684e-14
Adding vectors to lists for Etil = 2.000e+00...
bolsig temp      = 1.10333333E+00
bolsig mobility  = 5.81300000E+24
bolsig diffusion = 1.28900000E+25
bolsig collision rates
C0 = 2.14200000E-14
C1 = 3.20800000E-31
C2 = 2.33700000E-31
C3 = 8.79400000E-36
C4 = 0.00000000E+00
C5 = 1.28600000E-16
C6 = 2.00300000E-16
C7 = 1.68400000E-14
target ev range : (0.0000E+00, 1.1878E+01) ----> knots domain : (0.0000E+00, 3.2810E+00)
singularity pts :  [1.43783574 1.88778871 1.91282837 3.23524071 3.24576275 3.42026129
 3.77936958] v/vth and [ 2.281    3.932    4.037   11.54835 11.62359 12.907   15.75961] eV


python3 bte_0d3v_batched_driver.py -threads 16 -solver_type steady-state -l_max 1 -sp_order 3 -spline_qpts 5 -Efreq 0.0 -dt 1e-3 -plot_data 1 -Nr 127 --ee_collisions 0 -use_gpu 1 -cycles 3 -out_fname batched_bte/ss_test_1 -verbose 1 -max_iter 10000 -atol 1e-10 -rtol 1e-10 -c lxcat_data/eAr_crs.6sp_Tg_0.5eV -ev_max 8.248611111111112 -n_pts 3 -Te 1.10333333E+00

n0,ne,ni,Tg,E,energy,mobility,diffusion,C0,C1,C2,C3,C4,C5,C6,C7
3.22e+22,3.22e+20,3.22e+20,6000.0,64.39999999999999,1.655088194855104,5.811276750945065e+24,1.2235246608379894e+25,2.1420903714100646e-14,1.9276529014380293e-29,8.797713730333763e-30,0.0,0.0,1.2857236624591265e-16,2.0019405913776227e-16,1.6840849820164827e-14

2.1420903714100646e-14
1.9276529014380293e-29
8.797713730333763e-30
0.0
0.0
1.2857236624591265e-16
2.0019405913776227e-16
1.6840849820164827e-14
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