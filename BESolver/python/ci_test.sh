#!/bin/bash

##python3 bte_0d3v_batched_driver.py -threads 16 -solver_type steady-state -l_max 1 -sp_order 3 -spline_qpts 10 -n_pts 1 -Efreq 0.0 -dt 1e-3 -plot_data 1 -Nr 63 --ee_collisions 1 -ev_max 25.03333333333333  -Te 1.3419999999999999 -use_gpu 1 -cycles 10 -out_fname batched_bte/ss -verbose 1 -max_iter 10000 -atol 1e-15 -rtol 1e-12 -c lxcat_data/eAr_crs.Biagi.3sp2r
## python3 collision_driver_spherical_eulerian_dg.py -l_max 1 -c lxcat_data/eAr_crs.Biagi.3sp2r -sp_order 3 -spline_qpts 5 -steady 1 --sweep_values 63  -EbyN 1 100 1 -Tg 6000 -ion_deg 1e-2 -store_csv 1


#python3 collision_driver_spherical_eulerian_dg.py -l_max 1 -c lxcat_data/eAr_crs.Biagi.3sp2r -sp_order 3 -spline_qpts 5 -steady 1 --sweep_values 63  -EbyN 1 100 1 -Tg 6000 -ion_deg 0 -store_csv 1
#python3 bte_0d3v_batched_driver.py -threads 16 -solver_type steady-state -l_max 1 -sp_order 3 -spline_qpts 10 -n_pts 1 -Efreq 0.0 -dt 1e-3 -plot_data 1 -Nr 63 --ee_collisions 0 -ev_max 16.983333333333333 -Te 1.8313333333333333 -use_gpu 1 -cycles 10 -out_fname batched_bte/ss3 -verbose 1 -max_iter 10000 -atol 1e-15 -rtol 1e-12 -c lxcat_data/eAr_crs.Biagi.3sp2r

# python3 collision_driver_spherical_eulerian_dg.py -l_max 1 -c lxcat_data/eAr_crs.Biagi.3sp2r -sp_order 3 -spline_qpts 5 -steady 1 --sweep_values 63  -EbyN 10 100 1 -Tg 0 -ion_deg 0 -store_csv 1
#python3 bte_0d3v_batched_driver.py -threads 16 -solver_type steady-state -l_max 1 -sp_order 3 -spline_qpts 10 -n_pts 1 -Efreq 0.0 -dt 1e-3 -plot_data 1 -Nr 63 --ee_collisions 0 -ev_max 20  -Te 4.158666666666667 -use_gpu 1 -cycles 10 -out_fname batched_bte/ss3 -verbose 1 -max_iter 10000 -atol 1e-15 -rtol 1e-12 -c lxcat_data/eAr_crs.Biagi.3sp2r


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