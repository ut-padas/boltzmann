#!/bin/bash

cd /home/dbochkov/Dropbox/Code/boltzmann/Bolsig/

export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64

# Run bolsig
./bolsigminus.exe ./minimal-argon.dat

# Move outputs to results dir
#mv argon.out ../results2/.
#mv bolsiglog.txt ../results2/.
