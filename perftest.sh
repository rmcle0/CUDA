#!/bin/bash
grid_size=100
for (( ; grid_size <= 700; grid_size+=100 ))
do
    bsub -oo "thrust${grid_size}.txt" -gpu "num=1:mode=exclusive_process" ./thrust none "$grid_size"  100
    mpisubmit.pl -w 00:30 consequential -- none "$grid_size" 100
done