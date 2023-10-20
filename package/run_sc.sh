#!/bin/bash
#SBATCH --mem=100G
#SBATCH --time=2-0


module load ncurses/6.0
module load GCCcore/11.2.0


python3 run_sc.py -cp1 './datas/d1/sample.h5' -f1 "./datas/d3/" -r1 200 -c1 0.001 -e1 10 -cp2 './datas/d2/sample.h5' -f2 "./datas/d4/" -r2 200 -c2 0.001 -e2 10