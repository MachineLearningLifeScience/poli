#!/bin/bash
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate
BASEDIR=$(dirname "$0")
python $BASEDIR/my_objective_function.py