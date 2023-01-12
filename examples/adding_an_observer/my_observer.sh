#!/bin/bash
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate
BASEDIR=$(dirname "$0")
PYTHONPATH="${PYTHONPATH}:/${BASEDIR}"
python $BASEDIR/my_observer.py