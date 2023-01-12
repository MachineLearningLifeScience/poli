#!/bin/bash
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
# TODO: replace fixed path environment by more generic env
# we may for example assume that the user created the environment in the default conda folder
conda activate /home/simon/stuff/projects/bayesian_optimization/prot_bo/env
BASEDIR=$(dirname "$0")
PYTHONPATH="${PYTHONPATH}:${BASEDIR}/.."
#cd BASEDIR/..  # change working directory
#PYTHONPATH="${PYTHONPATH}:${PWD}"
#echo $PYTHONPATH
python $BASEDIR/../../src/poli/objective.py lambo_foldx.foldx_gfp_factory.FoldXGFPFactory