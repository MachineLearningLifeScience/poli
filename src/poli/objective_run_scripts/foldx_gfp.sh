#!/bin/bash
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
# TODO: replace fixed path environment by more generic env
# we may for example assume that the user created the environment in the default conda folder
conda activate /home/simon/stuff/projects/bayesian_optimization/prot_bo/env
BASEDIR=$(dirname "$0")
python $BASEDIR/../objective.py poli.objectives.lambo_foldx.foldx_gfp_factory.FoldXGFPFactory