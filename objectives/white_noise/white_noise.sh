#!/bin/bash
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate /home/simon/stuff/projects/bayesian_optimization/prot_bo/env
BASEDIR=$(dirname "$0")
PYTHONPATH="${PYTHONPATH}:/${BASEDIR}"
python $BASEDIR/../../src/poli/objective.py white_noise_factory.WhiteNoiseFactory