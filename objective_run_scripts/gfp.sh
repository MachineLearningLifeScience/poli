#!/bin/bash
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate /home/simon/stuff/projects/bayesian_optimization/prot_bo/lambo/lambo/lambo-env
python