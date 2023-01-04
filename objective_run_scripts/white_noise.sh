#!/bin/bash
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate
python objective.py objectives.white_noise.white_noise_factory.WhiteNoiseFactory