# `poli` 🧪, a library for discrete objective functions

[![poli base (dev, conda, python 3.9)](https://github.com/MachineLearningLifeScience/poli/actions/workflows/python-tox-testing-base.yml/badge.svg)](https://github.com/MachineLearningLifeScience/poli/actions/workflows/python-tox-testing-base.yml)
[![Link to documentation](https://img.shields.io/badge/documentation-poli_docs-blue)](https://machinelearninglifescience.github.io/poli-docs/)

`poli` is a library of discrete objective functions for benchmarking optimization algorithms.

## Black boxes

| Black box | References | Tests
|----------|----------|----------|
|   [Toy continuous functions (e.g. Ackley, Hartmann...)](https://machinelearninglifescience.github.io/poli-docs/using_poli/objective_repository/toy_continuous_problems.html) |  [(Al-Roomi 2015)](https://www.al-roomi.org/benchmarks/unconstrained), [(Surjanovic & Bingham 2013)](https://www.sfu.ca/~ssurjano/optimization.html)  |   [![poli base (dev, conda, python 3.9)](https://github.com/MachineLearningLifeScience/poli/actions/workflows/python-tox-testing-base.yml/badge.svg)](https://github.com/MachineLearningLifeScience/poli/actions/workflows/python-tox-testing-base.yml)  |
|   [Ehrlich functions](https://machinelearninglifescience.github.io/poli-docs/using_poli/objective_repository/ehrlich_functions.html)  |    [(Stanton et al. 2024)](https://arxiv.org/abs/2407.00236)  | [![poli base (dev, conda, python 3.9)](https://github.com/MachineLearningLifeScience/poli/actions/workflows/python-tox-testing-base.yml/badge.svg)](https://github.com/MachineLearningLifeScience/poli/actions/workflows/python-tox-testing-base.yml)
|   [PMO/GuacaMol benchmark](https://machinelearninglifescience.github.io/poli-docs/#small-molecules)  |   [(Brown et al. 2019)](https://arxiv.org/abs/1811.09621), [(Gao et al. 2022)](https://openreview.net/forum?id=yCZRdI0Y7G), [(Huang et al. 2021)](https://openreview.net/pdf?id=8nvgnORnoWr)  | [![poli tdc (dev, conda, python 3.9)](https://github.com/MachineLearningLifeScience/poli/actions/workflows/python-tox-testing-tdc-env.yml/badge.svg)](https://github.com/MachineLearningLifeScience/poli/actions/workflows/python-tox-testing-tdc-env.yml)
|   [Dockstring](https://machinelearninglifescience.github.io/poli-docs/using_poli/objective_repository/dockstring.html) |  [(García-Ortegón et al. 2022)](https://pubs.acs.org/doi/full/10.1021/acs.jcim.1c01334)  | [![poli dockstring (dev, conda, python 3.9)](https://github.com/MachineLearningLifeScience/poli/actions/workflows/python-tox-testing-dockstring-env.yml/badge.svg)](https://github.com/MachineLearningLifeScience/poli/actions/workflows/python-tox-testing-dockstring-env.yml)
|   [Rosetta Energy](https://machinelearninglifescience.github.io/poli-docs/using_poli/objective_repository/Rosetta_energy.html)  |   [(Chaudhury et al. 2010)](https://academic.oup.com/bioinformatics/article/26/5/689/212442)  | [![poli rosetta (conda, py3.9)](https://github.com/MachineLearningLifeScience/poli/actions/workflows/python-tox-testing-rosetta-env.yml/badge.svg)](https://github.com/MachineLearningLifeScience/poli/actions/workflows/python-tox-testing-rosetta-env.yml)
|   [RaSP](https://machinelearninglifescience.github.io/poli-docs/using_poli/objective_repository/RaSP.html)  |   [(Blaabjerg et al. 2023)](https://elifesciences.org/articles/82593)  | [![poli rasp (conda, py3.9)](https://github.com/MachineLearningLifeScience/poli/actions/workflows/python-tox-testing-rasp-env.yml/badge.svg)](https://github.com/MachineLearningLifeScience/poli/actions/workflows/python-tox-testing-rasp-env.yml)
|   [FoldX stability and SASA](https://machinelearninglifescience.github.io/poli-docs/#proteins)  |   [(Schymkowitz et al. 2005)](https://academic.oup.com/nar/article/33/suppl_2/W382/2505499?login=true) |  -  |

## Features
- 🔲 **isolation** of black box function calls inside conda environments. Don't worry about clashes w. black box requirements, poli will create the relevant conda environments for you.
- 🗒️ **logging** each black box call using observers.
-  A numpy interface. Inputs are `np.array`s of strings, outputs are `np.array`s of floats.
- `SMILES` and `SELFIES` support for small molecule manipulation.

## Getting started

To install `poli`, we recommend creating a fresh conda environment

```bash
conda create -n poli-base python=3.9
conda activate poli-base
pip install git+https://github.com/MachineLearningLifeScience/poli.git@dev
```

To check if everything went well, you can run

```bash
$ python -c "from poli import create"
```

### An example: dockstring

[![Open the minimal example in Colab](https://colab.research.google.com/assets/colab-badge.svg/)](https://colab.research.google.com/drive/1-IISCebWYfu0QhuCJ11wOag8aKOiPtls?usp=sharing)

In this next example, we estimate the docking score of the example provided in `dockstring`:
```python
import numpy as np
from poli import objective_factory

problem = objective_factory.create(
    name="dockstring",
    target_name="drd2"
)
f, x0 = problem.black_box, problem.x0
y0 = f(x0)

# x0: [['C' 'C' '1' '=' 'C' '(' 'C', ...]] (i.e. Risperidone's SMILES)
# y0: 11.9
print(x0, y0)
```

## Cite us and other relevant work

If you use certain black boxes, we expect you to cite the relevant work. [Check inside the documentation of each black box for the relevant references](https://machinelearninglifescience.github.io/poli-docs/using_poli/objective_repository/all_objectives.html).


## Where can I find the documentation?

The main documentation site is hosted as a GitHub page here: https://machinelearninglifescience.github.io/poli-docs/

### Building the documentation locally

If you install the `requirements-dev.txt` via

```bash
pip install -r requirements-dev.txt
```

then you will have access to `sphinx`. You should be able to build the documentation by going to the docs folder and building it:

```bash
cd docs/
make html
```

Afterwards, you can enter the `build` folder and open `index.html`.

