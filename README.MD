# poli 🧪, a library for discrete objective functions

[![Testing (conda, python 3.9)](https://github.com/MachineLearningLifeScience/poli/actions/workflows/python-tox-testing-including-conda.yml/badge.svg)](https://github.com/MachineLearningLifeScience/poli/actions/workflows/python-tox-testing-including-conda.yml)
[![Link to documentation](https://img.shields.io/badge/docs-poli_docs-blue)](https://machinelearninglifescience.github.io/poli-docs/)

poli is a library of discrete objective functions for benchmarking optimization algorithms. Examples include:
- 🔬 **stability** of mutations from a wildtype protein (using [foldx](https://foldxsuite.crg.eu/) or [rasp](https://github.com/KULL-Centre/_2022_ML-ddG-Blaabjerg)).
- 🧪 **docking scores** of ligands to proteins (using [dockstring](https://github.com/dockstring/dockstring), [pyscreener](https://github.com/coleygroup/pyscreener) and [pytdc](https://tdcommons.ai/functions/oracles/)).
- 💊 **druglikeness** or **synthetic acccesibility** of small molecules (using [rdkit](https://github.com/rdkit/rdkit) and [pytdc](https://tdcommons.ai/functions/oracles/)).

Some of `poli`'s features:
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
$ python -c "from poli import get_problems ; print(get_problems())"
['aloha', ..., 'white_noise']
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

