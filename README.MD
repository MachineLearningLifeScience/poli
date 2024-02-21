# poli 🧪, a library for discrete objective functions

[![Testing (conda, python 3.9)](https://github.com/MachineLearningLifeScience/poli/actions/workflows/python-tox-testing-including-conda.yml/badge.svg)](https://github.com/MachineLearningLifeScience/poli/actions/workflows/python-tox-testing-including-conda.yml)
[![Link to documentation](https://img.shields.io/badge/docs-poli_docs-blue)](https://machinelearninglifescience.github.io/poli-docs/)

poli is a library of discrete objective functions for benchmarking optimization algorithms. Examples include:
- 🔬 **stability** of mutations from a wildtype protein (using [foldx](https://foldxsuite.crg.eu/) or [rasp](https://github.com/KULL-Centre/_2022_ML-ddG-Blaabjerg)).
- 🧪 **docking scores** of ligands to proteins (using [dockstring](https://github.com/dockstring/dockstring), [pyscreener](https://github.com/coleygroup/pyscreener) and [pytdc](https://tdcommons.ai/functions/oracles/)).
- 💊 **druglikeness** or **synthetic acccesibility** of small molecules (using [rdkit](https://github.com/rdkit/rdkit) and [pytdc](https://tdcommons.ai/functions/oracles/)).

Some of `poli`'s features:
- 🔲 **isolation** of black box function calls inside conda environments. Don't worry about clashes w. black box requirements, poli will create the relevant conda environments for you.
- 🗒️ **logging** logic at the black box `__call__` level using observers.
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

f, x0, y0 = objective_factory.create(
    name="dockstring",
    target_name="drd2"
)

# x0: [['C' 'C' '1' '=' 'C' '(' 'C', ...]] (i.e. Risperidone's SMILES)
# y0: 11.9
print(x0, y0)
```

## Cite us and other relevant work

If you use `poli`, we encourage you to cite us

```
@software{GonzalezDuqueBartelsMichael:poli:2024,
author = {González-Duque, Miguel and Bartels, Simon and Michael, Richard},
month = jan,
title = {{poli: a libary of discrete sequence objectives}},
url = {https://github.com/MachineLearningLifeScience/poli},
version = {0.0.1},
year = {2024}
}
```

If you use certain black boxes, we also recommend citing the original work:

|Black box|Reference(s)|
|---------|---------|
|`dockstring`|[(García-Ortegón et al. 2022)](https://pubs.acs.org/doi/10.1021/acs.jcim.1c01334)|
|`drd3_docking`|[(Graff, Shakhnovich and Coley 2020)](https://pubs.acs.org/doi/10.1021/acs.jcim.1c01334), [(Graff and Coley 2022)](https://joss.theoj.org/papers/10.21105/joss.03950), [(Huang et al. 2022)](https://www.nature.com/articles/s41589-022-01131-2)|
|`foldx_*`|[(Schymkowitz et al. 2005)](https://academic.oup.com/nar/article/33/suppl_2/W382/2505499)|
|`gfp_cbas`|[(Brookes, Park and Listgarten 2019)](https://proceedings.mlr.press/v97/brookes19a.html)|
|`gfp_select`|[(Brookes, Park and Listgarten 2019)](https://proceedings.mlr.press/v97/brookes19a.html)|
|`penalized_logp_lambo`|[(Stanton et al. 2022)](https://github.com/samuelstanton/lambo)|
|`rasp`|[(Blaabjerg et al. 2022)](https://github.com/KULL-Centre/_2022_ML-ddG-Blaabjerg)|
|`rdkit_*`|[(rdkit)](https://github.com/rdkit/rdkit)|
|`rfp_foldx_*`|[(Schymkowitz et al. 2005)](https://academic.oup.com/nar/article/33/suppl_2/W382/2505499), [(Stanton et al. 2022)](https://github.com/samuelstanton/lambo)|
|`sa_tdc`|[(Huang et al. 2022)](https://www.nature.com/articles/s41589-022-01131-2), [(rdkit)](https://github.com/rdkit/rdkit), [(Ertl and Schuffenhauer 2009)](https://link.springer.com/article/10.1186/1758-2946-1-8)|
|`super_mario_bros`|[(Volz et al. 2018)](https://github.com/CIGbalance/DagstuhlGAN), [(González-Duque 2023)](https://github.com/miguelgondu/minimal_VAE_on_Mario) |
|`toy_continuous_problem`|[(Al-Roomi 2015)](https://www.al-roomi.org/benchmarks/unconstrained), [(Surjanovic and Bingham 2013)](https://www.sfu.ca/~ssurjano/optimization.html) |


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

