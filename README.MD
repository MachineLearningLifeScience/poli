# `poli`, a library for discrete objective functions

[![Testing (conda, python 3.9)](https://github.com/MachineLearningLifeScience/poli/actions/workflows/python-tox-testing-including-conda.yml/badge.svg)](https://github.com/MachineLearningLifeScience/poli/actions/workflows/python-tox-testing-including-conda.yml)

`poli` is an easy-to-use, plug-and-play library to query black-box functions in biology and cheminformatics. Examples include:
- Computing the **stability** of mutations from a wildtype protein (using `foldx`).
- Computing the **docking scores** of ligands to proteins (using [`pyscreener`](https://github.com/coleygroup/pyscreener) and [`pytdc`](https://tdcommons.ai/functions/oracles/)).

When dependencies get tough, this library provides a way to encapsulate objective functions into isolated `conda` environments. The main benefit is that this allows to develop optimization algorithms that use (say) tensorflow without having to worry about the specific dependencies of the objective function. Moreover, `poli` provides a way to inject logging into the objective function evaluations using observers.

## Basic usage

### Installation

Run the following from the main directory (where this README file is also located) to install the package in development mode (this way you could modify its contents and have these changes be reflected without having to re-install).
```
pip install -e .
```

If you are not interested in debugging locally, you could also install a stable version by running
```
pip install git+https://github.com/MachineLearningLifeScience/poli.git@master
```

The bleeding-edge is contained in the `dev` branch, so if you want to install that, change `@master` for `@dev`.

To make sure everything went well, you can test your `poli` installation by running

```bash
$ python -c "from poli.core.registry import get_problems ; print(get_problems())"
['aloha', 'white_noise']
```

If the installation isn't fresh/isn't the only one in your system, you might actually get some more names inside that list.

### Minimal working example
Defining an objective function that just returns a draw from a standard normal.
```python
# Check examples/minimal_working_example.py
import numpy as np
from poli import objective_factory

problem_info, f, x0, y0, run_info = objective_factory.create(name="white_noise")

x = np.array([["1", "2", "3"]])  # must be of shape [b, L], in this case [1, 3].
for _ in range(5):
    print(f"f(x) = {f(x)}")

```

### When you have the right dependencies...

If you have enough dependencies to run an objective function, it will become available. For example, try running `pip install rdkit selfies` followed by the `get_problems()` statement from above:

```bash
$ pip install rdkit selfies
$ python -c "from poli.core.registry import get_problems ; print(get_problems())"
['aloha', 'rdkit_logp', 'rdkit_qed', 'white_noise']
```

Now that both `rdkit` and `selfies` are in the current environment, problems like computing `logp` and `qed` of SELFIES or SMILES strings become available.

### Calling objective functions in isolated enviroments

To get a list of all avilable objective functions, you can pass the `include_repository=True` flag to `get_problems`:

```bash
$ python -c "from poli.core.registry import get_problems ; print(get_problems(include_repository=True))"
['aloha', 'drd3_docking', 'foldx_sasa', 'foldx_stability', ..., 'white_noise']
```

**Most of these objective functions can be run out-of-the-box** in isolated enviroments. For example, consider computing the synthetic accessibility of a molecule using `pytdc`. This problem is called `sa_tdc` in `poli`, and can easily be run without having the right dependencies installed:

```python
from poli import objective_factory
import numpy as np

problem_info, f, x0, y0, run_info = objective_factory.create(
    name="sa_tdc",
    force_register=True,
    string_representation="SELFIES",
)

x = np.array([["[C]", "[C]", "[C]"]])
print(f"f({x}) = {f(x)}")

```

## Where can I find the documentation?



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
