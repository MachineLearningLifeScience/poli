# Protein Objective Library (POLi)

[![Testing (conda, python 3.9)](https://github.com/MachineLearningLifeScience/poli/actions/workflows/python-tox-testing-including-conda.yml/badge.svg)](https://github.com/MachineLearningLifeScience/poli/actions/workflows/python-tox-testing-including-conda.yml)

An easy-to-use, plug-and-play library to benchmark protein-related discrete optimization algorithms.
Primarily, this library provides a way to encapsulate objective functions and their dependencies.
The main benefit is that this allows to develop optimization algorithms that use (say) tensorflow without having to worry that the objective was written in (say) torch.

For any code written by other authors (whether objective function or algorithm) this library allows to benchmark and analyse it without too much interaction.

On purpose, logging is kept at the objective function side.
This allows easier benchmarking of algorithms from other authors.
Algorithm-specific logging can be done internally, on the site of the algorithm if necessary.

## Basic usage

### Installation

Run the following from the main directory (where this README file is also located) to install the package in development mode (that is, modifications to the source code is directly visible to file importing it without the need for reinstallation).
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

### Calling objective functions from the repository

As you might have noticed, you can get a list of the registered problems using the `get_problems` method inside `poli.core.registry`. You can also get a list of objective functions available for installing/registration using `from poli.objective_repository import AVAILABLE_PROBLEM_FACTORIES`:

```bash
$ python -c "from poli.objective_repository import AVAILABLE_PROBLEM_FACTORIES ; print(AVAILABLE_PROBLEM_FACTORIES)"
'{"white_noise": <WhiteNoiseProblemFactory(L=inf)>, ...}'
```

If the function isn't there, you may:
- Install all the required dependencies for running the file. Check the relevant environment under `poli/objective_repository/problem_name/environment.yml`.
- Implement the problem yourself! An example of how to do this can be found in `poli/examples/a_simple_objective_function_registration`.

