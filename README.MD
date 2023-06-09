# Protein Objective Library (POLi)
An easy-to-use, plug-and-play library to benchmark protein-related Bayesian optimization algorithms.
Primarily, this library provides objective functions encapsulating any dependencies.
Main benefit is that this allows to develop algorithms that use (say) tensorflow without having to worry that the objective was written in (say) torch.

For any code written by other authors (whether objective function or algorithm) this library allows to benchmark and analyse it without too much interaction.

On purpose, logging is kept at the objective function side.
This allows easier benchmarking of algorithms from other authors.
Algorithmspecific logging can be done internally, on the site of the algorithm if necessary.
## Basic usage
### Installation
Run the following from the main directory (where this README file is also located) to install the package in development mode (that is, modifications to the source code is directly visible to file importing it without the need for reinstallation).
```
pip install -e .
```

### Minimal working example
Optimizing an objective function that just returns a draw from a standard normal.
```
import poli.objective_factory

problem_info, f, x0, y0, run_info = objective_factory.create("WHITE_NOISE", caller_info=None)
print(f(x0[:1, :]))
f.terminate()
```
### Calling a benchmark
Calling a registered benchmark works as above, except that you have to exchange "WHITE_NOISE" for your objective name.
You can see the registered benchmarks by calling `get_problems()` provided by `poli.core.registry`.
For registering your own objectives, write a class inheriting from `AbstractProblemFactory`. 
See `adding_an_objective_function` in the `examples` how to do this.

### Logging
You can define an observer that gets notified everytime the objective function is called.
For that you have to write a python class that inherits from `AbstractObserver`. 
See `adding_an_observer.py` in the `examples` how to do this.
NOTE: If your observer is NOT at the root of your project, you have to specify the argument `python_paths` accordingly.
Minimally, you should set `python_paths=[your_project_root]`.

## Software architecture
At the heart of the library are the files `objective.py` and `objective_factory.py`.
When a problem is registered in the `registry`, a shell script is created that takes three arguments:

1. the package name of instance of `AbstractProblemFactory` 
2. a port number
3. a password

When a problem is instantiated in `objective_factory.py`, a localhost connection is created listening at a free port.
The shell script is called with the desired problem factory, the port and the password.
Inside this script, the desired environment is started and then `objective.py` is called, which makes the connection to the server.
