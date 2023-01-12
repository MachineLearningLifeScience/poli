# TO DO
 * make install script that creates environments and run_scripts 
 * add example how to easily exchange loggers and observers
 * consider switching to Hydra (though this creates a dependency that might break in the future)
# Protein Objective Library (POLi)
An easy-to-use, plug-and-play library to benchmark protein-related Bayesian optimization algorithms.
Primarily, this library provides objective functions encapsulating any dependencies.
Main benefit is that this allows to develop algorithms that use (say) tensorflow without having to worry that the objective was written in (say) torch.

On purpose, logging is kept at the objective function side.
This allows easier benchmarking of algorithms from other authors.
Algorithmspecific logging can be done internally, on the site of the algorithm if necessary.
## Basic usage
### Installation
Run the following from the main directory (where this README file is also located) to install the package in development mode (that is, modifications to the source code is directly visible to file importing it without the need for reinstallation).
```
pip install -e .
```

### Calling a benchmark
The main file is `objective_function_factory.py`
### Minimal working example
Optimizing an objective function that just returns a draw from a standard normal.
```
import poli.objective_factory

f, x0, y0, run_info, terminate = objective_factory.create("WHITE_NOISE", caller_info=None)
print(f(x0[:1, :]))
terminate()
```
### Logging
You can define an observer that gets notified everytime the objective function is called.
For that you have to write a python class that inherits from `AbstractObserver` and a shell script that instantiates your environment. 
See `adding_an_observer.py` in the `examples` how to do this.
### More elaborate example
## Adding your own benchmark
### Design principles
 * This library depends only on a generic version of `numpy`.
 * Therefore make sure that imports are called only when your objective is actually used.
 * Having an associated shell script for each objective function allows to load different environments for conflicting objective function implementations.
### Minimal working example
Since imports are dynamic it is vital that you stick to the naming convention.
## Software architecture
