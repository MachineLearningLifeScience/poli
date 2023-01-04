# TO DO
 * make install script that creates environments and run_scripts 
 * make this a python package
 * add example how to easily exchange loggers and observers
# Protein Objective Library
An easy-to-use, plug-and-play library to benchmark protein-related Bayesian optimization algorithms.
Primarily, this library provides objective functions encapsulating any dependencies.
Main benefit is that this allows to develop algorithms that use (say) tensorflow without having to worry that the objective was written in (say) torch.

Furthermore, this library provides basic logging and plotting functions associated with Bayesian optimization for proteins.
## Basic usage
### Installation
Run the following from the main directory (where this README file is also located) to install the package in development mode (that is, modifications to the source code is directly visible to file importing it without the need for reinstallation).
```
pip install -e .
```

### Calling a benchmark
The main file is `objective_function_factory.py`
Available objectives are given as shell scripts in `objective_run_scripts`.
### Minimal working example
Optimizing an objective function that just returns a draw from a standard normal.
```
import poli.objective_factory

f, x0, y0, run_info, terminate = objective_factory.create("white_noise", caller_info=None)
print(f(x0[:1, :]))
terminate()
```
### Logging
In the `config.rc` file you need to define a logger and an observer.
With the logger you define the logging backend, with the observer you control which quantities will be recorded.
An example configuration file could look like this:
```
[DEFAULT]
observer = logger.observers.rudimentary_observer.RudimentaryObserver
logger = logger.wandb.wandb_logger.WandBLogger
```
This uses the WeightsAndBiases logging backend, and the rudimentary observer logs for each function evaluation the input  and the value.
### More elaborate example
## Adding your own benchmark
### Design principles
 * This library depends only on a generic version of `numpy`.
 * Therefore make sure that imports are called only when your objective is actually used.
 * Having an associated shell script for each objective function allows to load different environments for conflicting objective function implementations.
### Minimal working example
Since imports are dynamic it is vital that you stick to the naming convention.
## Software architecture
