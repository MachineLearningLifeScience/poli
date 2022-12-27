# Protein Objective Library
An easy-to-use, plug-and-play library to benchmark protein-related Bayesian optimization algorithms.
Primarily, this library provides objective functions encapsulating any dependencies.
Main benefit is that this allows to develop algorithms that use (say) tensorflow without having to worry that the objective was written in (say) torch.

Furthermore, this library provides basic logging and plotting functions associated with Bayesian optimization for proteins. 
## Basic usage
### Calling a benchmark
The main file is `objective_function_factory.py`
### Logging
In the `config.rc` file you need to define a logger and an observer.
With the logger you define the logging backend, with the observer you control which quantities will be recorded.
## Adding your own benchmark
### Design principles
 * This library depends only on a generic version of `numpy`.
 * Therefore imports have to be encapsulated.
   * They MUST NOT appear at the top of the file as usual.
### Naming conventions
Since imports are dynamic it is vital that you stick to the naming convention.
## Software architecture
