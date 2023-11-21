"""In this script, we show how to register an observer.

The core idea is _isolation_. We want to be able to
call an observer that has different dependencies from
the core of our code and experiments. This script
is then meant to run on a different environemnt
than the one that runs the experiments.

You could start by creating a new conda environment
called `poli__wandb`, which has wandb installed (alongside
all the other dependencies that might be necessary
to run your observer).

Check ./using_a_registered_observer.py for an example
of how to instantiate it after registration.
"""

from poli.core.registry import set_observer

from wandb_observer import WandbObserver

if __name__ == "__main__":
    set_observer(
        observer=WandbObserver(initial_step=0),
        conda_environment_location="poli__wandb",
        observer_name="wandb",
    )
