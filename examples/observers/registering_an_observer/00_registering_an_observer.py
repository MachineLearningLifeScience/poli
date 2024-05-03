"""In this script, we show how to register an observer.

The core idea is _isolation_. We want to be able to
call an observer that has different dependencies from
the core of our code and experiments. This script
is then meant to run on a different environment
than the one that runs the experiments.

You could start by creating a new conda environment
called `poli__wandb`, which has wandb installed (alongside
all the other dependencies that might be necessary
to run your observer).

Check ./01_using_a_registered_observer.py for an example
of how to instantiate it after registration.
"""

from poli.core.registry import register_observer

from print_observer import SimplePrintObserver

if __name__ == "__main__":
    register_observer(
        observer=SimplePrintObserver(),
        # conda_environment_location="poli",  # when not providing the environment, we use the current one
        observer_name="simple_print_observer",
        set_as_default_observer=False,  # this is True by default!
    )
