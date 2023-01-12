__author__ = 'Simon Bartels'

import os

from poli import objective_factory
from poli.core.registry import set_observer_run_script

# (once) we have to register our observer run_script with the registry
run_script_file = os.path.join(os.path.dirname(__file__), "my_observer.sh")
set_observer_run_script(run_script_file)

problem_info, f, x0, y0, run_info, terminate = objective_factory.create("WHITE_NOISE", caller_info=None)
# call objective function and observe that observer is called
print(f"The observer will be called {x0.shape[0]} time(s).")
f(x0)
terminate()
