__author__ = 'Simon Bartels'

import os

from examples.adding_an_objective_function.my_objective_function import MyProblemFactory
from poli import objective_factory
from poli.core.registry import register_problem

# (once) we have to register our run-script
run_script_file = os.path.join(os.path.dirname(__file__), "my_objective_function.sh")
my_problem_factory = MyProblemFactory()
register_problem(my_problem_factory, run_script_file)

# now we can instantiate our objective
#os.chdir(os.path.dirname(__file__))  # switch working directory to this folder, just in case
problem_name = my_problem_factory.get_setup_information().get_problem_name()
problem_info, f, x0, y0, run_info, terminate = objective_factory.create(problem_name, caller_info=None)
print(f(x0[:1, :]))
terminate()
