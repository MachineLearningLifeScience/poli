import unittest
import sys
import os

from poli import objective_factory
from poli.core import registry
from poli.core.abstract_problem_factory import AbstractProblemFactory


class ProblemRegistration(unittest.TestCase):
    def test_registering_toy_problem(self):
        factory_location = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'examples',
                                        'adding_an_objective_function')
        sys.path.append(os.path.abspath(factory_location))  # add folder to path
        import my_objective_function
        factory: AbstractProblemFactory = my_objective_function.MyProblemFactory()
        registry.register_problem(factory, "")

        location = registry.delete_observer_run_script()  # delete observer, we just want to test our little problem
        try:
            problem_information, f, x0, y0, observer_info = objective_factory.create(factory.get_setup_information().get_problem_name())
            f.terminate()
        finally:
            registry.set_observer_run_script(location)  # restore observer, even if something goes wrong


if __name__ == '__main__':
    unittest.main()
