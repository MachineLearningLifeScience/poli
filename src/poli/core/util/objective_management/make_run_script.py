__author__ = 'Simon Bartels'

import os
from os.path import basename, dirname, join
import inspect
import stat

from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.registered_objectives import __file__ as _RUN_SCRIPTS_FOLDER


RUN_SCRIPTS_FOLDER = dirname(_RUN_SCRIPTS_FOLDER)


def make_run_script(problem_factory: AbstractProblemFactory, conda_environment_location: str = None,
                    python_paths: list[str] = None) -> str:
    problem_factory_name = problem_factory.__class__.__name__
    factory_location = inspect.getfile(problem_factory.__class__)
    full_problem_factory_name = basename(factory_location)[:-2] + problem_factory_name
    run_script_location = join(RUN_SCRIPTS_FOLDER, problem_factory_name + ".sh")
    if conda_environment_location is not None:
        if python_paths is None:
            python_paths = [dirname(factory_location)]
        # TODO: check that location exists and is valid environment
        python_paths = ":".join(python_paths)
        with open(join(dirname(__file__), "run_script_template.sht"), "r") as run_script_template_file:
            run_script = run_script_template_file.read() % (conda_environment_location, python_paths,
                                                            full_problem_factory_name)
        with open(run_script_location, "w+") as run_script_file:
            run_script_file.write(run_script)
            os.chmod(run_script_location, os.stat(run_script_location).st_mode | stat.S_IEXEC)  # make script file executable
    else:
        raise NotImplementedError("Currently a conda environment is required.")
    return run_script_location
