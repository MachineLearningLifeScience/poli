__author__ = 'Simon Bartels'

import os
from os.path import basename, dirname, join
import inspect
import stat
from typing import List

from poli import objective
from poli.core.util import observer_wrapper
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.util.abstract_observer import AbstractObserver
from poli.registered_objectives import __file__ as _RUN_SCRIPTS_FOLDER


RUN_SCRIPTS_FOLDER = dirname(_RUN_SCRIPTS_FOLDER)


def make_run_script(problem_factory: AbstractProblemFactory, conda_environment_location: str = None,
                    python_paths: List[str] = None) -> str:
    return _make_run_script(problem_factory, conda_environment_location, python_paths)


def make_observer_script(observer: AbstractObserver, conda_environment_location: str = None,
                         python_paths: List[str] = None):
    return _make_run_script(observer, conda_environment_location, python_paths)


def _make_run_script(instantiated_object, conda_environment_location, python_paths):
    class_object = instantiated_object.__class__
    problem_factory_name = class_object.__name__  # TODO: potential vulnerability?
    factory_location = inspect.getfile(class_object)
    full_problem_factory_name = basename(factory_location)[:-2] + problem_factory_name
    run_script_location = join(RUN_SCRIPTS_FOLDER, problem_factory_name + ".sh")
    if conda_environment_location is not None:
        # make path to conda environment absolute
        conda_environment_location = str(os.path.abspath(conda_environment_location))
        if python_paths is None:
            python_paths = [dirname(factory_location)]
        # TODO: check that location exists and is valid environment
        python_paths = ":".join(python_paths)
        with open(join(dirname(__file__), "run_script_template.sht"), "r") as run_script_template_file:
            """
            load template file and fill in gaps which load conda enviroment, append python paths and call run with the 
            desired factory
            """
            run_script = run_script_template_file.read() % (conda_environment_location, python_paths,
                                                            factory_location + " " + full_problem_factory_name)
        with open(run_script_location, "w+") as run_script_file:
            # write out run script and make it executable
            run_script_file.write(run_script)
            os.chmod(run_script_location, os.stat(run_script_location).st_mode | stat.S_IEXEC)  # make script file executable
    else:
        raise NotImplementedError("Currently a conda environment is required.")
    return run_script_location
