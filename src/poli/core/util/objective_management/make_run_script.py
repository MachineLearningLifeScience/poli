__author__ = "Simon Bartels"

from typing import List, Union
from pathlib import Path
import os
import sys
from os.path import basename, dirname, join
import inspect
import stat

from poli import objective
from poli.objective import ADDITIONAL_IMPORT_SEARCH_PATHES_KEY
from poli.core.util import observer_wrapper
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.util.abstract_observer import AbstractObserver

# from poli.registered_objectives import __file__ as _RUN_SCRIPTS_FOLDER

HOME_DIR = Path.home().resolve()
RUN_SCRIPTS_FOLDER = HOME_DIR / ".poli_objectives"


def make_run_script(
    problem_factory: AbstractProblemFactory,
    conda_environment_name: Union[str, Path] = None,
    python_paths: List[str] = None,
    cwd=None,
    **kwargs,
) -> str:
    """
    Creates the run script for a given problem factory.

    Inputs:
        problem_factory: the problem factory to create the run script for.
        conda_environment_name: the conda environment to use for the run script.
        (Either a string containing the name, or a path to the environment)
        python_paths: a list of paths to append to the python path of the run script.
        cwd: the working directory of the run script.
    """
    command = inspect.getfile(objective)
    return _make_run_script(
        command, problem_factory, conda_environment_name, python_paths, cwd, **kwargs
    )


def make_observer_script(
    observer: AbstractObserver,
    conda_environment: Union[str, Path] = None,
    python_paths: List[str] = None,
    cwd=None,
):
    command = inspect.getfile(observer_wrapper)
    return _make_run_script(command, observer, conda_environment, python_paths, cwd)


def _make_run_script(
    command: str,
    instantiated_object,
    conda_environment_name: Union[str, Path],
    python_paths: List[str],
    cwd=None,
    **kwargs,
):
    """
    An internal function for creating run scripts.
    """
    if cwd is None:
        cwd = str(os.getcwd())

    class_object = instantiated_object.__class__
    problem_factory_name = class_object.__name__  # TODO: potential vulnerability?
    factory_location = inspect.getfile(class_object)
    # full_problem_factory_name = basename(factory_location)[:-2] + problem_factory_name
    package_name = inspect.getmodule(instantiated_object).__name__

    if package_name == "__main__":
        package_name = basename(factory_location)[:-3]

    # else:
    package_name += "."

    full_problem_factory_name = package_name + problem_factory_name
    run_script_location = join(RUN_SCRIPTS_FOLDER, problem_factory_name + ".sh")

    if isinstance(conda_environment_name, str):
        # TODO: check that conda environment exists
        ...
    elif isinstance(conda_environment_name, Path):
        conda_environment_name = conda_environment_name.resolve()
        if not conda_environment_name.exists():
            raise ValueError(
                f"conda_environment_location {conda_environment_name} does not exist."
            )

        conda_environment_name = str(conda_environment_name)
    else:
        raise ValueError("conda_environment_location must be a string or a Path.")

    # make path to conda environment absolute
    if python_paths is None:
        python_paths = [dirname(factory_location)]

    # TODO: check that location exists and is valid environment
    python_paths = ":".join(python_paths)

    with open(
        join(dirname(__file__), "run_script_template.sht"), "r"
    ) as run_script_template_file:
        """
        load template file and fill in gaps which load conda enviroment, append python paths and call run with the
        desired factory
        """
        run_script = run_script_template_file.read() % (
            cwd,
            conda_environment_name,
            python_paths,
            ADDITIONAL_IMPORT_SEARCH_PATHES_KEY,
            command,
            full_problem_factory_name,
        )
    with open(run_script_location, "w+") as run_script_file:
        # write out run script and make it executable
        run_script_file.write(run_script)
        os.chmod(
            run_script_location, os.stat(run_script_location).st_mode | stat.S_IEXEC
        )  # make script file executable

    return run_script_location
