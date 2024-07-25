"""This module contains utilities for creating run scripts for problems and observers.
"""

import inspect
import os
import stat
import sys
from os.path import basename, dirname, join
from pathlib import Path
from typing import List, Type, Union

from poli import external_isolated_function_script
from poli.core.abstract_isolated_function import AbstractIsolatedFunction
from poli.core.util import observer_wrapper
from poli.core.util.abstract_observer import AbstractObserver
from poli.external_isolated_function_script import ADDITIONAL_IMPORT_SEARCH_PATHES_KEY

# By default, we will store the run scripts inside the
# home folder of the user, on the hidden folder
# ~/.poli_objectives
HOME_DIR = Path.home().resolve()
RUN_SCRIPTS_FOLDER = HOME_DIR / ".poli_objectives"


def make_isolated_function_script(
    isolated_function: AbstractIsolatedFunction,
    conda_environment_name: Union[str, Path] = None,
    python_paths: List[str] = None,
    cwd=None,
    **kwargs,
):
    """
    Create a script to run the given black box, returning its location.

    Parameters
    ----------
    black_box : AbstractBlackBox
        The black box object to be executed.
    conda_environment_name : str or Path, optional
        The conda environment to activate before running the black box.
    python_paths : List[str], optional
        Additional Python paths to be added before running the black box.
    cwd : str or Path, optional
        The current working directory for the script execution.

    Returns
    -------
    run_script: str
        The path to the generated script.

    """
    command = inspect.getfile(external_isolated_function_script)
    return _make_run_script_from_template(
        command, isolated_function, conda_environment_name, python_paths, cwd, **kwargs
    )


def make_observer_script(
    observer: Type[AbstractObserver],
    conda_environment: Union[str, Path] = None,
    python_paths: List[str] = None,
    cwd=None,
):
    """
    Create a script to run the given observer.

    Parameters
    ----------
    observer : AbstractObserver
        The observer object to be executed.
    conda_environment : str or Path, optional
        The conda environment to activate before running the observer.
    python_paths : List[str], optional
        Additional Python paths to be added before running the observer.
    cwd : str or Path, optional
        The current working directory for the script execution.

    Returns
    -------
    run_script: str
        The path to the generated script.

    """
    command = inspect.getfile(observer_wrapper)
    return _make_run_script_from_template(
        command, observer, conda_environment, python_paths, cwd
    )


def _make_run_script_from_template(
    command: str,
    non_instantiated_object,
    conda_environment_name: Union[str, Path],
    python_paths: List[str],
    cwd=None,
):
    """
    An internal function for creating run scripts; returns the location of the run script.

    Parameters
    ----------
    command : str
        The command to be executed in the run script.
    instantiated_object : object
        The instantiated object representing the problem factory.
    conda_environment_name : str or Path
        The name or path of the conda environment to be used.
    python_paths : List[str]
        The list of python paths to be appended to the run script.
    cwd : str, optional
        The current working directory for the run script. If not provided, the current working directory is used.
    **kwargs : dict
        Additional keyword arguments (currently unused).

    Returns
    -------
    run_script: str
        The location of the generated run script.
    """
    if cwd is None:
        cwd = str(os.getcwd())

    # class_object = instantiated_object.__class__
    class_object = non_instantiated_object
    problem_factory_name = class_object.__name__  # TODO: potential vulnerability?
    factory_location = inspect.getfile(class_object)
    package_name = inspect.getmodule(non_instantiated_object).__name__

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
        if conda_environment_name is None:
            conda_environment_name = sys.executable[: -len("/bin/python")]
            conda_environment_name = str(os.path.abspath(conda_environment_name))
        else:
            raise ValueError(
                "If specified, conda_environment_location must be a string or a Path."
            )

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
