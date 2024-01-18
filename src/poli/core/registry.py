"""This module contains utilities for registering problems and observers.
"""
from typing import List, Union, Dict
import configparser
from pathlib import Path
import warnings
import subprocess

from poli.core.abstract_problem_factory import AbstractProblemFactory

from poli.core.util.abstract_observer import AbstractObserver
from poli.core.util.objective_management.make_run_script import (
    make_run_script,
    make_observer_script,
)

from poli.objective_repository import AVAILABLE_PROBLEM_FACTORIES, AVAILABLE_OBJECTIVES

_DEFAULT = "DEFAULT"
_OBSERVER = "observer"
_RUN_SCRIPT_LOCATION = "run_script_location"

HOME_DIR = Path.home().resolve()
(HOME_DIR / ".poli_objectives").mkdir(exist_ok=True)

config_file = str(HOME_DIR / ".poli_objectives" / "config.rc")
config = configparser.ConfigParser(defaults={_OBSERVER: ""})
ls = config.read(config_file)


def set_observer(
    observer: AbstractObserver,
    conda_environment_location: str = None,
    python_paths: List[str] = None,
    observer_name: str = None,
):
    """Defines an external observer to be run in a separate process.

    This function takes an observer, a conda environment, a list of python
    environments, and an observer name. With these, it creates a script that
    can be run to instantiate the observer in a separate process. If no
    observer name is passed, the observer is set as the default observer.

    After registering an observer using this function, the user can instantiate
    it by using the ExternalObserver class, passing the relevant observer name.

    Parameters
    ----------
    observer : AbstractObserver
        The observer to be registered.
    conda_environment_location : str
        The location of the conda environment to be used.
    python_paths : List[str]
        A list of paths to append to the python path of the run script.
    observer_name : str
        The name of the observer to be registered.

    Notes
    -----
    The observer script MUST accept port and password as arguments.
    """
    run_script_location = make_observer_script(
        observer, conda_environment_location, python_paths
    )
    set_observer_run_script(run_script_location, observer_name=observer_name)


def set_observer_run_script(script_file_name: str, observer_name: str = None) -> None:
    """Sets a run_script to be called on observer instantiation.

    This function takes as input the location of a script, and an observer name.
    Using these, it sets the configuration. If no observer name is passed, the
    observer is set as the default observer.

    Parameters
    ----------
    script_file_name : str
        The location of the script to be run.
    observer_name : str
        The name of the observer to be registered.

    Notes
    -----
    The observer script MUST accept port and password as arguments.
    """
    if observer_name is None:
        observer_name = _DEFAULT
    else:
        if observer_name not in config.sections():
            config.add_section(observer_name)

    # VERY IMPORTANT: the observer script MUST accept port and password as arguments
    config[observer_name][_OBSERVER] = script_file_name
    _write_config()


def delete_observer_run_script(observer_name: str = None) -> str:
    """Deletes the run script for the given observer.

    This function takes as input an observer name. Using this, it deletes the
    run script for the given observer. If no observer name is passed, the
    default observer is deleted.

    Parameters
    ----------
    observer_name : str
        The name of the observer to be deleted.

    Returns
    -------
    location : str
        The location of the deleted run script.

    Notes
    -----
    The observer script MUST accept port and password as arguments.
    """
    if observer_name is None:
        observer_name = _DEFAULT

    location = config[observer_name][_OBSERVER]  # no need to copy
    config[observer_name][_OBSERVER] = ""
    _write_config()
    return location


def register_problem(
    problem_factory: Union[AbstractProblemFactory, str],
    conda_environment_name: Union[str, Path] = None,
    python_paths: List[str] = None,
    force: bool = False,
    **kwargs,
):
    """Registers a problem.

    This function takes a problem factory, a conda environment, a list of python
    environments, and additional keyword arguments. With these, it creates a
    script that can be run to instantiate the problem factory in a separate
    process. It also sets the configuration so that the problem factory can be
    instantiated later.

    Parameters
    ----------
    problem_factory : AbstractProblemFactory or str
        The problem factory to be registered.
    conda_environment_name : str or Path
        The name or path of the conda environment to be used.
    python_paths : List[str]
        A list of paths to append to the python path of the run script.
    force : bool
        Flag indicating whether to overwrite the existing problem.
    **kwargs : dict
        Additional keyword arguments to be passed to the problem factory.
    """
    if "conda_environment_location" in kwargs:
        conda_environment_name = kwargs["conda_environment_location"]

    problem_name = problem_factory.get_setup_information().get_problem_name()
    if problem_name not in config.sections():
        config.add_section(problem_name)
    elif not force:
        # If force is false, we warn the user and ask for confirmation
        user_input = input(
            f"Problem {problem_name} already exists. "
            f"Do you want to overwrite it? (y/[n]) "
        )
        if user_input.lower() != "y":
            warnings.warn(f"Problem {problem_name} already exists. Not overwriting.")
            return

        warnings.warn(f"Problem {problem_name} already exists. Overwriting.")

    run_script_location = make_run_script(
        problem_factory, conda_environment_name, python_paths, **kwargs
    )
    config[problem_name][_RUN_SCRIPT_LOCATION] = run_script_location
    _write_config()


def register_problem_from_repository(name: str, quiet: bool = False):
    """Registers a problem from the repository.

    This function takes a problem name, and registers it. The problem name
    corresponds to a folder inside the objective_repository folder. The
    function will:
    1. create the environment from the yaml file
    2. run the file from said enviroment (since we can't
    import the factory: it may have dependencies that are
    not installed)

    Parameters
    ----------
    name : str
        The name of the problem to be registered.
    quiet : bool, optional
        If True, we squelch the feedback about environment creation and
        problem registration, by default False.
    """
    # the name is actually the folder inside
    # poli/objective_repository, so we need
    # to
    # 1. create the environment from the yaml file
    # 2. run the file from said enviroment (since
    #    we can't import the factory: it may have
    #    dependencies that are not installed)

    # Load up the environment name
    PATH_TO_REPOSITORY = (
        Path(__file__).parent.parent / "objective_repository"
    ).resolve()

    with open(PATH_TO_REPOSITORY / name / "environment.yml", "r") as f:
        # This is a really crude way of doing this,
        # but it works. We should probably use a
        # yaml parser instead, but the idea is to keep
        # the dependencies to a minimum.
        yml = f.read()
        lines = yml.split("\n")
        conda_env_name_line = lines[0]
        assert conda_env_name_line.startswith("name:"), (
            "The first line of the environment.yml file "
            "should be the name of the environment"
        )
        env_name = lines[0].split(":")[1].strip()

    # Moreover, we should only be doing this
    # if the problem is not already registered.
    # TODO: do we?
    if name in config.sections():
        warnings.warn(f"Problem {name} already registered. Skipping")
        return

    # 1. create the environment from the yaml file
    if not quiet:
        print(f"poli 🧪: creating environment {env_name} from {name}/environment.yml")
    try:
        subprocess.run(
            " ".join(
                [
                    "conda",
                    "env",
                    "create",
                    "-f",
                    str(PATH_TO_REPOSITORY / name / "environment.yml"),
                ]
            ),
            shell=True,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        if "already exists" in e.stderr.decode():
            if not quiet:
                print(
                    f"poli 🧪: creating environment {env_name} from {name}/environment.yml"
                )
            warnings.warn(f"Environment {env_name} already exists. Will not create it.")
        else:
            raise e

    # 2. run the file from said enviroment (since
    #    we can't import the factory: it may have
    #    dependencies that are not installed)

    # Running the file
    file_to_run = PATH_TO_REPOSITORY / name / "register.py"
    command = " ".join(["conda", "run", "-n", env_name, "python", str(file_to_run)])
    warnings.warn("Running the following command: %s. " % command)

    if not quiet:
        print(f"poli 🧪: running registration of {name} from environment {env_name}")
    try:
        subprocess.run(command, check=True, shell=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Found error when running {file_to_run} from environment {env_name}: \n"
            f"{e.stderr.decode()}"
        )


def delete_problem(problem_name: str):
    """Deletes a problem.

    This function takes a problem name, and deletes it from the configuration.

    Parameters
    ----------
    problem_name : str
        The name of the problem to be deleted.
    """
    config.remove_section(problem_name)
    _write_config()


def get_problems(only_available: bool = False) -> List[str]:
    """Returns a list of registered problems.

    Parameters
    ----------
    only_available : bool
        Whether to only include the problems that can be imported directly.

    Returns
    -------
    problem_list: List[str]
        A list of registered problems.

    Notes
    -----
    If only_available is False, the problems from the repository will be
    included in the list. Otherwise, only the problems registered by the user/readily available
    will be included.
    """
    problems = config.sections()
    # problems.remove(_DEFAULT)  # no need to remove default section

    # We also pad the get_problems() with the problems
    # the user can import already without any problem,
    # i.e. the AVAILABLE_PROBLEM_FACTORIES in the
    # objective_repository
    available_problems = list(AVAILABLE_PROBLEM_FACTORIES.keys())

    if not only_available:
        # We include the problems that the user _could_
        # install from the repo. These are available in the
        # AVAILABLE_OBJECTIVES list.
        available_problems += AVAILABLE_OBJECTIVES

    problems = sorted(list(set(problems + available_problems)))

    return problems


def get_problem_factories() -> Dict[str, AbstractProblemFactory]:
    """
    Returns a dictionary with the problem factories

    Returns
    -------
    problem_factories: Dict[str, AbstractProblemFactory]
        A dictionary with the problem factories that are available.
    """
    return AVAILABLE_PROBLEM_FACTORIES


def _write_config():
    with open(config_file, "w+") as configfile:
        config.write(configfile)
