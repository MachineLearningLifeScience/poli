from __future__ import annotations

import configparser
import importlib
import logging
import shutil
import subprocess
from pathlib import Path

from poli.core.registry import _ISOLATED_FUNCTION_SCRIPT_LOCATION, _OBSERVER
from poli.core.util.inter_process_communication.process_wrapper import ProcessWrapper

from .external_function import ExternalFunction

HOME_DIR = Path.home().resolve()
(HOME_DIR / ".poli_objectives").mkdir(exist_ok=True)

config_file = str(HOME_DIR / ".poli_objectives" / "config.rc")
config = configparser.ConfigParser(defaults={_OBSERVER: ""})
ls = config.read(config_file)


def load_config():
    """Loads the configuration file containing which objectives are registered.

    Returns
    -------
    config : configparser.ConfigParser
        The configuration file.

    """
    HOME_DIR = Path.home().resolve()
    config_file = str(HOME_DIR / ".poli_objectives" / "config.rc")
    config = configparser.ConfigParser(defaults={_OBSERVER: ""})
    _ = config.read(config_file)

    return config


def __read_env_name(environment_file: Path) -> str:
    with open(environment_file, "r") as f:
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
    return env_name


def __create_conda_env(environment_file: Path, quiet: bool = False):
    env_name = __read_env_name(environment_file=environment_file)

    # 1. create the environment from the yaml file
    if not quiet:
        print(f"poli 🧪: creating environment {env_name} from {environment_file}")
    try:
        subprocess.run(
            " ".join(["conda", "env", "create", "-f", str(environment_file)]),
            shell=True,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        if "already exists" in e.stderr.decode():
            if not quiet:
                print(f"poli 🧪: {env_name} already exists.")
        else:
            raise e


def __register_isolated_function_from_repository(
    name: str, quiet: bool = False
) -> None:
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
    # Load up the environment name
    PATH_TO_REPOSITORY = (
        Path(__file__).parent.parent.parent.parent / "objective_repository"
    ).resolve()

    name_without_isolated = name.replace("__isolated", "")
    environment_file = PATH_TO_REPOSITORY / name_without_isolated / "environment.yml"
    isolated_file = PATH_TO_REPOSITORY / name_without_isolated / "isolated_function.py"

    __register_isolated_file(
        environment_file=environment_file,
        isolated_file=isolated_file,
        name_for_show=name,
        quiet=quiet,
    )


def __register_isolated_function_from_core(name: str, quiet: bool = False) -> None:
    """
    Registers an isolated function from the core package.

    At the moment, we only have one isolated function
    available in the core package, which is the TDCIsolatedFunction.

    Parameters
    ----------
    name : str
        The name of the isolated function to register. At the moment,
        the only available isolated function is "tdc__isolated".
    quiet : bool, optional
        If True, we squelch the feedback about environment creation and
        problem registration, by default False.
    """
    ROOT_DIR_OF_POLI_PACKAGE = Path(__file__).parent.parent.parent.parent
    if name == "tdc__isolated":
        environment_file = (
            ROOT_DIR_OF_POLI_PACKAGE / "core" / "chemistry" / "environment.yml"
        )
        isolated_file = (
            ROOT_DIR_OF_POLI_PACKAGE / "core" / "chemistry" / "tdc_isolated_function.py"
        )
        __register_isolated_file(
            environment_file=environment_file,
            isolated_file=isolated_file,
            name_for_show="TDC isolated function",
            quiet=quiet,
        )
    else:
        raise NotImplementedError(
            "The only core isolated function available is the "
            "TDCIsolatedFunction (i.e. tdc__isolated)."
        )


def __run_file_in_env(env_name: str, file_path: Path):
    """
    Runs a file from a given conda env.
    """
    command = " ".join(["conda", "run", "-n", env_name, "python", str(file_path)])
    try:
        subprocess.run(command, check=True, shell=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Found error when running {file_path} from environment {env_name}: \n"
            f"{e.stderr.decode()}"
        ) from e


def __register_isolated_file(
    environment_file: Path,
    isolated_file: Path,
    name_for_show: str = None,
    quiet: bool = False,
):
    """
    Creates the conda environment in the environment file and
    runs the isolated file.
    """
    # 1. Create the conda env.
    __create_conda_env(environment_file, quiet=quiet)
    env_name = __read_env_name(environment_file)

    # 2. Running the file
    if not quiet:
        if name_for_show:
            print(
                f"poli 🧪: running registration of {name_for_show} from environment {env_name}"
            )
        else:
            print(f"poli 🧪: running {isolated_file} from environment {env_name}")

    __run_file_in_env(env_name, isolated_file)


def register_isolated_function(name: str, quiet: bool = False):
    """Registers the objective function if it is available in the repository.

    If the objective function is not available in the repository,
    then we raise an error. If it is available, then we ask the
    user for confirmation to register it. If the user confirms,
    then we register it. Otherwise, we raise an error.

    Parameters
    ----------
    name : str
        The name of the objective function. This corresponds to
        the folder name inside the objective repository. If the
        name contains a `__isolated`, then it is assumed
        that the name refers to an internal file called
        `isolated_function.py`. An exception to this
        is "tdc__isolated", which registers the
        TDCIsolatedFunction.
    quiet : bool, optional
        If True, we squelch the messages giving feedback about the creation process.
        By default, it is False.
    """
    config = load_config()
    if name not in config:
        # Register problem

        # Two cases:
        # (i) some of the isolated functions are not alongside
        # their black boxes and problem factories, but are rather inside
        # the core of poli. For now, the only case is tdc, but more may
        # come in the future.
        #
        # (ii) the isolated function is in the repository, living alongside
        # the black box and the problem factory.
        if name == "tdc__isolated":
            logging.debug(
                "poli 🧪: Registered the isolated function from the repository."
            )
            __register_isolated_function_from_core(name, quiet=quiet)
            config = load_config()
        else:
            logging.debug(
                "poli 🧪: Registered the isolated function from the repository."
            )
            __register_isolated_function_from_repository(name, quiet=quiet)
            # Refresh the config
            config = load_config()


def __create_function_as_isolated_process(
    name: str,
    quiet: bool = False,
    **kwargs_for_isolated_function,
) -> ExternalFunction:
    """Creates the objective function as an isolated process.

    If the problem is registered, we create it as an isolated
    process. Otherwise, we raise an error. That is, this function
    expects the problem to be registered.

    Parameters
    ----------
    name : str
        The name of the objective function.
    seed : int, optional
        The seed value for random number generation.
    quiet : bool, optional
        If True, we squelch the messages giving feedback about the creation process.
        By default, it is False.
    **kwargs_for_factory : dict, optional
        Additional keyword arguments for the factory.
    """
    config = load_config()
    if name not in config:
        raise ValueError(
            f"Objective function '{name.replace('__isolated', '')}' is not registered. "
        )

    if not quiet:
        print(
            f"poli 🧪: Starting the function {name.replace('__isolated', '')} as an isolated process."
        )

    seed = kwargs_for_isolated_function.get("seed", None)
    process_wrapper = ProcessWrapper(
        config[name][_ISOLATED_FUNCTION_SCRIPT_LOCATION], **kwargs_for_isolated_function
    )
    seed = kwargs_for_isolated_function.get("seed", None)
    # TODO: add signal listener that intercepts when proc ends
    # wait for connection from objective process
    # TODO: potential (unlikely) race condition! (process might try to connect before listener is ready!)
    # TODO: We could be sending all the kwargs for the black box here.
    process_wrapper.send(("SETUP", seed, kwargs_for_isolated_function))

    msg_type, *msg = process_wrapper.recv()
    if msg_type == "SETUP":
        # Then the instance of the black box
        # was correctly set-up.
        pass
    elif msg_type == "EXCEPTION":
        e, tb = msg
        print(tb)
        raise e
    else:
        raise ValueError(
            f"Internal error: received {msg_type} when expecting SETUP or EXCEPTION"
        )

    f = ExternalFunction(process_wrapper)

    return f


def instance_function_as_isolated_process(
    name: str,
    quiet: bool = False,
    **kwargs_for_black_box,
) -> ExternalFunction:
    # Check if the user has conda installed
    if shutil.which("conda") is None:
        raise RuntimeError(
            "Conda is not installed/not in the PATH. For poli's isolation mechanisms to work, \n"
            "we need conda to be installed.\n"
            "If you are not interested in using conda, you can install all the \n"
            "relevant dependencies for black boxes using pip and optional arguments.\n"
            "Check the documentation of the black box you are interested in for more information.\n"
            "https://machinelearninglifescience.github.io/poli-docs/."
        )

    # Register the problem if it hasn't been registered.
    register_isolated_function(name=name, quiet=quiet)

    f = __create_function_as_isolated_process(
        name=name,
        quiet=quiet,
        **kwargs_for_black_box,
    )

    return f


def get_inner_function(
    isolated_function_name: str,
    class_name: str,
    module_to_import: str,
    force_isolation: bool = False,
    quiet: bool = False,
    **kwargs,
):
    """Utility for creating an instance of an inner isolated function.

    This function is used in almost every single black box that requires isolation,
    and it abstracts away the logic of trying to import the relevant AbstractIsolatedLogic
    class from the sibling isolated_function.py file of each register.py.

    Parameters
    ----------
    isolated_function_name : str
        The name of the isolated function to be created (e.g. "foldx_stability__isolated").
    class_name : str
        The name of the class to be imported from the isolated function file (e.g. FoldXStabilityIsolatedLogic).
    module_to_import : str
        The full name of the module to import the class from (e.g.
        "poli.objective_repository.foldx_stability.isolated_function").
    force_isolation : bool, optional
        If True, then the function is forced to run in isolation, even if the module can be imported.
    quiet : bool, optional
        If True, we squelch the messages giving feedback about the creation process.
        By default, it is False.
    **kwargs : dict
        Additional keyword arguments for the isolated function.
    """
    if not force_isolation:
        try:
            module = importlib.import_module(module_to_import)
            InnerFunctionClass = getattr(module, class_name)
            inner_function = InnerFunctionClass(**kwargs)
        except ImportError:
            inner_function = instance_function_as_isolated_process(
                name=isolated_function_name, quiet=quiet, **kwargs
            )
    else:
        inner_function = instance_function_as_isolated_process(
            name=isolated_function_name, quiet=quiet, **kwargs
        )
    return inner_function
