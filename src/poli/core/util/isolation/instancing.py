from pathlib import Path
import configparser
import subprocess
import warnings

import logging
from poli.core.registry import (
    _DEFAULT,
    _OBSERVER,
    _RUN_SCRIPT_LOCATION,
    _ISOLATED_FUNCTION_SCRIPT_LOCATION,
)

# from poli.objective_repository import AVAILABLE_OBJECTIVES
from poli.core.util.inter_process_communication.process_wrapper import ProcessWrapper

from .external_black_box import ExternalBlackBox
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


def __register_isolated_function_from_repository(name: str, quiet: bool = False):
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
    assert name.endswith(
        "__isolated"
    ), "By convention, the names of isolated functions always end with '__isolated'"

    # Load up the environment name
    PATH_TO_REPOSITORY = (
        Path(__file__).parent.parent.parent.parent / "objective_repository"
    ).resolve()

    name_without_isolated = name.replace("__isolated", "")
    environment_file = PATH_TO_REPOSITORY / name_without_isolated / "environment.yml"
    isolated_file = PATH_TO_REPOSITORY / name_without_isolated / "isolated_function.py"

    __register_isolated_function(
        environment_file=environment_file,
        isolated_file=isolated_file,
        name=name,
        quiet=quiet,
    )


def __register_isolated_function_from_core(name: str, quiet: bool = False):
    ROOT_DIR_OF_POLI_PACKAGE = Path(__file__).parent.parent.parent.parent
    if name == "tdc__isolated":
        environment_file = (
            ROOT_DIR_OF_POLI_PACKAGE / "core" / "chemistry" / "environment.yml"
        )
        isolated_file = (
            ROOT_DIR_OF_POLI_PACKAGE / "core" / "chemistry" / "tdc_isolated_function.py"
        )
        __register_isolated_function(
            environment_file=environment_file,
            isolated_file=isolated_file,
            name="TDC isolated function",
            quiet=quiet,
        )
    else:
        raise NotImplementedError(
            "The only core isolated function available is the "
            "TDCIsolatedFunction (i.e. tdc__isolated)."
        )

        ...


def __register_isolated_function(
    environment_file: Path,
    isolated_file: Path,
    name: str = None,
    quiet: bool = False,
):
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

    # 2. run the file from said enviroment (since
    #    we can't import the factory: it may have
    #    dependencies that are not installed)

    # Running the file
    command = " ".join(["conda", "run", "-n", env_name, "python", str(isolated_file)])
    # warnings.warn("Running the following command: %s. " % command)

    if not quiet:
        if name:
            print(
                f"poli 🧪: running registration of {name} from environment {env_name}"
            )
        else:
            print(f"poli 🧪: running {isolated_file} from environment {env_name}")
    try:
        subprocess.run(command, check=True, shell=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Found error when running {isolated_file} from environment {env_name}: \n"
            f"{e.stderr.decode()}"
        ) from e


def register_isolated_function_if_available(
    name: str, force_register: bool = True, quiet: bool = False
):
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
    force_register : bool, optional
        If True, then the objective function is registered without asking
        for confirmation, overwriting any previous registration. By default,
        it is True.
    quiet : bool, optional
        If True, we squelch the messages giving feedback about the creation process.
        By default, it is False.
    """
    config = load_config()
    if name not in config:
        # if name not in AVAILABLE_OBJECTIVES:
        #     raise ValueError(
        #         f"Objective function '{name}' is not registered, "
        #         "and it is not available in the repository."
        #     )

        # At this point, we know that the function is available
        # in the repository
        if force_register:
            # Then we install it.
            answer = "y"
        else:
            # We ask the user for their confirmation
            answer = input(
                f"Objective function '{name}' is not registered, "
                "but it is available in the repository. Do you "
                "want to install it? (y/[n]): "
            )

        if answer != "y":
            raise ValueError(
                f"Objective function '{name}' is not registered. Aborting."
            )

        # Register problem
        if name == "tdc__isolated":
            logging.debug(
                f"poli 🧪: Registered the isolated function from the repository."
            )
            __register_isolated_function_from_core(name, quiet=quiet)
            config = load_config()
        else:
            logging.debug(
                f"poli 🧪: Registered the isolated function from the repository."
            )
            __register_isolated_function_from_repository(name, quiet=quiet)
            # Refresh the config
            config = load_config()


def __create_function_as_isolated_process(
    name: str,
    seed: int = None,
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

    process_wrapper = ProcessWrapper(
        config[name][_ISOLATED_FUNCTION_SCRIPT_LOCATION], **kwargs_for_isolated_function
    )
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
    seed: int = None,
    quiet: bool = False,
    force_register: bool = True,
    **kwargs_for_black_box,
) -> ExternalFunction:
    # Register the problem
    register_isolated_function_if_available(
        name=name, force_register=force_register, quiet=quiet
    )

    # Create the external process wrapper
    f = __create_function_as_isolated_process(
        name=name,
        seed=seed,
        quiet=quiet,
        **kwargs_for_black_box,
    )

    # return it.
    return f
