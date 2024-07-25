"""
Creates objective functions by providing a common interface to all factories in the repository.
"""

import configparser
import importlib
import logging
from pathlib import Path
from typing import Tuple

import numpy as np

from poli.core import registry
from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.problem import Problem
from poli.core.registry import (
    _DEFAULT,
    _DEFAULT_OBSERVER_RUN_SCRIPT,
    _OBSERVER,
    _RUN_SCRIPT_LOCATION,
    DEFAULT_OBSERVER_NAME,
    register_problem_from_repository,
)
from poli.core.util.abstract_observer import AbstractObserver
from poli.core.util.algorithm_observer_wrapper import AlgorithmObserverWrapper
from poli.core.util.default_observer import DefaultObserver
from poli.core.util.external_observer import ExternalObserver
from poli.core.util.inter_process_communication.process_wrapper import ProcessWrapper
from poli.core.util.isolation.external_black_box import ExternalBlackBox
from poli.external_problem_factory_script import dynamically_instantiate
from poli.objective_repository import AVAILABLE_OBJECTIVES, AVAILABLE_PROBLEM_FACTORIES


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


def __create_problem_from_repository(
    name: str,
    seed: int = None,
    batch_size: int = None,
    parallelize: bool = False,
    num_workers: int = None,
    evaluation_budget: int = float("inf"),
    force_isolation: bool = False,
    observer: AbstractObserver = None,
    **kwargs_for_factory,
) -> Problem:
    """Creates the objective function from the repository.

    We create the problem directly, without starting it as an isolated process.

    Parameters
    ----------
    name : str
        The name of the objective function.
    seed : int, optional
        The seed value for random number generation.
    batch_size : int, optional
        The batch size, passed to the black box to run evaluations on batches.
        If None, it will evaluate all inputs at once.
    parallelize : bool, optional
        If True, then the objective function runs in parallel.
    num_workers : int, optional
        When parallelize is True, this specifies the number of processes to use.
        By default, it uses half the number of available CPUs (rounded down).
    evaluation_budget : int, optional
        The maximum number of evaluations allowed. By default, it is infinity.
    observer : AbstractObserver, optional
        The observer to use.
    **kwargs_for_factory : dict, optional
        Additional keyword arguments for the factory.
    """
    if name not in AVAILABLE_PROBLEM_FACTORIES:
        if name in AVAILABLE_OBJECTIVES:
            # For poli developers:
            # If you get this error while developing a new black box,
            # it might be because you forgot to register your problem
            # factory and black box inside the __init__.py file of
            # the objective_repository. Remember that you need to add
            # them to AVAILABLE_PROBLEM_FACTORIES and AVAILABLE_BLACK_BOXES.
            raise ValueError(
                f"Objective function '{name}' is available in the repository, "
                "but it is not registered as available. This is an internal error. "
                "We encourage you to report it to the developers by creating an "
                "issue in the GitHub repository of poli."
            )
        else:
            raise ValueError(
                f"Objective function '{name}' is not available in the repository."
            )
    problem_factory: AbstractProblemFactory = AVAILABLE_PROBLEM_FACTORIES[name]()
    problem = problem_factory.create(
        seed=seed,
        batch_size=batch_size,
        parallelize=parallelize,
        num_workers=num_workers,
        evaluation_budget=evaluation_budget,
        force_isolation=force_isolation,
        **kwargs_for_factory,
    )

    if observer is not None:
        problem.black_box.set_observer(observer)

    return problem


def __create_problem_as_isolated_process(
    name: str,
    seed: int = None,
    batch_size: int = None,
    parallelize: bool = False,
    num_workers: int = None,
    evaluation_budget: int = float("inf"),
    quiet: bool = False,
    **kwargs_for_factory,
) -> Problem:
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
    batch_size : int, optional
        The batch size, passed to the black box to run evaluations on batches.
        If None, it will evaluate all inputs at once.
    parallelize : bool, optional
        If True, then the objective function runs in parallel.
    num_workers : int, optional
        When parallelize is True, this specifies the number of processes to use.
        By default, it uses half the number of available CPUs (rounded down).
    evaluation_budget : int, optional
        The maximum number of evaluations allowed. By default, it is infinity.
    quiet : bool, optional
        If True, we squelch the messages giving feedback about the creation process.
        By default, it is False.
    **kwargs_for_factory : dict, optional
        Additional keyword arguments for the factory.
    """
    config = load_config()
    if name not in config:
        raise ValueError(f"Objective function '{name}' is not registered. ")

    # start objective process
    # VERY IMPORTANT: the script MUST accept port and password as arguments
    kwargs_for_factory["batch_size"] = batch_size
    kwargs_for_factory["parallelize"] = parallelize
    kwargs_for_factory["num_workers"] = num_workers
    kwargs_for_factory["evaluation_budget"] = evaluation_budget

    if not quiet:
        print(f"poli 🧪: Starting the problem {name} as an isolated objective process.")

    process_wrapper = ProcessWrapper(
        config[name][_RUN_SCRIPT_LOCATION], **kwargs_for_factory
    )
    # TODO: add signal listener that intercepts when proc ends
    # wait for connection from objective process
    # TODO: potential (unlikely) race condition! (process might try to connect before listener is ready!)
    # TODO: We could be sending all the kwargs for the factory here.
    process_wrapper.send(("SETUP", seed))

    msg_type, *msg = process_wrapper.recv()
    if msg_type == "SETUP":
        # Then the instance of the abstract factory
        # was correctly set-up, and
        x0 = msg[0]
    elif msg_type == "EXCEPTION":
        e, tb = msg
        print(tb)
        raise e
    else:
        raise ValueError(
            f"Internal error: received {msg_type} when expecting SETUP or EXCEPTION"
        )

    f = ExternalBlackBox(process_wrapper)
    external_problem = Problem(
        black_box=f,
        x0=x0,
    )

    return external_problem


def __register_objective_if_available(name: str, quiet: bool = False):
    """Registers the objective function if it is available in the repository.

    If the objective function is not available in the repository,
    then we raise an error. If it is available, then we ask the
    user for confirmation to register it. If the user confirms,
    then we register it. Otherwise, we raise an error.

    Parameters
    ----------
    name : str
        The name of the objective function.
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
        if name not in AVAILABLE_OBJECTIVES:
            raise ValueError(
                f"Objective function '{name}' is not registered, "
                "and it is not available in the repository."
            )

        # Register problem
        register_problem_from_repository(name, quiet=quiet)
        logging.debug(f"poli 🧪: Registered the objective from the repository.")

        # Refresh the config
        config = load_config()


def create(
    name: str,
    *,
    seed: int = None,
    observer_init_info: dict = None,
    observer_name: str = None,
    force_register: bool = True,
    force_isolation: bool = False,
    batch_size: int = None,
    parallelize: bool = False,
    num_workers: int = None,
    evaluation_budget: int = float("inf"),
    quiet: bool = False,
    **kwargs_for_factory,
) -> Problem:
    """
    Instantiantes a black-box function by calling the `create` method of the associated factory.

    Parameters
    ----------
    name : str
        The name of the objective function.
    seed : int, optional
        The seed value for random number generation.
    observer_init_info : dict, optional
        Optional information about the caller that is forwarded to the logger to initialize the run.
    observer_name : str, optional
        The observer to use.
    force_register : bool, deprecated
        Force the registration of the objective function. This is deprecated and will be removed in the future.
        As it stands, this kwarg is not used.
    force_isolation : bool, optional
        If True, then the objective function is instantiated as an isolated
        process.
    batch_size : int, optional
        The batch size, passed to the black box to run evaluations on batches.
        If None, it will evaluate all inputs at once.
    parallelize : bool, optional
        If True, then the objective function runs in parallel.
    num_workers : int, optional
        When parallelize is True, this specifies the number of processes to use.
        By default, it uses half the number of available CPUs (rounded down).
    evaluation_budget : int, optional
        The maximum number of evaluations allowed. By default, it is infinity.
    quiet : bool, optional
        If True, we squelch the messages giving feedback about the creation process.
        By default, it is False.
    **kwargs_for_factory : dict, optional
        Additional keyword arguments for the factory.

    Returns
    -------
    problem : AbstractProblem
        The black-box function, initial value, and related information.
    """
    # If the user can run it with the envionment they currently
    # have, then we do not need to install it.
    # if not force_isolation:
    #     if not quiet:
    #         print(f"poli 🧪: Creating the objective {name} from the repository.")

    #     problem = __create_problem_from_repository(
    #         name,
    #         seed=seed,
    #         batch_size=batch_size,
    #         parallelize=parallelize,
    #         num_workers=num_workers,
    #         evaluation_budget=evaluation_budget,
    #         **kwargs_for_factory,
    #     )
    # else:
    #     # Check if the name is indeed registered, or
    #     # available in the objective repository
    #     # This function will
    #     # 1. Create the conda environment for the objective
    #     # 2. Run registration inside said environment.

    #     # Assert that the problem has an isolated_function.py

    #     # Register the isolated function.

    #     # Create the problem as usual. The isolated function
    #     # will run just fine.
    #     __register_objective_if_available(name, quiet=quiet)

    #     # At this point, we know the name is registered.
    #     # Thus, we should be able to start it as an isolated process
    #     if not quiet:
    #         print(f"poli 🧪: Creating an isolated problem ({name}).")
    #     problem = __create_problem_as_isolated_process(
    #         name,
    #         seed=seed,
    #         batch_size=batch_size,
    #         parallelize=parallelize,
    #         num_workers=num_workers,
    #         evaluation_budget=evaluation_budget,
    #         quiet=quiet,
    #         **kwargs_for_factory,
    #     )
    problem = __create_problem_from_repository(
        name,
        seed=seed,
        batch_size=batch_size,
        parallelize=parallelize,
        num_workers=num_workers,
        evaluation_budget=evaluation_budget,
        force_isolation=force_isolation,
        **kwargs_for_factory,
    )

    # instantiate observer (if desired)
    observer = _instantiate_observer(observer_name, quiet)
    observer_info = observer.initialize_observer(
        problem.black_box.info, observer_init_info, seed
    )

    # TODO: Should we send the y0 to the observer initialization?
    # f, x0 = problem.black_box, problem.x0
    # y0 = f(x0)
    f = problem.black_box
    f.set_observer(observer)
    f.set_observer_info(observer_info)
    problem.set_observer(AlgorithmObserverWrapper(observer), observer_info)

    return problem


def start(
    name: str,
    seed: int = None,
    caller_info: dict = None,
    observer_name: str = None,
    force_register: bool = False,
    force_isolation: bool = False,
    **kwargs_for_factory,
) -> AbstractBlackBox:
    """Starts the black-box function.

    Works just like create, but it returns only the black-box function
    and resets the evaluation budget.

    One example of running this function:

    ```python
    from poli import objective_factory

    with objective_factory.start(name="aloha") as f:
        x = np.array(["F", "L", "E", "A", "S"]).reshape(1, -1)
        y = f(x)
        print(x, y)
    ```

    If the black-box function is created as an isolated process, then
    it will automatically close when the context manager is closed.

    Parameters
    ----------
    name : str
        The name of the objective function.
    seed : int, optional
        The seed value for random number generation.
    caller_info : dict, optional
        Optional information about the caller that is forwarded to the logger to initialize the run.
    observer_name : str, optional
        The observer to use.
    force_register : bool, optional
        If True, then the objective function is registered without asking
        for confirmation, overwriting any previous registration.
    force_isolation : bool, optional
        If True, then the objective function is instantiated as an isolated
        process.
    **kwargs_for_factory : dict, optional
        Additional keyword arguments for the factory.
    """
    problem = create(
        name=name,
        seed=seed,
        observer_init_info=caller_info,
        observer_name=observer_name,
        force_register=force_register,
        force_isolation=force_isolation,
        **kwargs_for_factory,
    )

    f = problem.black_box
    # Reset the counter of evaluations
    f.reset_evaluation_budget()

    return f


def _instantiate_observer(observer_name: str, quiet: bool = False) -> AbstractObserver:
    """
    This function attempts to locally instantiate an observer and if that fails starts the observer in the dedicated environment.

    Parameters
    ----------
    observer_name : str
        The observer to use.
    quiet : bool, optional
        If True, we squelch the messages giving feedback about the creation process.

    Returns
    -------
    observer : AbstractObserver
        The observer, either dynamically instantiated or started as an isolated process.
    """
    if _OBSERVER not in registry.config[_DEFAULT]:
        registry.config[_DEFAULT][_OBSERVER] = _DEFAULT_OBSERVER_RUN_SCRIPT

    observer_script: str = registry.config[_DEFAULT][_OBSERVER]
    if observer_name is not None:
        if observer_name != DEFAULT_OBSERVER_NAME:
            observer_script = registry.config[_OBSERVER][observer_name]
        else:
            observer_script = _DEFAULT_OBSERVER_RUN_SCRIPT

    if observer_script == _DEFAULT_OBSERVER_RUN_SCRIPT:
        observer = DefaultObserver()
    else:
        if not quiet:
            print(f"poli 🧪: initializing the observer.")
        try:
            f = open(observer_script, "r")
            observer_class = (
                f.readlines()[-1].split("--objective-name=")[1].split(" --port")[0]
            )
            f.close()
            observer = dynamically_instantiate(observer_class)
        except:
            if not quiet:
                print(f"poli 🧪: attempting isolated observer instantiation.")
            observer = ExternalObserver(observer_name=observer_name)
    return observer
