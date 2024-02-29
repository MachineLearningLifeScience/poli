"""
Creates objective functions by providing a common interface to all factories in the repository.
"""

from typing import Tuple, Any
import numpy as np
from pathlib import Path
import configparser
import logging

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.problem_setup_information import ProblemSetupInformation
from poli.core.registry import (
    _RUN_SCRIPT_LOCATION,
    _OBSERVER,
    register_problem_from_repository,
)
from poli.core.util.abstract_observer import AbstractObserver
from poli.core.util.inter_process_communication.process_wrapper import ProcessWrapper
from poli.core.util.isolation.external_black_box import ExternalBlackBox
from poli.core.abstract_problem import AbstractProblem

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
    observer: AbstractObserver = None,
    **kwargs_for_factory,
) -> Tuple[AbstractBlackBox, np.ndarray, np.ndarray]:
    """Creates the objective function from the repository.

    If the problem is available in AVAILABLE_PROBLEM_FACTORIES
    (i.e. if the user could import all the dependencies directly),
    we create the problem directly. Otherwise, we raise an error.

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
) -> Tuple[AbstractBlackBox, np.ndarray, np.ndarray]:
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
        print(f"poli 🧪: starting the isolated objective process.")

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
        x0, y0, problem_information = msg
    elif msg_type == "EXCEPTION":
        e, tb = msg
        print(tb)
        raise e
    else:
        raise ValueError(
            f"Internal error: received {msg_type} when expecting SETUP or EXCEPTION"
        )

    f = ExternalBlackBox(problem_information, process_wrapper)

    return f, x0, y0


def __register_objective_if_available(
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

        if answer == "y":
            # Register problem
            logging.debug(f"poli 🧪: Registered the objective from the repository.")
            register_problem_from_repository(name, quiet=quiet)
            # Refresh the config
            config = load_config()
        else:
            raise ValueError(
                f"Objective function '{name}' won't be registered. Aborting."
            )


def create(
    name: str,
    *,
    seed: int = None,
    observer_init_info: dict = None,
    observer: AbstractObserver = None,
    force_register: bool = True,
    force_isolation: bool = False,
    batch_size: int = None,
    parallelize: bool = False,
    num_workers: int = None,
    evaluation_budget: int = float("inf"),
    quiet: bool = False,
    **kwargs_for_factory,
) -> AbstractProblem:
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
    observer : AbstractObserver, optional
        The observer to use.
    force_register : bool, optional
        If True, then the objective function is registered without asking
        for confirmation, overwriting any previous registration. By default,
        it is True.
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
    if name in AVAILABLE_PROBLEM_FACTORIES and not force_isolation:
        if not quiet:
            print(f"poli 🧪: Creating the objective from the repository.")

        problem = __create_problem_from_repository(
            name,
            seed=seed,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
            **kwargs_for_factory,
        )
        x0 = problem.x0
        y0 = problem.black_box(x0)

        if observer is not None:
            if not quiet:
                print(f"poli 🧪: initializing the observer.")
            observer_info = observer.initialize_observer(
                problem.black_box.info, observer_init_info, x0, y0, seed
            )
            problem.black_box.set_observer(observer)
            problem.black_box.set_observer_info(observer_info)

        return problem

    # Check if the name is indeed registered, or
    # available in the objective repository
    __register_objective_if_available(name, force_register=force_register, quiet=quiet)

    # At this point, we know the name is registered.
    # Thus, we should be able to start it as an isolated process
    if not quiet:
        print(f"poli 🧪: creating an isolated black box function.")
    problem = __create_problem_as_isolated_process(
        name,
        seed=seed,
        batch_size=batch_size,
        parallelize=parallelize,
        num_workers=num_workers,
        evaluation_budget=evaluation_budget,
        quiet=quiet,
        **kwargs_for_factory,
    )
    black_box_information = problem.black_box.info

    # instantiate observer (if desired)
    observer_info = None
    if observer is not None:
        f, x0 = problem.black_box, problem.x0
        y0 = f(x0)

        observer_info = observer.initialize_observer(
            black_box_information, observer_init_info, x0, y0, seed
        )

        f.set_observer(observer)
        f.set_observer_info(observer_info)

    return problem


def start(
    name: str,
    seed: int = None,
    caller_info: dict = None,
    observer: AbstractObserver = None,
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
    observer : AbstractObserver, optional
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
    # Check if we can import the function immediately
    if name in AVAILABLE_PROBLEM_FACTORIES and not force_isolation:
        f, _, _ = __create_problem_from_repository(
            name, seed=seed, **kwargs_for_factory
        )
    else:
        # Check if the name is indeed registered, or
        # available in the objective repository
        __register_objective_if_available(name, force_register=force_register)

        # If not, then we create it as an isolated process
        f, _, _ = __create_problem_as_isolated_process(
            name, seed=seed, **kwargs_for_factory
        )

    # instantiate observer (if desired)
    if observer is not None:
        observer.initialize_observer(f.info, caller_info, None, None, seed)

    f.set_observer(observer)

    # Reset the counter of evaluations
    f.reset_evaluation_budget()

    return f
