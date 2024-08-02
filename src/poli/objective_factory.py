"""
Creates objective functions by providing a common interface to all factories in the repository.
"""

import configparser
from pathlib import Path

from poli.core import registry
from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.problem import Problem
from poli.core.registry import (
    _DEFAULT,
    _DEFAULT_OBSERVER_RUN_SCRIPT,
    _OBSERVER,
    DEFAULT_OBSERVER_NAME,
)
from poli.core.util.abstract_observer import AbstractObserver
from poli.core.util.algorithm_observer_wrapper import AlgorithmObserverWrapper
from poli.core.util.default_observer import DefaultObserver
from poli.core.util.external_observer import ExternalObserver
from poli.external_isolated_function_script import dynamically_instantiate
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


def create(
    name: str,
    *,
    seed: int = None,
    observer_init_info: dict = None,
    observer_name: str = None,
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
            print("poli 🧪: initializing the observer.")
        try:
            f = open(observer_script, "r")
            observer_class = (
                f.readlines()[-1].split("--objective-name=")[1].split(" --port")[0]
            )
            f.close()
            observer = dynamically_instantiate(observer_class)
        except Exception:
            if not quiet:
                print("poli 🧪: attempting isolated observer instantiation.")
            observer = ExternalObserver(observer_name=observer_name)
    return observer
