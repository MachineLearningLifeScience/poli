"""Executable script used for isolation of objective factories and functions.

The equivalent of objective, but for isolated black boxes instead of problem factories.
"""

import argparse
import logging
import os
import sys
import traceback

from poli.core.abstract_isolated_function import AbstractIsolatedFunction
from poli.core.util.inter_process_communication.process_wrapper import get_connection
from poli.core.util.seeding import seed_python_numpy_and_torch

ADDITIONAL_IMPORT_SEARCH_PATHES_KEY = "ADDITIONAL_IMPORT_PATHS"


def dynamically_instantiate(obj: str, **kwargs):
    """Dynamically instantiates an object from a string.

    This function is used internally to instantiate objective
    factories dynamically, inside isolated processes. It is
    also used to instantiate external observers.

    Parameters
    ----------
    obj : str
        The string containing the name of the object to be instantiated.
    **kwargs : dict
        The keyword arguments to be passed to the object constructor.
    """
    # FIXME: this method opens up a serious security vulnerability
    # TODO: possible alternative: importlib
    # TODO: another possible alternative: hydra
    # sys.path.append(os.getcwd())
    sys.path.extend(os.environ[ADDITIONAL_IMPORT_SEARCH_PATHES_KEY].split(":"))
    # sys.path.extend(os.environ['PYTHONPATH'].split(':'))
    last_dot = obj.rfind(".")

    command = (
        "from "
        + obj[:last_dot]
        + " import "
        + obj[last_dot + 1 :]
        + " as DynamicObject"
    )
    try:
        exec(command)
        instantiated_object = eval("DynamicObject")(**kwargs)
    except ImportError as e:
        logging.fatal(f"Path: {os.environ['PATH']}")
        logging.fatal(f"Python path: {sys.path}")
        logging.fatal(f"Path: {os.environ[ADDITIONAL_IMPORT_SEARCH_PATHES_KEY]}")
        if "PYTHONPATH" in os.environ.keys():
            logging.fatal(f"Path: {os.environ['PYTHONPATH']}")
        else:
            logging.fatal("PYTHONPATH is not part of the environment variables.")
        raise e
    return instantiated_object


def run(objective_name: str, port: int, password: str) -> None:
    """Starts an objective function listener loop to wait for requests.

    Parameters
    ----------
    factory_kwargs : str
        The string containing the factory kwargs (see ProcessWrapper
        for details about how this factory_kwargs strings is built).
    objective_name : str
        The name of the objective function to be instantiated.
    port : int
        The port number for the connection with the mother process.
    password : str
        The password for the connection with the mother process.
    """
    # kwargs = parse_factory_kwargs(factory_kwargs)

    # make connection with the mother process
    conn = get_connection(port, password)

    # TODO: We could be receiving the kwargs for the factory here.
    msg_type, seed, kwargs_for_function = conn.recv()

    if seed is not None:
        seed_python_numpy_and_torch(seed)

    # dynamically load objective function module
    # At this point, the black box objective function
    # is exactly the same as the one used in the
    # registration (?).
    try:
        f: AbstractIsolatedFunction = dynamically_instantiate(
            objective_name, **kwargs_for_function
        )

        # give mother process the signal that we're ready
        conn.send(["SETUP", None])
    except Exception as e:
        tb = traceback.format_exc()
        conn.send(["EXCEPTION", e, tb])
        raise e

    # now wait for objective function calls
    while True:
        msg_type, *msg = conn.recv()

        if msg_type == "QUIT":
            break
        try:
            if msg_type == "QUERY":
                x, context = msg
                y = f(x, context=context)
                conn.send(["QUERY", y])
            elif msg_type == "ATTRIBUTE":
                attribute_name = msg[0]
                attribute = getattr(f, attribute_name)
                conn.send(["ATTRIBUTE", attribute])
            elif msg_type == "IS_METHOD":
                attribute_name = msg[0]
                is_method = callable(getattr(f, attribute_name))
                conn.send(["IS_METHOD", is_method])
            elif msg_type == "METHOD":
                method_name = msg[0]
                method_args = msg[1]
                method_kwargs = msg[2]
                method = getattr(f, method_name)
                result = method(*method_args, **method_kwargs)
                conn.send(["METHOD", result])
        except Exception as e:
            tb = traceback.format_exc()
            conn.send(["EXCEPTION", e, tb])

    # conn.close()
    # exit()  # kill other threads, and close file handles


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective-name", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--password", required=True, type=str)

    args, factory_kwargs = parser.parse_known_args()
    run(args.objective_name, args.port, args.password)
