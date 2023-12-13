import logging
import os
import sys
import argparse
import traceback

from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.util.inter_process_communication.process_wrapper import get_connection


ADDITIONAL_IMPORT_SEARCH_PATHES_KEY = "ADDITIONAL_IMPORT_PATHS"


def parse_factory_kwargs(factory_kwargs: str) -> dict:
    if factory_kwargs == "":
        # Then the user didn't pass any arguments
        kwargs = {}
    else:
        factory_kwargs = factory_kwargs.split()
        kwargs = {}
        for item in factory_kwargs:
            item = item.strip("--")
            key, value = item.split("=")
            if value.startswith("list:"):
                # Then we assume that the value was a list
                value = value.strip("list:")
                value = value.split(",")
            elif value.startswith("int:"):
                value = int(value.strip("int:"))
            elif value.startswith("float:"):
                if value == "float:inf":
                    value = float("inf")
                elif value == "float:-inf":
                    value = float("-inf")
                else:
                    value = float(value.strip("float:"))
            elif value.startswith("bool:"):
                value = value.strip("bool:") == "True"
            elif value.startswith("none:"):
                value = None

            kwargs[key] = value

    return kwargs


def dynamically_instantiate(obj: str, **kwargs):
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


def run(factory_kwargs: str, objective_name: str, port: int, password: str) -> None:
    """
    Starts an objective function listener loop to wait for requests.
    :param objective_name:
        problem factory name including python packages, e.g. package.subpackage.MyFactoryName
    """
    kwargs = parse_factory_kwargs(factory_kwargs)

    # make connection with the mother process
    conn = get_connection(port, password)

    # TODO: We could be receiving the kwargs for the factory here.
    msg_type, seed = conn.recv()

    # dynamically load objective function module
    # At this point, the black box objective function
    # is exactly the same as the one used in the
    # registration (?).
    try:
        objective_factory: AbstractProblemFactory = dynamically_instantiate(
            objective_name
        )
        f, x0, y0 = objective_factory.create(seed, **kwargs)

        # give mother process the signal that we're ready
        conn.send(["SETUP", x0, y0, objective_factory.get_setup_information()])
    except Exception as e:
        tb = traceback.format_exc()
        conn.send(["EXCEPTION", e, tb])
        raise e

    # now wait for objective function calls
    while True:
        msg_type, *msg = conn.recv()
        # x, context = msg
        if msg_type == "QUIT":
            break
        try:
            if msg_type == "QUERY":
                x, context = msg
                y = f(x, context=context)
                conn.send(["QUERY", y])
            elif msg_type == "ATTRIBUTE":
                attribute = getattr(f, msg[0])
                conn.send(["ATTRIBUTE", attribute])
        except Exception as e:
            tb = traceback.format_exc()
            conn.send(["EXCEPTION", e, tb])

    # conn.close()
    # exit()  # kill other threads, and close file handles


if __name__ == "__main__":
    # TODO: modify this to allow for passing more
    # information to run. Said information can be interpreted
    # as being used for instantiating the problem factory.
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective-name", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--password", required=True, type=str)

    args, factory_kwargs = parser.parse_known_args()
    run(factory_kwargs[0], args.objective_name, args.port, args.password)
