import logging
import os
import sys
import click

from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.util.inter_process_communication.process_wrapper import get_connection


ADDITIONAL_IMPORT_SEARCH_PATHES_KEY = "ADDITIONAL_IMPORT_PATHS"


def dynamically_instantiate(obj: str, **kwargs):
    # FIXME: this method opens up a serious security vulnerability
    # TODO: possible alternative: importlib
    # TODO: another possible alternative: hydra
    # sys.path.append(os.getcwd())
    sys.path.extend(os.environ[ADDITIONAL_IMPORT_SEARCH_PATHES_KEY].split(":"))
    # sys.path.extend(os.environ['PYTHONPATH'].split(':'))
    last_dot = obj.rfind(".")
    try:
        exec(
            "from "
            + obj[:last_dot]
            + " import "
            + obj[last_dot + 1 :]
            + " as DynamicObject"
        )
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


@click.command(
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    }
)
@click.option("--objective-name", required=True)
@click.option("--port", required=True, type=int)
@click.option("--password", required=True, type=str)
@click.pass_context
def run(context, objective_name: str, port: int, password: str) -> None:
    """
    Starts an objective function listener loop to wait for requests.
    :param objective_name:
        problem factory name including python packages, e.g. package.subpackage.MyFactoryName
    """
    kwargs = dict([item.strip("--").split("=") for item in context.args])
    # make connection with the mother process
    conn = get_connection(port, password)
    seed = conn.recv()

    # dynamically load objective function module
    objective_factory: AbstractProblemFactory = dynamically_instantiate(
        objective_name, **kwargs
    )
    f, x0, y0 = objective_factory.create(seed)

    # give mother process the signal that we're ready
    conn.send([x0, y0, objective_factory.get_setup_information()])

    # now wait for objective function calls
    while True:
        msg = conn.recv()
        # x, context = msg
        if msg is None:
            break
        y = f(*msg)
        conn.send(y)
    # conn.close()
    # exit()  # kill other threads, and close file handles


if __name__ == "__main__":
    # TODO: modify this to allow for passing more
    # information to run. Said information can be interpreted
    # as being used for instantiating the problem factory.
    run()
    # if len(sys.argv) == 4:
    #     run(sys.argv[1], int(sys.argv[2]), sys.argv[3])
    # elif len(sys.argv) > 4:
    #     run(sys.argv[1], int(sys.argv[2]), sys.argv[3], *sys.argv[4:])
    # else:
    #     raise ValueError(
    #         "Not enough arguments. Expected: objective_name, port, password"
    #     )
