import logging
import os
import sys
from multiprocessing.connection import Client

from poli.core.abstract_problem_factory import AbstractProblemFactory


def dynamically_instantiate(obj: str):
    sys.path.append(os.getcwd())
    last_dot = obj.rfind('.')
    try:
        exec("from " + obj[:last_dot] + " import " + obj[last_dot+1:] + " as DynamicObject")
        instantiated_object = eval("DynamicObject()")
    except ModuleNotFoundError as e:
        logging.fatal(f"Python path: {sys.path}")
        raise e
    return instantiated_object


def run(objective_name: str) -> None:
    """
    Starts an objective function listener loop to wait for requests.
    :param objective_name:
        problem factory name including python packages, e.g. package.subpackage.MyFactoryName
    """
    # make connection with the mother process
    address = ('localhost', 6000)
    conn = Client(address, authkey=b'secret password')
    seed = conn.recv()

    # dynamically load objective function module
    objective_factory: AbstractProblemFactory = dynamically_instantiate(objective_name)
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
    #conn.close()
    #exit()  # kill other threads, and close file handles


if __name__ == '__main__':
    run(sys.argv[1])
