import sys
import numpy as np
import signal
import configparser
from multiprocessing.connection import Client

from core.AbstractProblemFactory import AbstractProblemFactory
from core.registry import INIT_DATA_FILE, INPUT_DATA_FILE, OUTPUT_DATA_FILE
from core.util.abstract_logger import AbstractLogger
from core.util.abstract_observer import AbstractObserver


def dynamically_instantiate(obj: str):
    last_dot = obj.rfind('.')
    exec("from " + obj[:last_dot] + " import " + obj[last_dot+1:] + " as DynamicObject")
    instantiated_object = eval("DynamicObject()")
    return instantiated_object


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.rc')
    # instantiate logger
    logger: AbstractLogger = dynamically_instantiate(config['DEFAULT']['logger'])

    # instantiate observer
    observer: AbstractObserver = dynamically_instantiate(config['DEFAULT']['observer'])
    observer.initialize(logger)

    # dynamically load objective function module
    objective_name = sys.argv[1]
    objective_factory: AbstractProblemFactory = dynamically_instantiate(objective_name)

    address = ('localhost', 6000)
    conn = Client(address, authkey=b'secret password')
    caller_info = conn.recv()
    run_info = logger.initialize_logger(objective_factory.get_setup_information(), caller_info)

    f, x0, y0 = objective_factory.create()
    # add observer
    f.set_observer(observer)

    conn.send([x0, y0, run_info])

    # now wait for objective function calls
    while not conn.closed:
        x = conn.recv()
        y = f(x)
        conn.send(y)
