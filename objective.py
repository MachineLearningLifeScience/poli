import sys
import configparser
from multiprocessing.connection import Client

from core.AbstractProblemFactory import AbstractProblemFactory
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

    # make connection with the mother process
    address = ('localhost', 6000)
    conn = Client(address, authkey=b'secret password')

    # dynamically load objective function module
    objective_name = sys.argv[1]
    objective_factory: AbstractProblemFactory = dynamically_instantiate(objective_name)
    f, x0, y0 = objective_factory.create()

    # instantiate logger
    logger: AbstractLogger = dynamically_instantiate(config['DEFAULT']['logger'])
    caller_info = conn.recv()
    run_info = logger.initialize_logger(objective_factory.get_setup_information(), caller_info)

    # give mother process the signal that we're ready
    conn.send([x0, y0, run_info])

    # instantiate observer
    observer: AbstractObserver = dynamically_instantiate(config['DEFAULT']['observer'])
    observer.initialize(logger)
    f.set_observer(observer)

    # now wait for objective function calls
    while True:
        x = conn.recv()
        if x is None:
            break
        y = f(x)
        conn.send(y)
    logger.finish()
