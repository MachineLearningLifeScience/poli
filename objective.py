import sys
import numpy as np
import signal
import configparser

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
    signal.raise_signal(signal.SIGUSR1)
    try:
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

        # TODO: read caller_info
        caller_info = {}
        run_info = logger.initialize_logger(objective_factory.get_setup_information(), caller_info)

        f, x0, y0 = objective_factory.create()
        # add observer
        f.set_observer(observer)

        # tell mother-process that initial data is ready
        np.save(INIT_DATA_FILE, np.hstack([x0, y0]))
        signal.raise_signal(signal.SIG_UNBLOCK)

        # now wait for objective function calls
        while True:
            sig = signal.sigwait([signal.SIG_BLOCK, signal.SIGQUIT])
            if sig == signal.SIGQUIT:
                break
            x = np.load(INPUT_DATA_FILE)
            y = f(x)
            np.save(OUTPUT_DATA_FILE, y)
            signal.raise_signal(signal.SIG_UNBLOCK)
    except Exception as e:
        signal.raise_signal(signal.SIGQUIT)
        raise e
