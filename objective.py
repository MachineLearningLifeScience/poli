import sys
import numpy as np
import signal

from core.AbstractProblemFactory import AbstractProblemFactory
from core.util.abstract_logger import AbstractLogger
from core.util.abstract_observer import AbstractObserver

if __name__ == '__main__':
    # instantiate logger
    logger: AbstractLogger = None  # TODO: read from config
    # instantiate desired observers
    run_info = logger.set_up()  # TODO: use pickle?

    observer: AbstractObserver = None  # TODO: read from config

    # dynamically load objective function module
    objective_name = sys.argv[1]
    eval("from objectives import " + objective_name + " as " + objective_name)
    objective_factory: AbstractProblemFactory = eval(objective_name + '.create()')
    f, x0, y0 = objective_factory.create()

    # add observer
    f.set_observer(observer)

    # tell mother-process that initial data is ready
    np.save(file_name, [x0, y0])
    signal.raise_signal()

    # now wait for objective function calls
    while True:
        signal.sigwait()
        x = np.load(file_name)
        y = f(x)
        np.save(file_name, y)
        signal.raise_signal()
