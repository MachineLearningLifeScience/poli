import sys
from multiprocessing.connection import Client

from poli.core import AbstractProblemFactory
from poli.core.util.abstract_observer import AbstractObserver
from poli.core.util.external_observer import ExternalObserver
from poli.core.registry import config, _DEFAULT, _OBSERVER


def dynamically_instantiate(obj: str):
    last_dot = obj.rfind('.')
    exec("from " + obj[:last_dot] + " import " + obj[last_dot+1:] + " as DynamicObject")
    instantiated_object = eval("DynamicObject()")
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

    # dynamically load objective function module
    objective_factory: AbstractProblemFactory = dynamically_instantiate(objective_name)
    f, x0, y0 = objective_factory.create()

    # instantiate observer
    caller_info = conn.recv()
    observer_script = config[_DEFAULT][_OBSERVER]
    if observer_script != '':
        observer: AbstractObserver = ExternalObserver(observer_script)
        observer_info = observer.initialize_observer(objective_factory.get_setup_information(), caller_info, x0, y0)
        f.set_observer(observer)
    else:
        observer_info = None

    # give mother process the signal that we're ready
    conn.send([x0, y0, objective_factory.get_setup_information(), observer_info])

    # now wait for objective function calls
    while True:
        msg = conn.recv()
        # x, context = msg
        if msg is None:
            break
        y = f(*msg)
        # the observer has been called inside f
        # the main reason is that x can be of shape [N, L] whereas observers are guaranteed objects of shape [1, L]
        conn.send(y)
    if observer_script != '':
        observer.finish()


if __name__ == '__main__':
    run(sys.argv[1])
