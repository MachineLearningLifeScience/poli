from multiprocessing.connection import Client

from poli.core.util.abstract_observer import AbstractObserver
from poli.objective import dynamically_instantiate


def start_observer_process(observer_name):
    # make connection with the mother process
    address = ('localhost', 6001)
    conn = Client(address, authkey=b'secret password')

    # get setup info from external_observer
    setup_info, caller_info, x0, y0 = conn.recv()
    # instantiate observer
    observer: AbstractObserver = dynamically_instantiate(observer_name)
    #try:
    observer_info = observer.initialize_observer(setup_info, caller_info, x0, y0)
    # give mother process the signal that we're ready
    conn.send(observer_info)
    # except Exception as e:
    #     conn.send(e)
    #     return

    # now wait for observe calls
    while True:
        msg = conn.recv()
        if msg is None:
            break
        observer.observe(*msg)
    observer.finish()
    conn.send(None)
    conn.close()
    # TODO: reinsert?
    exit()  # kill other threads, and close file handles
