import sys
from multiprocessing.connection import Client

from poli.core.util.abstract_observer import AbstractObserver
from poli.core.util.ipc import get_connection
from poli.objective import dynamically_instantiate


def start_observer_process(observer_name, port: int, password: str):
    # make connection with the mother process
    conn = get_connection(port, password)

    # get setup info from external_observer
    setup_info, caller_info, x0, y0 = conn.recv()
    # instantiate observer
    observer: AbstractObserver = dynamically_instantiate(observer_name)
    observer_info = observer.initialize_observer(setup_info, caller_info, x0, y0)
    # give mother process the signal that we're ready
    conn.send(observer_info)

    # now wait for observe calls
    while True:
        msg = conn.recv()
        if msg is None:
            break
        observer.observe(*msg)
    observer.finish()
    #conn.close()
    #exit()  # kill other threads, and close file handles


if __name__ == '__main__':
    start_observer_process(sys.argv[1], int(sys.argv[2]), sys.argv[3])
