"""
This is the main module relevant for user-defined observers.
When registering an observer, this module is called and instantiates the user's observer.
"""

import sys
import argparse

from poli.core.util.abstract_observer import AbstractObserver
from poli.core.util.inter_process_communication.process_wrapper import get_connection
from poli.objective import dynamically_instantiate


def start_observer_process(observer_name, port: int, password: str):
    # make connection with the mother process
    conn = get_connection(port, password)

    # get setup info from external_observer
    setup_info, caller_info, x0, y0, seed = conn.recv()
    # instantiate observer
    observer: AbstractObserver = dynamically_instantiate(observer_name)
    observer_info = observer.initialize_observer(setup_info, caller_info, x0, y0, seed)
    # give mother process the signal that we're ready
    conn.send(observer_info)

    # now wait for observe calls
    while True:
        msg = conn.recv()
        if msg is None:
            break
        observer.observe(*msg)
    observer.finish()
    # conn.close()
    # exit()  # kill other threads, and close file handles


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective-name", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--password", required=True, type=str)

    args, factory_kwargs = parser.parse_known_args()
    start_observer_process(args.objective_name, args.port, args.password)
