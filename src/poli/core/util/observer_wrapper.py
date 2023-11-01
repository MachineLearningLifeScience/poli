"""
This is the main module relevant for user-defined observers.
When registering an observer, this module is called and instantiates the user's observer.
"""

import sys
import argparse
import traceback

from poli.core.util.abstract_observer import AbstractObserver
from poli.core.util.inter_process_communication.process_wrapper import get_connection
from poli.objective import dynamically_instantiate, parse_factory_kwargs


def start_observer_process(observer_name, port: int, password: str):
    # make connection with the mother process
    conn = get_connection(port, password)

    # get setup info from external_observer
    setup_info, caller_info, x0, y0, seed, observer_kwargs = conn.recv()

    # instantiate observer
    observer: AbstractObserver = dynamically_instantiate(
        observer_name, **observer_kwargs
    )

    try:
        observer_info = observer.initialize_observer(
            setup_info, caller_info, x0, y0, seed
        )
        # give mother process the signal that we're ready
        conn.send(["SETUP", observer_info])
    except Exception as e:
        tb = traceback.format_exc()
        conn.send(["EXCEPTION", e, tb])
        sys.exit(1)

    # now wait for observe calls
    while True:
        msg_type, *msg = conn.recv()
        if msg_type == "OBSERVATION":
            # How should we inform the external observer
            # if something fails during observation?
            # (TODO).
            try:
                observer.observe(*msg)
                conn.send(["OBSERVATION", None])
            except Exception as e:
                tb = traceback.format_exc()
                conn.send(["EXCEPTION", e, tb])
        elif msg_type == "ATTRIBUTE":
            try:
                conn.send(["ATTRIBUTE", getattr(observer, msg[0])])
            except Exception as e:
                tb = traceback.format_exc()
                conn.send(["EXCEPTION", e, tb])
        elif msg_type == "QUIT":
            break
    observer.finish()
    # conn.close()
    # exit()  # kill other threads, and close file handles


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective-name", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--password", required=True, type=str)

    args, _ = parser.parse_known_args()
    start_observer_process(args.objective_name, args.port, args.password)
