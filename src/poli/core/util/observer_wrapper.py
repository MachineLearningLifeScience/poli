"""Script that gets called by the mother process to start an external observer process.
"""

import argparse
import sys
import traceback

from poli.core.util.abstract_observer import AbstractObserver
from poli.core.util.inter_process_communication.process_wrapper import get_connection
from poli.external_isolated_function_script import dynamically_instantiate


def start_observer_process(observer_name, port: int, password: str):
    """
    Starts the observer process.

    Parameters
    ----------
    observer_name : str
        The name of the observer to instantiate.
    port : int
        The port number for the connection with the mother process.
    password : str
        The password for the connection with the mother process.

    Notes
    -----
    This function starts the observer process by establishing a connection with the mother process,
    receiving setup information, instantiating the observer, initializing the observer with the setup
    information, and then waiting for observe calls. If an exception occurs during observation or
    attribute retrieval, it is sent back to the mother process.

    The observer process can be terminated by sending a "QUIT" message.
    """
    # make connection with the mother process
    conn = get_connection(port, password)

    # get setup info from external_observer
    setup_info, caller_info, seed, observer_kwargs = conn.recv()

    # instantiate observer
    observer: AbstractObserver = dynamically_instantiate(
        observer_name, **observer_kwargs
    )

    try:
        observer_info = observer.initialize_observer(setup_info, caller_info, seed)
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
        elif msg_type == "LOG":
            # How should we inform the external observer
            # if something fails during observation?
            # (TODO).
            try:
                observer.log(*msg)
                conn.send(["LOG", None])
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
