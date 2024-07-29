"""
Module that wraps utility functions for interprocess communication.
"""

import logging
import subprocess
import time
from multiprocessing.connection import Client, Listener
from pathlib import Path
from uuid import uuid4


def get_connection(port: int, password: str) -> Client:
    """
    Get a connection to a server.

    Parameters
    ----------
    port : int
        The port number to connect to.

    password : str
        The password for authentication.

    Returns
    -------
    Client
        The client object representing the connection to the server.

    Raises
    ------
    EOFError
        If the host process is not ready yet.
    ConnectionRefusedError
        If the connection is refused by the server.

    Notes
    -----
    This function attempts to establish a connection to a server using the given port and password.
    It retries the connection up to two times before raising an exception.
    """
    address = ("", port)
    retries = 2
    while retries > 0:
        time.sleep(1)  # wait a second and then try to make a connection
        try:
            # if we manage to establish a connection we exit the function
            return Client(address, authkey=password.encode())
        # maybe the host process isn't ready yet
        except EOFError:
            pass
        except ConnectionRefusedError:
            pass
        retries -= 1
    # when we get here, e must have been instantiated
    logging.fatal("Could not connect to host process.")
    raise ConnectionError("Could not connect to host process.")


class ProcessWrapper:
    def __init__(self, run_script, **kwargs_for_factory):
        """
        Initialize the connection for inter process communication.

        Parameters
        ----------
        run_script : str
            The run script to execute. The run script should accept a port and a password as arguments.

        **kwargs_for_factory : dict
            Additional keyword arguments to be passed to the run script.
            These will be passed to the inner objective factory.

        Notes
        -----
        This class sets up a server for inter process communication. It generates a password for authentication,
        creates a listener on a random port, and starts a subprocess to execute the run script. The run script
        is expected to take the port and password as arguments, as well as any other arguments passed by the user
        when calling `objective_factory.create`.

        The `kwargs_for_factory` dictionary is used to convert the keyword arguments into a string format that can
        be passed to the run script. The string format is determined based on the type of the argument.
        """
        address = ("", 0)  # ('localhost', 6000)
        self.password = _generate_password()
        self.listener = Listener(address, authkey=self.password.encode())

        # TODO: very hacky way to read out the socket! (but the listener is not very cooperative)
        self.port = self.listener._listener._socket.getsockname()[1]
        # here is a VERY crucial step
        # we expect the shell script to take port and password as arguments, as well as other arguments passed by the user
        # when calling objective_factory.create
        # TODO: This is a very silly way to handle communication between processes,
        # and it is also very dangerous, because the user can pass arbitrary arguments
        # to the shell script. We should instead use a proper IPC library.
        # TODO: if we decide to pass this information in the set-up phase (instead
        # of here), we can remove this.
        string_for_kwargs = ""
        for key, value in kwargs_for_factory.items():
            if isinstance(value, str):
                string_for_kwargs += f"--{key}={str(value)} "
            elif isinstance(value, Path):
                string_for_kwargs += f"--{key}={str(value)} "
            elif isinstance(value, bool):
                string_for_kwargs += f"--{key}=bool:{str(value)} "
            elif isinstance(value, int):
                string_for_kwargs += f"--{key}=int:{str(value)} "
            elif isinstance(value, float):
                string_for_kwargs += f"--{key}=float:{str(value)} "
            elif isinstance(value, list):
                string_for_kwargs += (
                    f"--{key}=list:{','.join([str(v) for v in value])} "
                )
            elif value is None:
                string_for_kwargs += f"--{key}=none:None "

        self.run_script = run_script
        self.proc = subprocess.Popen(
            [run_script, str(self.port), self.password, string_for_kwargs],
            stdout=None,
            stderr=None,
        )

        self.conn = self.listener.accept()  # wait for the process to connect

    def send(self, *args):
        """
        Send data through the connection.

        Parameters
        ----------
        *args : tuple
            The data to be sent.

        Returns
        -------
        object
            The return value of the send operation.

        """
        return self.conn.send(*args)

    def recv(self):
        """
        Receive data from the connection.

        Returns
        -------
        object
            The received data.
        """
        return self.conn.recv()

    def close(self):
        """
        Closes the connection.
        """
        # TODO: potentially dangerous to wait here!
        self.proc.wait()  # wait for objective function process to finish
        self.listener.close()

    def __str__(self) -> str:
        return f"ProcessWrapper(port={self.port}, script_location={self.run_script})"

    def __repr__(self) -> str:
        return f"<{self.__str__()}>"


def _generate_password() -> str:
    """
    Generates a random password using uuid4.

    Returns
    -------
    password: str
        The generated password.
    """
    return str(uuid4())
