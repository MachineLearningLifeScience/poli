"""
Module that wraps utility functions for interprocess communication.
"""
__author__ = 'Simon Bartels'

import logging
import os
import subprocess
import time
from multiprocessing.connection import Listener, Client


def get_connection(port: int, password: str):
    """
    Function for clients to get a connection to a server.
    """
    address = ('', port)
    retries = 2
    while retries > 0:
        time.sleep(1)  # wait a second and then try to make a connection
        try:
            # if we manage to establish a connection we exit the function
            return Client(address, authkey=password.encode())
        # maybe the host process isn't ready yet
        except EOFError as e:
            pass
        except ConnectionRefusedError as e:
            pass
        retries -= 1
    # when we get here, e must have been instantiated
    logging.fatal("Could not connect to host process.")
    raise e


class ProcessWrapper:
    def __init__(self, run_script):
        """
        Server class for inter process communication.

        :param run_script:
            which run script to execute
            IMPORTANT: The run script has to accept a port and a password as arguments and should call #get_connection
            with these parameters.
        """
        address = ('', 0)  #('localhost', 6000)
        self.password = _generate_password()
        self.listener = Listener(address, authkey=self.password.encode())
        # TODO: very hacky way to read out the socket! (but the listener is not very cooperative)
        self.port = self.listener._listener._socket.getsockname()[1]
        # here is a VERY crucial step
        # we expect the shell script to take port and password as arguments
        self.proc = subprocess.Popen([run_script, str(self.port), self.password], stdout=None, stderr=None)
        self.conn = self.listener.accept()  # wait for the process to connect

    def send(self, *args):
        return self.conn.send(*args)

    def recv(self):
        return self.conn.recv()

    def close(self):
        # TODO: potentially dangerous to wait here!
        self.proc.wait()  # wait for objective function process to finish
        self.listener.close()


def _generate_password() -> str:
    # TODO: actually generate safe password
    return 'secret password'
