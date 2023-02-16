"""
Module that wraps utility functions for interprocess communication.
"""

__author__ = 'Simon Bartels'

from multiprocessing.connection import Listener, Client


def get_connection(port: int, password: str):
    address = ('', port)
    conn = Client(address, authkey=password.encode())
    return conn


def instantiate_listener():
    address = ('', 0)  #('localhost', 6000)
    password = _generate_password()
    listener = Listener(address, authkey=password.encode())
    # TODO: very hacky way to read out the socket! (but the listener is not very cooperative)
    port = listener._listener._socket.getsockname()[1]
    return listener, port, password


def _generate_password() -> str:
    # TODO: actually generate safe password
    return 'secret password'
