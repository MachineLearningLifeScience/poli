"""Utilities for verifying the integrity of files downloaded from the internet.

This module contains utilities for checking the integrity of
files downloaded from the internet. This is useful for
reproducibility purposes, and to make sure no malicious
code is being executed.
"""

import hashlib
from pathlib import Path


def compute_md5_from_filepath(filepath: Path, read_mode: str = "rb") -> str:
    """
    Computes the MD5 hex digest from a filepath,
    opening it in binary form by default.

    Parameters
    ----------
    filepath : Path
        The path to the file for which the MD5 digest will be computed.
    read_mode : str, optional
        The mode in which the file will be opened. Default is 'rb' (read in binary mode).

    Returns
    -------
    hex_digest : str
        The MD5 hex digest of the file.

    Examples:
    --------
    >>> compute_md5_from_filepath('/path/to/file.txt')
    'd41d8cd98f00b204e9800998ecf8427e'
    """
    with open(filepath, read_mode) as fp:
        checksum = hashlib.md5(fp.read()).hexdigest()

    return checksum
