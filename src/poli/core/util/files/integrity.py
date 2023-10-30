"""
This module contains utilities for checking the integrity of
files downloaded from the internet. This is useful for
reproducibility purposes, and to make sure no malicious
code is being executed.
"""
from pathlib import Path

import hashlib


def compute_md5_from_filepath(filepath: Path, read_mode: str = "rb") -> str:
    """
    Computes the MD5 hex digest from a filepath,
    opening it in binary form by default.
    """
    with open(filepath, read_mode) as fp:
        checksum = hashlib.md5(fp.read()).hexdigest()

    return checksum
