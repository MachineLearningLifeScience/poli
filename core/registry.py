#  Copyright (c) 2023.
import os

from objectives.common import __file__ as common_path

COMMONS = os.path.dirname(common_path)

base_ipc_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp", "ipc")
if not os.path.exists(base_ipc_folder):
    os.makedirs(base_ipc_folder)
INIT_DATA_FILE = os.path.join(base_ipc_folder , "init_data.npy")
INPUT_DATA_FILE = os.path.join(base_ipc_folder , "input_data.npy")
OUTPUT_DATA_FILE = os.path.join(base_ipc_folder , "output_data.npy")
