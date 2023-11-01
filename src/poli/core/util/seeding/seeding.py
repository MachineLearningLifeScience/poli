import numpy as np
import random


def seed_numpy(seed: int = None) -> None:
    np.random.seed(seed)


def seed_python(seed: int = None) -> None:
    random.seed(seed)
