__author__ = 'Simon Bartels'

import sys
import numpy as np
import logging

from poli.core.problem_setup_information import ProblemSetupInformation
from poli.core.util.abstract_observer import AbstractObserver




    # we have to register our observer run_script with the registry
    from poli.core.registry import set_observer_run_script
    set_observer_run_script(observer_name)


