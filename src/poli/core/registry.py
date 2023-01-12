import os
import configparser
import warnings

from poli.core.abstract_problem_factory import AbstractProblemFactory

_DEFAULT = 'DEFAULT'
_OBSERVER = 'observer'
_RUN_SCRIPT_LOCATION = 'run_script_location'


config_file = os.path.join(os.path.dirname(__file__), '..', 'config.rc')
config = configparser.ConfigParser(defaults={_OBSERVER: ''})
ls = config.read(config_file)
# if len(ls) == 0:
#     warnings.warn("Could not find configuration file: %s" % config_file)


def set_observer_run_script(script_file_name: str):
    config[_DEFAULT][_OBSERVER] = script_file_name
    _write_config()


def delete_observer_run_script():
    config[_DEFAULT][_OBSERVER] = ''
    _write_config()


def register_problem(problem_factory: AbstractProblemFactory, run_script_location: str):
    problem_name = problem_factory.get_setup_information().get_problem_name()
    if problem_name not in config.sections():
        config.add_section(problem_name)
    else:
        warnings.warn(f"Problem {problem_name} already exists. Overwriting.")
    config[problem_name][_RUN_SCRIPT_LOCATION] = run_script_location
    _write_config()


def delete_problem(problem_name: str):
    config.remove_section(problem_name)
    _write_config()


def _write_config():
    with open(config_file, 'w+') as configfile:
        config.write(configfile)
