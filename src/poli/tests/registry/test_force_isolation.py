"""This module tests whether forcing the isolation of black
box objectives (i) creates the relevant conda environment, and
(ii) writes the isolated function script in the config,
(iii) getting the inner function works as expected."""

import subprocess

from poli.core.util.isolation.instancing import get_inner_function


def test_force_isolation_on_tdc():
    from poli import objective_factory

    problem = objective_factory.create(
        name="deco_hop",
        force_isolation=True,
    )

    # After creating the problem, we check that:
    # (i) creates the relevant conda environment
    assert "poli__tdc" in subprocess.check_output("conda env list".split()).decode()

    # (ii) writes the isolated function script in the config
    from poli.objective_factory import load_config

    config = load_config()
    assert config["tdc__isolated"]

    # (iii) we can call the inner function
    inner_f = get_inner_function(
        isolated_function_name="tdc__isolated",
        class_name="TDCIsolatedLogic",
        module_to_import="poli.core.chemistry.tdc_isolated_function",
        force_isolation=True,
        oracle_name="deco_hop",
        from_smiles=True,
        quiet=True,
    )
    assert (problem.black_box(problem.x0) == inner_f(problem.x0)).all()
