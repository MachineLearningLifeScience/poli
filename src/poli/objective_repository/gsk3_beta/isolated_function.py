from typing import Literal

from poli.core.chemistry.tdc_isolated_function import TDCIsolatedFunction
from poli.core.registry import register_isolated_function


class GSK3BetaIsolatedFunction(TDCIsolatedFunction):
    """
    An isolated function for Glycogen Synthase
    Kinase 3 Beta, using the Therapeutics Data
    Commons' oracles.

    Parameters
    ----------
    string_representations : string, optional
        The string representation of the molecules.
        Either "SMILES" or "SELFIES".

    Attributes
    ----------
    oracle_name : str
        The name of the oracle.

    Methods
    -------
    __init__(self, string_representation) -> None
        Initializes the isolated function.
    """

    def __init__(
        self,
        string_representation: Literal["SMILES", "SELFIES"] = "SMILES",
    ):
        oracle_name = "GSK3B"
        super().__init__(
            oracle_name=oracle_name,
            from_smiles=string_representation == "SMILES",
        )


if __name__ == "__main__":
    register_isolated_function(
        GSK3BetaIsolatedFunction,
        name="gsk3_beta__isolated",
        conda_environment_name="poli__tdc",
    )
