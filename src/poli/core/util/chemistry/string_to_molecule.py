"""
Utilities for transforming string representations of
molecules into molecules in RDKit.
"""
from rdkit import Chem

import selfies as sf


def smiles_to_molecule(smiles_string: str) -> Chem.Mol:
    """
    Convert a SMILES string to an RDKit molecule.

    Parameters
    ----------
    smiles : str
        A SMILES string.

    Returns
    -------
    Chem.Mol
        An RDKit molecule.
    """
    molecule = Chem.MolFromSmiles(smiles_string)
    if molecule is None:
        raise ValueError("Failed to parse molecule.")

    return molecule


def selfies_to_molecule(selfies_string: str) -> Chem.Mol:
    """
    Convert a selfies string to an RDKit molecule.

    Parameters
    ----------
    selfies : str
        A selfies string.
    strict : bool, optional
        If True, raise an error if selfies fails to decode the selfies string.

    Returns
    -------
    Chem.Mol
        An RDKit molecule.
    """
    try:
        smiles_string = sf.decoder(selfies_string)
    except sf.DecoderError:
        # If selfies failed to decode the selfies to smiles, return NaN
        raise ValueError("Failed to decode selfies string.")

    molecule = smiles_to_molecule(smiles_string)
    return molecule


def string_to_molecule(molecule_string: str, from_selfies: bool = False):
    """
    Convert a string representation of a molecule to an RDKit molecule.

    Parameters
    ----------
    molecule_string : str
        A string representation of a molecule.
    from_selfies : bool, optional
        If True, the molecule string is assumed to be a selfies string.
        Otherwise, the molecule string is assumed to be a SMILES string.

    Returns
    -------
    Chem.Mol
        An RDKit molecule.
    """
    if from_selfies:
        return selfies_to_molecule(molecule_string)
    else:
        return smiles_to_molecule(molecule_string)
