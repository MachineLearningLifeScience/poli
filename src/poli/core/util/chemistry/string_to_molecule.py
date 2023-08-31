"""
Utilities for transforming string representations of
molecules into molecules in RDKit.
"""
from typing import List

from rdkit import Chem

import selfies as sf


def smiles_to_molecules(smiles_strings: List[str], strict: bool = False) -> Chem.Mol:
    """
    Convert a SMILES string to an RDKit molecule. If strict is True,
    it raises an error if a SMILES string in the list cannot be parsed.

    Parameters
    ----------
    smiles : List[str]
        A list of SMILES string.

    Returns
    -------
    Chem.Mol
        An RDKit molecule.
    """
    molecules = []
    for smiles_string in smiles_strings:
        molecule = Chem.MolFromSmiles(smiles_string)

        if molecule is None and strict:
            raise ValueError(f"Failed to parse molecule.")

        molecules.append(molecule)

    return molecules


def selfies_to_molecules(selfies_strings: List[str]) -> Chem.Mol:
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
        smiles_strings = [
            sf.decoder(selfies_string) for selfies_string in selfies_strings
        ]
    except sf.DecoderError:
        # If selfies failed to decode the selfies to smiles, return NaN
        raise ValueError("Failed to decode selfies string.")

    molecule = smiles_to_molecules(smiles_strings)
    return molecule


def strings_to_molecules(molecule_strings: List[str], from_selfies: bool = False):
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
        return selfies_to_molecules(molecule_strings)
    else:
        return smiles_to_molecules(molecule_strings)
