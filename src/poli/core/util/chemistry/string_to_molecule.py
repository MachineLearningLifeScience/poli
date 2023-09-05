"""
Utilities for transforming string representations of
molecules into molecules in RDKit.
"""
from typing import List

from rdkit import Chem

import selfies as sf


def translate_smiles_to_selfies(
    smiles_strings: List[str], strict: bool = False
) -> List[str]:
    """
    Given a list of SMILES strings, returns the translation
    into SELFIES strings. If strict is True, it raises an error
    if a SMILES string in the list cannot be parsed. Else, it
    returns None for those.
    """
    selfies_strings = []
    for smile in smiles_strings:
        try:
            selfies_strings.append(sf.encoder(smile))
        except sf.EncoderError:
            if strict:
                raise ValueError(f"Failed to encode SMILES to SELFIES.")
            else:
                selfies_strings.append(None)

    return selfies_strings


def translate_selfies_to_smiles(
    selfies_strings: List[str], strict: bool = False
) -> List[str]:
    """
    Given a list of SELFIES strings, returns the translation
    into SMILES strings. If strict is True, it raises an error
    if a SELFIES string in the list cannot be parsed. Else, it
    returns None for those.
    """
    smiles_strings = []
    for selfies in selfies_strings:
        try:
            smiles_strings.append(sf.decoder(selfies))
        except sf.DecoderError:
            if strict:
                raise ValueError(f"Failed to decode SELFIES to SMILES.")
            else:
                smiles_strings.append(None)

    return smiles_strings


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
    smiles_strings = translate_smiles_to_selfies(selfies_strings, strict=True)
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
