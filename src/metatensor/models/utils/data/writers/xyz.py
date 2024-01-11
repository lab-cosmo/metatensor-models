from typing import List

import ase
import ase.io
import torch
from metatensor.torch import TensorMap
from rascaline.torch.system import System


def write_xyz(filename: str, predictions: TensorMap, structures: List[System]) -> None:
    """An ase based xyz file writer

    :param filename: name of the file to read
    :param predictions: prediction values written to the file.
    :param structures: strcutures additional written to the file.
    """
    # Get the target property name:
    target_name = next(iter(predictions.keys()))

    frames = []
    for i_system, system in enumerate(structures):
        info = {
            target_name: float(predictions[target_name].block().values[i_system, 0])
        }
        atoms = ase.Atoms(symbols=system.species, positions=system.positions, info=info)

        if torch.any(system.cell != 0):
            atoms.pbc = True
            atoms.cell = system.cell

        frames.append(atoms)

    ase.io.write(filename, frames)
