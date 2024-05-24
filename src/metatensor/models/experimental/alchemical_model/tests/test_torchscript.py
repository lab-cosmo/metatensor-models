import copy

import ase
import torch
from metatensor.torch.atomistic import systems_to_torch

from metatensor.models.experimental.alchemical_model import AlchemicalModel
from metatensor.models.utils.data import DatasetInfo, TargetInfo

from . import MODEL_HYPERS


def test_torchscript():
    """Tests that the model can be jitted."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": TargetInfo(
                quantity="energy",
                unit="eV",
            )
        },
    )

    model = AlchemicalModel(MODEL_HYPERS, dataset_info)
    torch.jit.script(model, {"energy": model.outputs["energy"]})


def test_torchscript_with_identity():
    """Tests that the model can be jitted."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": TargetInfo(
                quantity="energy",
                unit="eV",
            )
        },
    )
    hypers = copy.deepcopy(MODEL_HYPERS)
    model = AlchemicalModel(hypers, dataset_info)
    model = torch.jit.script(model)

    system = ase.Atoms(
        "OHCN",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]],
    )
    model(
        [systems_to_torch(system)],
        {"energy": model.outputs["energy"]},
    )


def test_torchscript_save_load():
    """Tests that the model can be jitted and saved."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": TargetInfo(
                quantity="energy",
                unit="eV",
            )
        },
    )
    model = AlchemicalModel(MODEL_HYPERS, dataset_info)
    torch.jit.save(
        torch.jit.script(
            model,
            {"energy": model.outputs["energy"]},
        ),
        "alchemical_model.pt",
    )

    torch.jit.load("model.pt")
