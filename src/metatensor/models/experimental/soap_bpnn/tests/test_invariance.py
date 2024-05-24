import copy

import ase.io
import torch
from metatensor.torch.atomistic import systems_to_torch

from metatensor.models.experimental.soap_bpnn import SOAPBPNN
from metatensor.models.utils.data import DatasetInfo, TargetInfo

from . import DATASET_PATH, MODEL_HYPERS


def test_rotational_invariance():
    """Tests that the model is rotationally invariant."""

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
    soap_bpnn = SOAPBPNN(MODEL_HYPERS, dataset_info)

    system = ase.io.read(DATASET_PATH)
    original_system = copy.deepcopy(system)
    system.rotate(48, "y")

    original_output = soap_bpnn(
        [systems_to_torch(original_system)],
        {"energy": soap_bpnn.outputs["energy"]},
    )
    rotated_output = soap_bpnn(
        [systems_to_torch(system)],
        {"energy": soap_bpnn.outputs["energy"]},
    )

    torch.testing.assert_close(
        original_output["energy"].block().values,
        rotated_output["energy"].block().values,
    )
