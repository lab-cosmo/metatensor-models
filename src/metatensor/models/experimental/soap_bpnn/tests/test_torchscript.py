import torch
from metatensor.torch.atomistic import ModelCapabilities, ModelOutput

from metatensor.models.experimental.soap_bpnn import DEFAULT_HYPERS, Model


def test_torchscript():
    """Tests that the model can be jitted."""

    capabilities = ModelCapabilities(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        outputs={
            "energy": ModelOutput(
                quantity="energy",
                unit="eV",
            )
        },
    )
    soap_bpnn = Model(capabilities, DEFAULT_HYPERS["model"])
    torch.jit.script(soap_bpnn, {"energy": soap_bpnn.capabilities.outputs["energy"]})


def test_torchscript_save():
    """Tests that the model can be jitted and saved."""

    capabilities = ModelCapabilities(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        outputs={
            "energy": ModelOutput(
                quantity="energy",
                unit="eV",
            )
        },
    )
    soap_bpnn = Model(capabilities, DEFAULT_HYPERS["model"])
    torch.jit.save(
        torch.jit.script(
            soap_bpnn, {"energy": soap_bpnn.capabilities.outputs["energy"]}
        ),
        "soap_bpnn.pt",
    )
