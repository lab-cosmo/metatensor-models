import torch

from metatensor.models.soap_bpnn import DEFAULT_HYPERS, Model


def test_torchscript():
    """Tests that the model can be jitted."""

    all_species = [1, 6, 7, 8]
    soap_bpnn = Model(all_species, DEFAULT_HYPERS["model"]).to(torch.float64)
    torch.jit.script(soap_bpnn)
