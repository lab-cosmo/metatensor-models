import torch
torch.random.manual_seed(0)
import ase.io
import rascaline.torch
import copy

import yaml

from metatensor_models.soap_bpnn import SoapBPNN


def test_regression_init():
    """Perform a regression test on the model at initialization"""

    all_species = [1, 6, 7, 8, 9]
    hypers = yaml.safe_load(open("../default.yml", "r"))
    soap_bpnn = SoapBPNN(all_species, hypers).to(torch.float64)
    
    structures = ase.io.read("data/qm9_reduced_100.xyz", ":5")

    output = soap_bpnn([rascaline.torch.systems_to_torch(structure) for structure in structures])
    expected_output = torch.tensor(
        [[ 0.051073328644],
        [ 0.226986056644],
        [-0.069471667587],
        [-0.218890301061],
        [-0.043132789197]],
        dtype=torch.float64
    )

    assert torch.allclose(output["energy"].block().values, expected_output)


def test_regression_train():
    """Perform a regression test on the model when trained for 2 epoch trained on a small dataset"""
    # TODO: Implement this test
