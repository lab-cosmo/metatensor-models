import random

import ase.io
import numpy as np
import torch
from metatensor.torch.atomistic import ModelCapabilities, ModelOutput, systems_to_torch
from omegaconf import OmegaConf

from metatensor.models.experimental.soap_bpnn import DEFAULT_HYPERS, Model, train
from metatensor.models.utils.data import Dataset, DatasetInfo, TargetInfo
from metatensor.models.utils.data.readers import read_systems, read_targets

from . import DATASET_PATH


# reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def test_regression_init():
    """Perform a regression test on the model at initialization"""

    capabilities = ModelCapabilities(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        outputs={
            "mtm::U0": ModelOutput(
                quantity="energy",
                unit="eV",
            )
        },
        interaction_range=DEFAULT_HYPERS["model"]["soap"]["cutoff"],
        dtype="float32",
    )
    soap_bpnn = Model(capabilities, DEFAULT_HYPERS["model"])

    # Predict on the first five systems
    systems = ase.io.read(DATASET_PATH, ":5")

    output = soap_bpnn(
        [systems_to_torch(system) for system in systems],
        {"mtm::U0": soap_bpnn.capabilities.outputs["mtm::U0"]},
    )
    expected_output = torch.tensor(
        [[-0.0564], [0.0296], [0.0182], [-0.1102], [-0.0547]]
    )

    torch.set_printoptions(precision=12)
    print(output["mtm::U0"].block().values)

    torch.testing.assert_close(
        output["mtm::U0"].block().values, expected_output, rtol=1e-3, atol=1e-08
    )


def test_regression_train():
    """Perform a regression test on the model when
    trained for 2 epoch on a small dataset"""

    systems = read_systems(DATASET_PATH)

    conf = {
        "mtm::U0": {
            "quantity": "energy",
            "read_from": DATASET_PATH,
            "file_format": ".xyz",
            "key": "U0",
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets = read_targets(OmegaConf.create(conf))
    dataset = Dataset({"system": systems, "mtm::U0": targets["mtm::U0"]})

    hypers = DEFAULT_HYPERS.copy()
    hypers["training"]["num_epochs"] = 2

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        targets={
            "mtm::U0": TargetInfo(
                quantity="energy",
                unit="eV",
            ),
        },
    )
    soap_bpnn = train([dataset], [dataset], dataset_info, [torch.device("cpu")], hypers)

    # Predict on the first five systems
    output = soap_bpnn(
        systems[:5], {"mtm::U0": soap_bpnn.capabilities.outputs["mtm::U0"]}
    )

    expected_output = torch.tensor(
        [[-40.4387], [-56.4573], [-76.2703], [-77.3089], [-93.4303]]
    )

    torch.testing.assert_close(
        output["mtm::U0"].block().values, expected_output, rtol=1e-3, atol=1e-08
    )
