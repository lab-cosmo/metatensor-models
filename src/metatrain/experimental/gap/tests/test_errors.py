import copy
import random
import re

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from metatrain.experimental.gap import GAP, Trainer
from metatrain.utils.data import (
    Dataset,
    DatasetInfo,
    TargetInfo,
    TargetInfoDict,
    read_systems,
    read_targets,
)

from . import DATASET_ETHANOL_PATH, DEFAULT_HYPERS


torch.set_default_dtype(torch.float64)  # GAP only supports float64


# reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def test_ethanol_regression_train_and_invariance():
    """test the error if the number of sparse point
    is bigger than the number of environments
    """

    systems = read_systems(DATASET_ETHANOL_PATH, dtype=torch.float64)

    conf = {
        "energy": {
            "quantity": "energy",
            "read_from": DATASET_ETHANOL_PATH,
            "file_format": ".xyz",
            "key": "energy",
            "unit": "kcal/mol",
            "forces": {
                "read_from": DATASET_ETHANOL_PATH,
                "file_format": ".xyz",
                "key": "forces",
            },
            "stress": False,
            "virial": False,
        }
    }

    targets, _ = read_targets(OmegaConf.create(conf), dtype=torch.float64)
    dataset = Dataset({"system": systems[:2], "energy": targets["energy"][:2]})

    hypers = copy.deepcopy(DEFAULT_HYPERS)
    hypers["model"]["krr"]["num_sparse_points"] = 30

    target_info_dict = TargetInfoDict(
        energy=TargetInfo(quantity="energy", unit="kcal/mol")
    )

    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types={1, 6, 7, 8}, targets=target_info_dict
    )

    gap = GAP(hypers["model"], dataset_info)
    trainer = Trainer(hypers["training"])
    with pytest.raises(
        ValueError,
        match=re.escape(
            """number of sparse points (30)
 should be smaller than the number of environments (18)"""
        ),
    ):
        trainer.train(gap, [torch.device("cpu")], [dataset], [dataset], ".")
