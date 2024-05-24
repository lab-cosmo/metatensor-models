"""Test command line interface for the export functions.

Actual unit tests for the function are performed in `tests/utils/test_export`.
"""

import subprocess
from pathlib import Path

import pytest
from metatensor.torch.atomistic import load_atomistic_model

from metatensor.models.cli.export import export_model
from metatensor.models.experimental.soap_bpnn import __model__
from metatensor.models.utils.data import DatasetInfo, TargetInfo

from . import MODEL_HYPERS, RESOURCES_PATH


@pytest.mark.parametrize("path", [Path("exported.pt"), "exported.pt"])
def test_export(monkeypatch, tmp_path, path):
    """Tests the export_model function."""
    monkeypatch.chdir(tmp_path)

    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types=[1],
        targets={
            "energy": TargetInfo(
                quantity="energy", unit="eV", per_atom=False, gradients=[]
            )
        },
    )
    model = __model__(model_hypers=MODEL_HYPERS, dataset_info=dataset_info)
    export_model(model, path)

    assert Path(path).is_file()


@pytest.mark.parametrize("output", [None, "exported.pt"])
def test_export_cli(monkeypatch, tmp_path, output):
    """Test that the export cli runs without an error raise."""
    monkeypatch.chdir(tmp_path)
    command = [
        "metatensor-models",
        "export",
        "experimental.soap_bpnn",
        str(RESOURCES_PATH / "model-32-bit.ckpt"),
    ]

    if output is not None:
        command += ["-o", output]
    else:
        output = "exported-model.pt"

    subprocess.check_call(command)
    assert Path(output).is_file()


def test_reexport(monkeypatch, tmp_path):
    """Test that an already exported model can be loaded and again exported."""
    monkeypatch.chdir(tmp_path)

    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": TargetInfo(
                quantity="energy", unit="eV", per_atom=False, gradients=[]
            )
        },
    )
    model = __model__(model_hypers=MODEL_HYPERS, dataset_info=dataset_info)

    export_model(model, "exported.pt")

    model_loaded = load_atomistic_model("exported.pt")
    export_model(model_loaded, "exported_new.pt")

    assert Path("exported_new.pt").is_file()
