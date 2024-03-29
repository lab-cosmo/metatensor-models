import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import metatensor.torch
import torch
from metatensor.learn.data.dataset import Dataset
from metatensor.torch import Labels, TensorBlock, TensorMap
from omegaconf import DictConfig, OmegaConf

from ..utils.data import collate_fn, read_systems, read_targets, write_predictions
from ..utils.errors import ArchitectureError
from ..utils.evaluate_model import evaluate_model
from ..utils.io import load
from ..utils.logging import MetricLogger
from ..utils.metrics import RMSEAccumulator
from ..utils.neighbors_lists import get_system_with_neighbors_lists
from ..utils.omegaconf import expand_dataset_config
from .formatter import CustomHelpFormatter


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _add_eval_model_parser(subparser: argparse._SubParsersAction) -> None:
    """Add the `eval_model` paramaters to an argparse (sub)-parser"""

    if eval_model.__doc__ is not None:
        description = eval_model.__doc__.split(r":param")[0]
    else:
        description = None

    # If you change the synopsis of these commands or add new ones adjust the completion
    # script at `src/metatensor/models/share/metatensor-models-completion.bash`.
    parser = subparser.add_parser(
        "eval",
        description=description,
        formatter_class=CustomHelpFormatter,
    )
    parser.set_defaults(callable="eval_model")
    parser.add_argument(
        "model",
        type=load,
        help="Saved exported model to be evaluated.",
    )
    parser.add_argument(
        "options",
        type=OmegaConf.load,
        help="Eval options file to define a dataset for evaluation.",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        type=str,
        required=False,
        default="output.xyz",
        help="filename of the predictions (default: %(default)s)",
    )


def _concatenate_tensormaps(
    tensormaps: List[Dict[str, TensorMap]]
) -> Dict[str, TensorMap]:
    # Concatenating TensorMaps is tricky, because the model does not know the
    # "number" of the system it is predicting. For example, if a model predicts
    # 3 batches of 4 atoms each, the system labels will be [0, 1, 2, 3],
    # [0, 1, 2, 3], [0, 1, 2, 3] for the three batches, respectively. Due
    # to this, the join operation would not achieve the desired result
    # ([0, 1, 2, ..., 11, 12]). Here, we fix this by renaming the system labels.

    system_counter = 0
    tensormaps_shifted_systems = []
    for tensormap_dict in tensormaps:
        tensormap_dict_shifted = {}
        for name, tensormap in tensormap_dict.items():
            new_keys = []
            new_blocks = []
            for key, block in tensormap.items():
                new_key = key
                where_system = block.samples.names.index("system")
                n_systems = torch.max(block.samples.column("system")) + 1
                new_samples_values = block.samples.values
                new_samples_values[:, where_system] += system_counter
                new_block = TensorBlock(
                    values=block.values,
                    samples=Labels(block.samples.names, values=new_samples_values),
                    components=block.components,
                    properties=block.properties,
                )
                for gradient_name, gradient_block in block.gradients():
                    new_block.add_gradient(
                        gradient_name,
                        gradient_block,
                    )
                new_keys.append(new_key)
                new_blocks.append(new_block)
            tensormap_dict_shifted[name] = TensorMap(
                keys=Labels(
                    names=tensormap.keys.names,
                    values=torch.stack([new_key.values for new_key in new_keys]),
                ),
                blocks=new_blocks,
            )
        tensormaps_shifted_systems.append(tensormap_dict_shifted)
        system_counter += n_systems

    return {
        target: metatensor.torch.join(
            [pred[target] for pred in tensormaps_shifted_systems], axis="samples"
        )
        for target in tensormaps_shifted_systems[0].keys()
    }


def _eval_targets(
    model: torch.jit._script.RecursiveScriptModule,
    dataset: Union[Dataset, torch.utils.data.Subset],
    options: Dict[str, List[str]],
    return_predictions: bool,
) -> Optional[Dict[str, TensorMap]]:
    """Evaluates an exported model on a dataset and prints the RMSEs for each target.
    Optionally, it also returns the predictions of the model."""

    if len(dataset) == 0:
        logger.info("This dataset is empty. No evaluation will be performed.")

    # Attach neighbor lists to the systems:
    # TODO: these might already be present... find a way to avoid recomputing
    # if already present (e.g. if this function is called after training)
    for sample in dataset:
        system = sample.system
        get_system_with_neighbors_lists(system, model.requested_neighbors_lists())

    # Infer the device from the model
    device = next(model.parameters()).device

    # Create a dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,  # TODO: allow to set from outside!!
        collate_fn=collate_fn,
        shuffle=False,
    )

    # Initialize RMSE accumulator:
    rmse_accumulator = RMSEAccumulator()

    # If we're returning the predictions, we need to store them:
    if return_predictions:
        all_predictions = []

    # Evaluate the model
    for batch in dataloader:
        systems, targets = batch
        systems = [system.to(device=device) for system in systems]
        targets = {key: value.to(device=device) for key, value in targets.items()}
        batch_predictions = evaluate_model(model, systems, options, is_training=False)
        rmse_accumulator.update(batch_predictions, targets)
        if return_predictions:
            all_predictions.append(batch_predictions)

    # Finalize the RMSEs
    rmse_values = rmse_accumulator.finalize()
    # print the RMSEs with MetricLogger
    metric_logger = MetricLogger(
        model_capabilities=model.capabilities(),
        initial_metrics=rmse_values,
    )
    metric_logger.log(rmse_values)

    if return_predictions:
        # concatenate the TensorMaps
        all_predictions_joined = _concatenate_tensormaps(all_predictions)
        return all_predictions_joined
    else:
        return None


def eval_model(
    model: torch.nn.Module, options: DictConfig, output: Union[Path, str] = "output.xyz"
) -> None:
    """Evaluate an exported model on a given data set.

    If ``options`` contains a ``targets`` sub-section, RMSE values will be reported. If
    this sub-section is missing, only a xyz-file with containing the properties the
    model was trained against is written.

    :param model: Saved model to be evaluated.
    :param options: DictConfig to define a test dataset taken for the evaluation.
    :param output: Path to save the predicted values
    """
    if not isinstance(model, torch.jit._script.RecursiveScriptModule):
        raise ValueError(
            "The model must already be exported to be used in `eval`. "
            "If you are trying to evaluate a checkpoint, export it first "
            "with the `metatensor-models export` command."
        )
    logger.info("Setting up evaluation set.")

    # TODO: once https://github.com/lab-cosmo/metatensor/pull/551 is merged and released
    # use capabilities instead of this workaround
    dtype = next(model.parameters()).dtype

    if isinstance(output, str):
        output = Path(output)

    options_list = expand_dataset_config(options)
    for i, options in enumerate(options_list):
        if len(options_list) == 1:
            extra_log_message = ""
            file_index_suffix = ""
        else:
            extra_log_message = f" with index {i}"
            file_index_suffix = f"_{i}"
        logger.info(f"Evaluating dataset{extra_log_message}")

        eval_systems = read_systems(
            filename=options["systems"]["read_from"],
            fileformat=options["systems"]["file_format"],
            dtype=dtype,
        )

        if hasattr(options, "targets"):
            # in this case, we only evaluate the targets specified in the options
            # and we calculate RMSEs
            eval_targets = read_targets(options["targets"], dtype=dtype)
            eval_outputs = {
                target: tensormaps[0].block().gradients_list()
                for target, tensormaps in eval_targets.items()
            }
        else:
            # in this case, we have no targets: we evaluate everything
            # (but we don't/can't calculate RMSEs)
            # TODO: allow the user to specify which outputs to evaluate
            eval_targets = {}
            gradients = ["positions"]
            if all(not torch.all(system.cell == 0) for system in eval_systems):
                # only add strain if all structures have cells
                gradients.append("strain")
            eval_outputs = {
                target: gradients for target in model.capabilities().outputs.keys()
            }

        eval_dataset = Dataset(system=eval_systems, **eval_targets)

        # Evaluate the model
        try:
            predictions = _eval_targets(
                model=model,
                dataset=eval_dataset,
                options=eval_outputs,
                return_predictions=True,
            )
        except Exception as e:
            raise ArchitectureError(e)

        write_predictions(
            filename=f"{output.stem}{file_index_suffix}{output.suffix}",
            systems=eval_systems,
            capabilities=model.capabilities(),
            predictions=predictions,
        )
