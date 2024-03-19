import logging
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import torch
from metatensor.learn.data import Dataset, group_and_join
from metatensor.torch import TensorMap
from torch import Generator, default_generator
from torch.utils.data import Subset, random_split


logger = logging.getLogger(__name__)


@dataclass
class TargetInfo:
    """A class that contains information about a target.

    :param quantity: The quantity of the target.
    :param unit: The unit of the target.
    :param per_atom: Whether the target is a per-atom quantity.
    """

    quantity: str
    unit: str
    per_atom: bool = False


@dataclass
class DatasetInfo:
    """A class that contains information about one or more datasets.

    This dataclass is used to communicate additional dataset details to the
    training functions of the individual models.

    :param length_unit: The unit of length used in the dataset.
    :param targets: The information about targets in the dataset.
    """

    length_unit: str
    targets: Dict[str, TargetInfo]


def get_all_species(datasets: Union[Dataset, List[Dataset]]) -> List[int]:
    """
    Returns the list of all species present in a dataset or list of datasets.

    :param datasets: the dataset, or list of datasets.
    :returns: The sorted list of species present in the datasets.
    """

    if not isinstance(datasets, list):
        datasets = [datasets]

    # Iterate over all single instances of the dataset:
    species = []
    for dataset in datasets:
        for index in range(len(dataset)):
            system = dataset[index][0]  # extract the system from the NamedTuple
            species += system.types.tolist()

    # Remove duplicates and sort:
    result = list(set(species))
    result.sort()

    return result


def get_all_targets(datasets: Union[Dataset, List[Dataset]]) -> List[str]:
    """
    Returns the list of all targets present in a dataset or list of datasets.

    :param datasets: the dataset(s).
    :returns: list of targets present in the dataset(s), sorted according
        to the ``sort()`` method of Python lists.
    """

    if not isinstance(datasets, list):
        datasets = [datasets]

    # The following does not work because the `dataset` can also
    # be a `Subset` object:
    # return list(dataset.targets.keys())

    # Iterate over all single instances of the dataset:
    target_names = []
    for dataset in datasets:
        for sample in dataset:
            sample = sample._asdict()  # NamedTuple -> dict
            sample.pop("system")  # system not needed
            target_names += list(sample.keys())

    # Remove duplicates:
    result = list(set(target_names))
    result.sort()

    return result


def collate_fn(batch: List[NamedTuple]) -> Tuple[List, Dict[str, TensorMap]]:
    """
    Wraps the `metatensor-learn` default collate function `group_and_join` to
    return the data fields as a list of systems, and a dictionary of nameed
    targets.
    """

    collated_targets = group_and_join(batch)._asdict()
    systems = collated_targets.pop("system")
    return systems, collated_targets


def check_datasets(
    train_datasets: List[Dataset],
    validation_datasets: List[Dataset],
    raise_incompatibility_error: bool = True,
):
    """
    This is a helper function that checks that the training and validation sets
    are compatible with one another.

    Although these checks will not fit all use cases, most models would be expected
    to be able to use this function. If the validation set contains chemical species
    or targets that are not present in the training set, this function will raise a
    warning or an error, depending on the ``raise_incompatibility_error`` flag.

    The option to warn is intended for model fine tuning, where a species or target
    in the validation set might not be present in the current training set, but it
    might have been present in the training of the base model.

    :param train_datasets: A list of training datasets.
    :param validation_datasets: A list of validation datasets.
    :param raise_incompatibility_error: Whether to error (if ``true``) or warn
        (if ``false``) upon detection of a chemical species or target in the
        validation set that is not present in the training set.
    """

    # Get all targets in the training and validation sets:
    train_targets = get_all_targets(train_datasets)
    validation_targets = get_all_targets(validation_datasets)

    # Check that the validation sets do not have targets that are not in the
    # training sets:
    for target in validation_targets:
        if target not in train_targets:
            error_or_warning = f"The validation dataset has a target ({target}) "
            "that is not present in the training dataset."
            if raise_incompatibility_error:
                raise ValueError(error_or_warning)
            else:
                logger.warning(error_or_warning)

    # Get all the species in the training and validation sets:
    all_training_species = get_all_species(train_datasets)
    all_validation_species = get_all_species(validation_datasets)

    # Check that the validation sets do not have species that are not in the
    # training sets:
    for species in all_validation_species:
        if species not in all_training_species:

            error_or_warning = f"The validation dataset has a species ({species}) "
            "that is not in the training dataset. This could be "
            "a result of a random train/validation split. You can "
            "avoid this by providing a validation dataset manually."
            if raise_incompatibility_error:
                raise ValueError(error_or_warning)
            else:
                logger.warning(error_or_warning)


def _train_test_random_split(
    train_dataset: Dataset,
    train_size: float,
    test_size: float,
    generator: Optional[Generator] = default_generator,
) -> List[Subset]:
    if train_size <= 0:
        raise ValueError("Fraction of the train set is smaller or equal to 0!")

    # normalize fractions
    lengths = torch.tensor([train_size, test_size])
    lengths /= lengths.sum()

    return random_split(dataset=train_dataset, lengths=lengths, generator=generator)
