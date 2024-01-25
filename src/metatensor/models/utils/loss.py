from typing import Dict, Optional, Tuple

import torch
from metatensor.torch import TensorMap


# This file defines losses for metatensor models.


class TensorMapLoss:
    """
    A loss function that operates on two `metatensor.torch.TensorMap`s.

    The loss is computed as the sum of the loss on the block values and
    the loss on the gradients, with weights specified at initialization.

    At the moment, this loss function assumes that all the gradients
    declared at initialization are present in both TensorMaps.

    :param reduction: The reduction to apply to the loss. See `torch.nn.MSELoss`.
    :param weight: The weight to apply to the loss on the block values.
    :param gradient_weights: The weights to apply to the loss on the gradients.

    :returns: The loss as a scalar `torch.Tensor`, as well as an information
        dictionary with the sum of squared errors and number of samples for values
        and each of the gradients.
    """

    def __init__(
        self,
        reduction: str = "mean",
        weight: float = 1.0,
        gradient_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Dict[str, Tuple[float, int]]]]:
        self.loss = torch.nn.MSELoss(reduction=reduction)
        self.weight = weight
        self.gradient_weights = {} if gradient_weights is None else gradient_weights

    def __call__(
        self, tensor_map_1: TensorMap, tensor_map_2: TensorMap
    ) -> torch.Tensor:
        # Check that the two have the same metadata, except for the samples,
        # which can be different due to batching, but must have the same size:
        if tensor_map_1.keys != tensor_map_2.keys:
            raise ValueError(
                "TensorMapLoss requires the two TensorMaps to have the same keys."
            )
        if tensor_map_1.block().properties != tensor_map_2.block().properties:
            raise ValueError(
                "TensorMapLoss requires the two TensorMaps to have the same properties."
            )
        if tensor_map_1.block().components != tensor_map_2.block().components:
            raise ValueError(
                "TensorMapLoss requires the two TensorMaps to have the same components."
            )
        if len(tensor_map_1.block().samples) != len(tensor_map_2.block().samples):
            raise ValueError(
                "TensorMapLoss requires the two TensorMaps "
                "to have the same number of samples."
            )
        for gradient_name in self.gradient_weights.keys():
            if len(tensor_map_1.block().gradient(gradient_name).samples) != len(
                tensor_map_2.block().gradient(gradient_name).samples
            ):
                raise ValueError(
                    "TensorMapLoss requires the two TensorMaps "
                    "to have the same number of gradient samples."
                )
            if (
                tensor_map_1.block().gradient(gradient_name).properties
                != tensor_map_2.block().gradient(gradient_name).properties
            ):
                raise ValueError(
                    "TensorMapLoss requires the two TensorMaps "
                    "to have the same gradient properties."
                )
            if (
                tensor_map_1.block().gradient(gradient_name).components
                != tensor_map_2.block().gradient(gradient_name).components
            ):
                raise ValueError(
                    "TensorMapLoss requires the two TensorMaps "
                    "to have the same gradient components."
                )

        # If the two TensorMaps have different symmetry keys:
        if len(tensor_map_1) != 1:
            raise NotImplementedError(
                "TensorMapLoss does not yet support multiple symmetry keys."
            )

        # Compute the loss and info:
        loss = torch.zeros(
            (),
            dtype=tensor_map_1.block().values.dtype,
            device=tensor_map_1.block().values.device,
        )
        info = {}

        values_1 = tensor_map_1.block().values
        values_2 = tensor_map_2.block().values
        loss += self.weight * self.loss(values_1, values_2)
        info["values"] = (torch.sum((values_1 - values_2) ** 2).item(), values_1.numel())

        for gradient_name, gradient_weight in self.gradient_weights.items():
            values_1 = tensor_map_1.block().gradient(gradient_name).values
            values_2 = tensor_map_2.block().gradient(gradient_name).values
            loss += gradient_weight * self.loss(values_1, values_2)
            info[gradient_name] = (
                torch.sum((values_1 - values_2) ** 2).item(),
                values_1.numel(),
            )
        
        return loss, info


class TensorMapDictLoss:
    """
    A loss function that operates on two `Dict[str, metatensor.torch.TensorMap]`.

    At initialization, the user specifies a list of keys to use for the loss,
    along with a weight for each key (as well as gradient weights).

    The loss is then computed as a weighted sum. Any keys that are not present
    in the dictionaries are ignored.

    :param weights: A dictionary mapping keys to weights. Each weight is itself
        a dictionary mapping "values" to the weight to apply to the loss on the
        block values, and gradient names to the weights to apply to the loss on
        the gradients.
    :param reduction: The reduction to apply to the loss. See `torch.nn.MSELoss`.

    :returns: The loss as a scalar `torch.Tensor`, as well as an information
        dictionary with the sum of squared errors and number of samples for values
        and each of the gradients.
    """

    def __init__(
        self,
        weights: Dict[str, Dict[str, float]],
        reduction: str = "mean",
    ):
        self.losses = {}
        for key, weight in weights.items():
            # Remove the value weight from the gradient weights and store it separately:
            value_weight = weight.pop("values")
            # Define the loss relative to this key:
            self.losses[key] = TensorMapLoss(
                reduction=reduction, weight=value_weight, gradient_weights=weight
            )

    def __call__(
        self,
        tensor_map_dict_1: Dict[str, TensorMap],
        tensor_map_dict_2: Dict[str, TensorMap],
    ) -> Tuple[torch.Tensor, Dict[str, Tuple[float, int]]]:
        
        # Assert that the two have the keys:
        assert set(tensor_map_dict_1.keys()) == set(tensor_map_dict_2.keys())

        # Initialize the loss:
        first_values = next(iter(tensor_map_dict_1.values())).block(0).values
        loss = torch.zeros((), dtype=first_values.dtype, device=first_values.device)
        info = {}

        # Compute the loss and associated info:
        for target in tensor_map_dict_1.keys():
            target_loss, target_info = self.losses[target](tensor_map_dict_1[target], tensor_map_dict_2[target])
            loss += target_loss
            info[target] = target_info["values"]
            for gradient_name in target_info.keys():
                if gradient_name != "values":
                    info[f"{target}_{gradient_name}_gradients"] = target_info[gradient_name]

        return loss, info
