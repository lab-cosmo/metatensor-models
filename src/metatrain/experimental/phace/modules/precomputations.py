# import sphericart.torch
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
import math


def _spherical_harmonics(lmax: int, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sh_0_0 = torch.ones_like(x)
    if lmax == 0:
        return torch.stack([
            sh_0_0,
        ], dim=-1)

    sh_1_0 = x
    sh_1_1 = y
    sh_1_2 = z
    if lmax == 1:
        return torch.stack([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2
        ], dim=-1)

    sh_2_0 = math.sqrt(3.0) * x * z
    sh_2_1 = math.sqrt(3.0) * x * y
    y2 = y.pow(2)
    x2z2 = x.pow(2) + z.pow(2)
    sh_2_2 = y2 - 0.5 * x2z2
    sh_2_3 = math.sqrt(3.0) * y * z
    sh_2_4 = math.sqrt(3.0) / 2.0 * (z.pow(2) - x.pow(2))

    if lmax == 2:
        return torch.stack([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4
        ], dim=-1)

    sh_3_0 = math.sqrt(5.0 / 6.0) * (sh_2_0 * z + sh_2_4 * x)
    sh_3_1 = math.sqrt(5.0) * sh_2_0 * y
    sh_3_2 = math.sqrt(3.0 / 8.0) * (4.0 * y2 - x2z2) * x
    sh_3_3 = 0.5 * y * (2.0 * y2 - 3.0 * x2z2)
    sh_3_4 = math.sqrt(3.0 / 8.0) * z * (4.0 * y2 - x2z2)
    sh_3_5 = math.sqrt(5.0) * sh_2_4 * y
    sh_3_6 = math.sqrt(5.0 / 6.0) * (sh_2_4 * z - sh_2_0 * x)

    if lmax == 3:
        return torch.stack([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6
        ], dim=-1)
    
    raise ValueError(f"lmax={lmax} not supported")


class Precomputer(torch.nn.Module):
    def __init__(self, l_max, normalize=True):
        super().__init__()
        self.spherical_harmonics_split_list = [(2 * l + 1) for l in range(l_max + 1)]
        self.normalize = normalize
        # self.spherical_harmonics_calculator = sphericart.torch.SphericalHarmonics(
        #     l_max, normalized=True
        # )
        self.l_max = l_max

    def forward(
        self,
        positions,
        cells,
        species,
        cell_shifts,
        pairs,
        structure_pairs,
        structure_offsets,
    ):
        cartesian_vectors = get_cartesian_vectors(
            positions,
            cells,
            species,
            cell_shifts,
            pairs,
            structure_pairs,
            structure_offsets,
        )

        bare_cartesian_vectors = cartesian_vectors.values.squeeze(dim=-1)
        r = torch.sqrt((bare_cartesian_vectors**2).sum(dim=-1))

        xyz_normalized = bare_cartesian_vectors / r.unsqueeze(-1)
        spherical_harmonics = _spherical_harmonics(self.l_max, xyz_normalized[..., 1], xyz_normalized[..., 2], xyz_normalized[..., 0])

        # spherical_harmonics = self.spherical_harmonics_calculator.compute(
        #     bare_cartesian_vectors
        # )  # Get the spherical harmonics
        if self.normalize:
            spherical_harmonics = spherical_harmonics * (4.0 * torch.pi) ** (
                0.5
            )  # normalize them
        spherical_harmonics = torch.split(
            spherical_harmonics, self.spherical_harmonics_split_list, dim=1
        )  # Split them into l chunks

        spherical_harmonics_blocks = [
            TensorBlock(
                values=spherical_harmonics_l.unsqueeze(-1),
                samples=cartesian_vectors.samples,
                components=[
                    Labels(
                        names=("o3_mu",),
                        values=torch.arange(
                            start=-l, end=l + 1, dtype=torch.int32
                        ).reshape(2 * l + 1, 1),
                    ).to(device=cartesian_vectors.values.device)
                ],
                properties=Labels(
                    names=["_"],
                    values=torch.zeros(
                        1, 1, dtype=torch.int32, device=cartesian_vectors.values.device
                    ),
                ),
            )
            for l, spherical_harmonics_l in enumerate(spherical_harmonics)
        ]
        spherical_harmonics_map = TensorMap(
            keys=Labels(
                names=["o3_lambda"],
                values=torch.arange(
                    len(spherical_harmonics_blocks), device=r.device
                ).reshape(len(spherical_harmonics_blocks), 1),
            ),
            blocks=spherical_harmonics_blocks,
        )

        r_block = TensorBlock(
            values=r.unsqueeze(-1),
            samples=cartesian_vectors.samples,
            components=[],
            properties=Labels(
                names=["_"],
                values=torch.zeros(1, 1, dtype=torch.int32, device=r.device),
            ),
        )

        return r_block, spherical_harmonics_map


def get_cartesian_vectors(
    positions, cells, species, cell_shifts, pairs, structure_pairs, structure_offsets
):
    """
    Wraps direction vectors into TensorBlock object with metadata information
    """

    # calculate interatomic vectors
    pairs_offsets = structure_offsets[structure_pairs]
    shifted_pairs = pairs_offsets[:, None] + pairs
    shifted_pairs_i = shifted_pairs[:, 0]
    shifted_pairs_j = shifted_pairs[:, 1]
    direction_vectors = (
        positions[shifted_pairs_j]
        - positions[shifted_pairs_i]
        + torch.einsum(
            "ab, abc -> ac", cell_shifts.to(cells.dtype), cells[structure_pairs]
        )
    )

    # find associated metadata
    pairs_i = pairs[:, 0]
    pairs_j = pairs[:, 1]
    labels = torch.stack(
        [
            structure_pairs,
            pairs_i,
            pairs_j,
            species[shifted_pairs_i],
            species[shifted_pairs_j],
            cell_shifts[:, 0],
            cell_shifts[:, 1],
            cell_shifts[:, 2],
        ],
        dim=-1,
    )

    # build TensorBlock
    block = TensorBlock(
        values=direction_vectors.unsqueeze(dim=-1),
        samples=Labels(
            names=[
                "structure",
                "center",
                "neighbor",
                "species_center",
                "species_neighbor",
                "cell_x",
                "cell_y",
                "cell_z",
            ],
            values=labels,
        ),
        components=[
            Labels(
                names=["cartesian_dimension"],
                values=torch.tensor([-1, 0, 1], dtype=torch.int32).reshape((-1, 1)),
            ).to(device=direction_vectors.device)
        ],
        properties=Labels(
            names=["_"],
            values=torch.zeros(
                1, 1, dtype=torch.int32, device=direction_vectors.device
            ),
        ),
    )

    return block
