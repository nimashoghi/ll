from collections.abc import Iterable
from typing import Annotated, Any, Literal, TypeAlias

import torch.nn as nn

from .config import Field, TypedConfig


class AdamWConfig(TypedConfig):
    name: Literal["adamw"] = "adamw"

    lr: float
    """Learning rate for the optimizer."""

    weight_decay: float = 1.0e-2
    """Weight decay (L2 penalty) for the optimizer."""

    betas: tuple[float, float] = (0.9, 0.999)
    """
    Betas for the optimizer:
    (beta1, beta2) are the coefficients used for computing running averages of
    gradient and its square.
    """

    eps: float = 1e-8
    """Term added to the denominator to improve numerical stability."""

    amsgrad: bool = False
    """Whether to use the AMSGrad variant of this algorithm."""

    def create_optimizer(
        self,
        parameters: Iterable[nn.Parameter] | Iterable[dict[str, Any]],
    ):
        from torch.optim import AdamW

        return AdamW(
            parameters,
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.betas,
            eps=self.eps,
            amsgrad=self.amsgrad,
        )


OptimizerConfig: TypeAlias = Annotated[
    AdamWConfig,
    Field(discriminator="name"),
]
