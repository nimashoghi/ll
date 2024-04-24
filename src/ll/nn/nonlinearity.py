from abc import ABC, abstractmethod
from typing import Annotated, Literal

import torch
import torch.nn as nn
from typing_extensions import override

from ..config import Field, TypedConfig


class BaseNonlinearityConfig(TypedConfig, ABC):
    @abstractmethod
    def create_module(self) -> nn.Module:
        pass


class ReLUNonlinearityConfig(BaseNonlinearityConfig):
    name: Literal["relu"] = "relu"

    @override
    def create_module(self) -> nn.Module:
        return nn.ReLU()


class SigmoidNonlinearityConfig(BaseNonlinearityConfig):
    name: Literal["sigmoid"] = "sigmoid"

    @override
    def create_module(self) -> nn.Module:
        return nn.Sigmoid()


class TanhNonlinearityConfig(BaseNonlinearityConfig):
    name: Literal["tanh"] = "tanh"

    @override
    def create_module(self) -> nn.Module:
        return nn.Tanh()


class SoftmaxNonlinearityConfig(BaseNonlinearityConfig):
    name: Literal["softmax"] = "softmax"

    @override
    def create_module(self) -> nn.Module:
        return nn.Softmax(dim=1)


class SoftplusNonlinearityConfig(BaseNonlinearityConfig):
    name: Literal["softplus"] = "softplus"

    @override
    def create_module(self) -> nn.Module:
        return nn.Softplus()


class SoftsignNonlinearityConfig(BaseNonlinearityConfig):
    name: Literal["softsign"] = "softsign"

    @override
    def create_module(self) -> nn.Module:
        return nn.Softsign()


class ELUNonlinearityConfig(BaseNonlinearityConfig):
    name: Literal["elu"] = "elu"

    @override
    def create_module(self) -> nn.Module:
        return nn.ELU()


class LeakyReLUNonlinearityConfig(BaseNonlinearityConfig):
    name: Literal["leaky_relu"] = "leaky_relu"

    negative_slope: float | None = None

    @override
    def create_module(self) -> nn.Module:
        kwargs = {}
        if self.negative_slope is not None:
            kwargs["negative_slope"] = self.negative_slope
        return nn.LeakyReLU(**kwargs)


class PReLUConfig(BaseNonlinearityConfig):
    name: Literal["prelu"] = "prelu"

    @override
    def create_module(self) -> nn.Module:
        return nn.PReLU()


class GELUNonlinearityConfig(BaseNonlinearityConfig):
    name: Literal["gelu"] = "gelu"

    @override
    def create_module(self) -> nn.Module:
        return nn.GELU()


class SwishNonlinearityConfig(BaseNonlinearityConfig):
    name: Literal["swish"] = "swish"

    @override
    def create_module(self) -> nn.Module:
        return nn.SiLU()


class SiLUNonlinearityConfig(BaseNonlinearityConfig):
    name: Literal["silu"] = "silu"

    @override
    def create_module(self) -> nn.Module:
        return nn.SiLU()


class MishNonlinearityConfig(BaseNonlinearityConfig):
    name: Literal["mish"] = "mish"

    @override
    def create_module(self) -> nn.Module:
        return nn.Mish()


class SwiGLU(nn.SiLU):
    @override
    def forward(self, input: torch.Tensor):
        input, gate = input.chunk(2, dim=-1)
        return input * super().forward(gate)


class SwiGLUNonlinearityConfig(BaseNonlinearityConfig):
    name: Literal["swiglu"] = "swiglu"

    @override
    def create_module(self) -> nn.Module:
        return SwiGLU()


NonlinearityConfig = Annotated[
    ReLUNonlinearityConfig
    | SigmoidNonlinearityConfig
    | TanhNonlinearityConfig
    | SoftmaxNonlinearityConfig
    | SoftplusNonlinearityConfig
    | SoftsignNonlinearityConfig
    | ELUNonlinearityConfig
    | LeakyReLUNonlinearityConfig
    | PReLUConfig
    | GELUNonlinearityConfig
    | SwishNonlinearityConfig
    | SiLUNonlinearityConfig
    | MishNonlinearityConfig
    | SwiGLUNonlinearityConfig,
    Field(discriminator="name"),
]
