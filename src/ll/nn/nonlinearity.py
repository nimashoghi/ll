from abc import ABC, abstractclassmethod, abstractmethod
from typing import Annotated, Literal, TypeAlias

import torch.nn as nn
from typing_extensions import override

from ..config import Field, TypedConfig


class NonlinearityConfig(TypedConfig, ABC):
    @classmethod
    @abstractmethod
    def name(cls) -> str: ...

    @abstractmethod
    def create_module(self) -> nn.Module:
        pass


class ReLUNonlinearityConfig(NonlinearityConfig):
    @override
    @classmethod
    def name(cls):
        return "relu"

    @override
    def create_module(self) -> nn.Module:
        return nn.ReLU()


class SigmoidNonlinearityConfig(NonlinearityConfig):
    @override
    @classmethod
    def name(cls):
        return "sigmoid"

    @override
    def create_module(self) -> nn.Module:
        return nn.Sigmoid()


class TanhNonlinearityConfig(NonlinearityConfig):
    @override
    @classmethod
    def name(cls):
        return "tanh"

    @override
    def create_module(self) -> nn.Module:
        return nn.Tanh()


class SoftmaxNonlinearityConfig(NonlinearityConfig):
    @override
    @classmethod
    def name(cls):
        return "softmax"

    @override
    def create_module(self) -> nn.Module:
        return nn.Softmax(dim=1)


class SoftplusNonlinearityConfig(NonlinearityConfig):
    @override
    @classmethod
    def name(cls):
        return "softplus"

    @override
    def create_module(self) -> nn.Module:
        return nn.Softplus()


class SoftsignNonlinearityConfig(NonlinearityConfig):
    @override
    @classmethod
    def name(cls):
        return "softsign"

    @override
    def create_module(self) -> nn.Module:
        return nn.Softsign()


class ELUNonlinearityConfig(NonlinearityConfig):
    @override
    @classmethod
    def name(cls):
        return "elu"

    @override
    def create_module(self) -> nn.Module:
        return nn.ELU()


class LeakyReLUNonlinearityConfig(NonlinearityConfig):
    @override
    @classmethod
    def name(cls):
        return "leaky_relu"

    negative_slope: float | None = None

    @override
    def create_module(self) -> nn.Module:
        kwargs = {}
        if self.negative_slope is not None:
            kwargs["negative_slope"] = self.negative_slope
        return nn.LeakyReLU(**kwargs)


class PReLUConfig(NonlinearityConfig):
    @override
    @classmethod
    def name(cls):
        return "prelu"

    @override
    def create_module(self) -> nn.Module:
        return nn.PReLU()


class GELUNonlinearityConfig(NonlinearityConfig):
    @override
    @classmethod
    def name(cls):
        return "gelu"

    @override
    def create_module(self) -> nn.Module:
        return nn.GELU()


class SwishNonlinearityConfig(NonlinearityConfig):
    @override
    @classmethod
    def name(cls):
        return "swish"

    @override
    def create_module(self) -> nn.Module:
        return nn.SiLU()


class SiLUNonlinearityConfig(NonlinearityConfig):
    @override
    @classmethod
    def name(cls):
        return "silu"

    @override
    def create_module(self) -> nn.Module:
        return nn.SiLU()


class MishNonlinearityConfig(NonlinearityConfig):
    @override
    @classmethod
    def name(cls):
        return "mish"

    @override
    def create_module(self) -> nn.Module:
        return nn.Mish()
