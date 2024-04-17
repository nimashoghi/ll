from collections.abc import Iterable, Iterator
from typing import Generic, TypeVar, overload

import torch.nn as nn
from typing_extensions import override

TModule = TypeVar("TModule", bound=nn.Module)


class TypedModuleList(nn.ModuleList, Generic[TModule]):
    def __init__(self, modules: Iterable[TModule] | None = None) -> None:
        super().__init__(modules)

    @overload
    def __getitem__(self, idx: int) -> TModule: ...

    @overload
    def __getitem__(self, idx: slice) -> "TypedModuleList[TModule]": ...

    @override
    def __getitem__(self, idx: int | slice) -> TModule | "TypedModuleList[TModule]":
        return super().__getitem__(idx)  # type: ignore

    @override
    def __setitem__(self, idx: int, module: TModule) -> None:  # type: ignore
        return super().__setitem__(idx, module)

    @override
    def __iter__(self) -> Iterator[TModule]:
        return super().__iter__()  # type: ignore

    @override
    def __iadd__(self, modules: Iterable[TModule]) -> "TypedModuleList[TModule]":  # type: ignore
        return super().__iadd__(modules)  # type: ignore

    @override
    def __add__(self, modules: Iterable[TModule]) -> "TypedModuleList[TModule]":  # type: ignore
        return super().__add__(modules)  # type: ignore

    @override
    def insert(self, idx: int, module: TModule) -> None:  # type: ignore
        return super().insert(idx, module)

    @override
    def append(self, module: TModule) -> "TypedModuleList[TModule]":  # type: ignore
        return super().append(module)  # type: ignore

    @override
    def extend(self, modules: Iterable[TModule]) -> "TypedModuleList[TModule]":  # type: ignore
        return super().extend(modules)  # type: ignore
