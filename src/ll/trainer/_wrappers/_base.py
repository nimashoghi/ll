from abc import ABC, abstractmethod
from logging import getLogger
from typing import Any, Literal, Protocol, runtime_checkable

from lightning.pytorch import LightningModule
from typing_extensions import TypeAlias

log = getLogger(__name__)


MethodName: TypeAlias = Literal[
    "training_step",
    "validation_step",
    "test_step",
    "predict_step",
]
ALL_METHODS: list[MethodName] = [
    "training_step",
    "validation_step",
    "test_step",
    "predict_step",
]


@runtime_checkable
class StepFunction(Protocol):
    def __call__(
        self,
        batch: Any,
        batch_idx: int,
        *args,
        **kwargs,
    ) -> Any: ...


class LightningModuleWrapper(ABC):
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def wrapped_step(
        self,
        module: LightningModule,
        fn: StepFunction,
        fn_name: MethodName,
        batch: Any,
        batch_idx: int,
        *args,
        **kwargs,
    ) -> Any: ...

    def _wrap(self, module: LightningModule, fn_name: MethodName):
        old_step_fn = getattr(module, fn_name)

        def new_step(module: LightningModule, batch, batch_idx, *args, **kwargs):
            nonlocal self, old_step_fn, fn_name
            return self.wrapped_step(
                module,
                old_step_fn,
                fn_name,
                batch,
                batch_idx,
                *args,
                **kwargs,
            )

        setattr(module, fn_name, new_step.__get__(module))

    def wrap_lightning_module_methods(
        self,
        module: LightningModule,
        methods: list[MethodName] = ALL_METHODS,
    ):
        name = self.name()
        methods_str = ", ".join(methods)

        log.info(f"Wrapping {methods_str} for {name}")
        for method in methods:
            self._wrap(module, method)
        log.info(f"Wrapped {methods_str} for {name}")

    @classmethod
    def wrap_lightning_module(
        cls,
        module: LightningModule,
        methods: list[MethodName] = ALL_METHODS,
    ):
        wrapper = cls()
        wrapper.wrap_lightning_module_methods(module, methods)
        return wrapper
