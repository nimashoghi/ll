import contextlib
import warnings
from logging import getLogger
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Generic,
    Mapping,
    MutableMapping,
    cast,
)

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import ParamSpec, Self, TypeVar, deprecated, override

log = getLogger(__name__)


class _MISSING:
    pass


MISSING = cast(Any, _MISSING)

TConfig = TypeVar("TConfig", bound=BaseModel)
P = ParamSpec("P")


class ConfigBuilder(contextlib.AbstractContextManager, Generic[TConfig]):
    def __init__(
        self,
        config_cls: Callable[P, TConfig],
        /,
        strict: bool = True,
        *_args: P.args,
        **kwargs: P.kwargs,
    ):
        assert isinstance(config_cls, type), "config_cls must be a class"
        assert not len(
            _args
        ), f"Only keyword arguments are supported for config classes. Got {_args=}."

        self.__config_cls = cast(type[TConfig], config_cls)
        self.__strict = strict
        self.__built_config: TConfig | None = None

        self.config = self.__config_cls.model_construct(**kwargs)

    def __build_if_needed(self) -> TConfig:
        if self.__built_config is None:
            self.__built_config = self.__config_cls.model_validate(
                self.config,
                strict=self.__strict,
            )

        return self.__built_config

    def build(self):
        return self.__build_if_needed()

    __call__ = build

    @override
    def __enter__(self) -> Self:
        return self

    @override
    def __exit__(
        self,
        exc_type: type[BaseException],
        exc_value: BaseException,
        traceback: TracebackType,
    ) -> None:
        _ = self.__build_if_needed()


_MutableMappingBase = MutableMapping[str, Any]
if TYPE_CHECKING:
    _MutableMappingBase = object


class TypedConfig(BaseModel, _MutableMappingBase):
    MISSING: ClassVar[Any] = MISSING

    model_config = ConfigDict(
        # By default, Pydantic will throw a warning if a field starts with "model_",
        # so we need to disable that warning (beacuse "model_" is a popular prefix for ML).
        protected_namespaces=(),
        validate_assignment=True,
        strict=True,
        revalidate_instances="always",
    )

    @override
    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)

        # Make sure there are no `MISSING` values in the config
        for key, value in self.model_dump().items():
            if value is MISSING:
                raise ValueError(
                    f"Config value for key '{key}' is `MISSING`.\n"
                    "Please provide a value for this key."
                )

    @classmethod
    def builder(cls, strict: bool = True, **kwargs: Any) -> ConfigBuilder[Self]:
        return ConfigBuilder(cls, strict=strict, **kwargs)

    # region MutableMapping implementation
    # These are under `if not TYPE_CHECKING` to prevent vscode from showing
    # all the MutableMapping methods in the editor
    if not TYPE_CHECKING:

        @property
        def _ll_dict(self):
            return self.model_dump()

        # we need to make sure every config class
        # is a MutableMapping[str, Any] so that it can be used
        # with lightning's hparams
        def __getitem__(self, key: str):
            # key can be of the format "a.b.c"
            # so we need to split it into a list of keys
            [first_key, *rest_keys] = key.split(".")
            value = self._ll_dict[first_key]

            for key in rest_keys:
                if isinstance(value, Mapping):
                    value = value[key]
                else:
                    value = getattr(value, key)

            return value

        def __setitem__(self, key: str, value: Any):
            # key can be of the format "a.b.c"
            # so we need to split it into a list of keys
            [first_key, *rest_keys] = key.split(".")
            if len(rest_keys) == 0:
                self._ll_dict[first_key] = value
                return

            # we need to traverse the keys until we reach the last key
            # and then set the value
            current_value = self._ll_dict[first_key]
            for key in rest_keys[:-1]:
                if isinstance(current_value, Mapping):
                    current_value = current_value[key]
                else:
                    current_value = getattr(current_value, key)

            # set the value
            if isinstance(current_value, MutableMapping):
                current_value[rest_keys[-1]] = value
            else:
                setattr(current_value, rest_keys[-1], value)

        def __delitem__(self, key: str):
            # this is unsupported for this class
            raise NotImplementedError

        @override
        def __iter__(self):
            return iter(self._ll_dict)

        def __len__(self):
            return len(self._ll_dict)

    # endregion

    @deprecated("No longer supported, use rich library instead.")
    def pprint(self):
        try:
            from rich import print as rprint
        except ImportError:
            warnings.warn(
                "rich is not installed, falling back to default print function"
            )
            print(self)
        else:
            rprint(self)


__all__ = [
    "MISSING",
    "Field",
    "ConfigBuilder",
    "TypedConfig",
]
