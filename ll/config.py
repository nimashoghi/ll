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


MISSING = cast(Any, _MISSING())

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
        self.__model_kwargs = kwargs
        self.__exit_stack = contextlib.ExitStack()
        self.__warning_list: list[warnings.WarningMessage] | None = None

    def build(self, config: TConfig) -> TConfig:
        return config.model_validate(
            config.model_dump(round_trip=True),
            strict=self.__strict,
        )

    __call__ = build

    @override
    def __enter__(self) -> tuple[Self, TConfig]:
        self.__warning_list = self.__exit_stack.enter_context(
            warnings.catch_warnings(record=True)
        )
        config = self.__config_cls.model_construct(**self.__model_kwargs)
        return self, config

    @override
    def __exit__(
        self,
        exc_type: type[BaseException],
        exc_value: BaseException,
        traceback: TracebackType,
    ):
        return_value = self.__exit_stack.__exit__(exc_type, exc_value, traceback)
        if warning_list := self.__warning_list:
            for warning in warning_list:
                if (
                    isinstance(warning.message, UserWarning)
                    and "pydantic" in warning.message.args[0].lower()
                ):
                    continue

                warnings.showwarning(
                    message=warning.message,
                    category=warning.category,
                    filename=warning.filename,
                    lineno=warning.lineno,
                    file=warning.file,
                    line=warning.line,
                )

        return return_value


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

        # This fixes the issue w/ `copy.deepcopy` not working properly when
        # the object was created using `cls.model_construct`.
        if not hasattr(self, "__pydantic_private__"):
            object.__setattr__(self, "__pydantic_private__", None)

        # Make sure there are no `MISSING` values in the config
        for key, value in self.model_dump().items():
            if value is MISSING:
                raise ValueError(
                    f"Config value for key '{key}' is `MISSING`.\n"
                    "Please provide a value for this key."
                )

    @classmethod  # pyright: ignore[reportArgumentType]
    def builder(cls: type[TConfig], /, strict: bool = True):
        return ConfigBuilder(cls, strict=strict)

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
