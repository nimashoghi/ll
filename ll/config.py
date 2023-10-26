import warnings
from abc import ABC
from logging import getLogger
from typing import TYPE_CHECKING, Any, Mapping, MutableMapping, Sequence, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self, dataclass_transform, override

log = getLogger(__name__)


class _MISSING:
    pass


MISSING = cast(Any, _MISSING)


_ModelBase = BaseModel
if TYPE_CHECKING:
    _ModelBase = ABC

_MutableMappingBase = MutableMapping[str, Any]
if TYPE_CHECKING:
    _MutableMappingBase = object


def _validate_no_missing(value: Any, labels: list[str]):
    """Makes sure that no MISSING values are present"""

    match value:
        # If the value is a Mapping, then validate all the values
        case Mapping():
            for k, v in value.items():
                _validate_no_missing(v, labels + [k])
        # If the value is a list, then validate all the values
        case Sequence():
            for i, v in enumerate(value):
                _validate_no_missing(v, labels + [str(i)])
        # If the value is a BaseModel, then validate all the fields
        case BaseModel():
            for field_name, field in value.model_fields.items():
                if (value := getattr(value, field_name, None)) is None:
                    continue
                _validate_no_missing(value, labels + [field_name])
        # If the value is MISSING, then raise an error
        case value if value is MISSING:
            raise ValueError("MISSING is not allowed")
        # Otherwise: do nothing
        case _:
            pass


@dataclass_transform(kw_only_default=True)
class TypedConfig(_ModelBase, _MutableMappingBase):
    model_config = ConfigDict(
        # By default, Pydantic will throw a warning if a field starts with "model_",
        # so we need to disable that warning (beacuse "model_" is a popular prefix for ML).
        protected_namespaces=(),
        validate_assignment=True,
    )

    if not TYPE_CHECKING:
        _ll_builder: bool = False

    # region Post-Init
    def __post_init__(self):
        # Override this method to perform any post-initialization
        # actions on the model.
        pass

    def model_post_init(self, __context: Any):
        # Ignore if this is a builder
        if self._ll_builder:  # type: ignore
            return

        self.__post_init__()

    # endregion

    @classmethod
    @property
    def _as_pydantic_model_cls(cls):
        return cast(BaseModel, cls)

    @property
    def _as_pydantic_model(self):
        return cast(BaseModel, self)

    # region construction methods
    @classmethod
    def from_dict(cls, model_dict: Mapping[str, Any]):
        return cast(Self, cls._as_pydantic_model_cls.model_validate(model_dict))

    @classmethod
    def builder(cls, **fields):
        builder = cast(
            cls,
            cls._as_pydantic_model_cls.model_construct(_ll_builder=True, **fields),
        )
        return builder

    def build(self):
        model_dict = self._as_pydantic_model.model_dump(round_trip=True)
        instance = cast(
            Self,
            self._as_pydantic_model_cls.model_validate(model_dict),
        )
        return instance

    def validate_no_missing(self):
        """Makes sure that no MISSING values are present"""

        _validate_no_missing(self, [])

    def validate(
        self,
        strict: bool = True,
        check_missing: bool = True,
    ):
        # Make sure this is not a builder
        if self._ll_builder:  # type: ignore
            raise ValueError("A builder cannot be used as a config.")

        if check_missing:
            self.validate_no_missing()

        # Validate the model by dumping it and then loading it
        model_dict = self._as_pydantic_model.model_dump(round_trip=True)
        _ = self._as_pydantic_model.model_validate(model_dict, strict=strict)

    # endregion

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


__all__ = [
    "MISSING",
    "Field",
    "TypedConfig",
    "field_validator",
    "model_validator",
]
