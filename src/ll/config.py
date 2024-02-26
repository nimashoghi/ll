from collections.abc import Mapping, MutableMapping
from typing import TYPE_CHECKING, Any, ClassVar, cast

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from typing_extensions import override

_MutableMappingBase = MutableMapping[str, Any]
if TYPE_CHECKING:
    _MutableMappingBase = object


class _MISSING:
    pass


MISSING = cast(Any, _MISSING())


class TypedConfig(BaseModel, _MutableMappingBase):
    _is_draft_config: bool = PrivateAttr(default=False)
    """
    Whether this config is a draft config or not.

    Draft configs are configs that are not yet fully validated.
    They allow for a nicer API when creating configs, e.g.:

        ```python
        config = MyConfig.draft()

        # Set some values
        config.a = 10
        config.b = "hello"

        # Finalize the config
        config = config.finalize()
        ```
    """

    MISSING: ClassVar[Any] = MISSING

    model_config: ClassVar[ConfigDict] = ConfigDict(
        # By default, Pydantic will throw a warning if a field starts with "model_",
        # so we need to disable that warning (beacuse "model_" is a popular prefix for ML).
        protected_namespaces=(),
        validate_assignment=True,
        strict=True,
        revalidate_instances="always",
        arbitrary_types_allowed=True,
    )

    def __draft_pre_init__(self):
        """Called right before a draft config is finalized."""
        pass

    def __post_init__(self):
        """Called after the final config is validated."""
        pass

    @classmethod
    def from_dict(cls, model_dict: Mapping[str, Any]):
        return cls.model_validate(model_dict)

    def model_deep_validate(self, strict: bool = True):
        """
        Validate the config and all of its sub-configs.

        Args:
            config: The config to validate.
            strict: Whether to validate the config strictly.
        """
        config = self.model_validate(self.model_dump(round_trip=True), strict=strict)

        # Make sure that this is not a draft config
        if config._is_draft_config:
            raise ValueError("Draft configs are not valid. Call `finalize` first.")

        return config

    @classmethod
    def draft(cls, **kwargs):
        config = cls.model_construct(_is_draft_config=True, **kwargs)
        config._is_draft_config = True
        return config

    def finalize(self, strict: bool = True):
        # This must be a draft config, otherwise we raise an error
        if not self._is_draft_config:
            raise ValueError("Finalize can only be called on drafts.")

        # First, we call `__draft_pre_init__` to allow the config to modify itself a final time
        self.__draft_pre_init__()

        # Then, we dump the config to a dict and then re-validate it
        config_dict = self.model_dump(round_trip=True)

        # We need to remove the `_is_draft_config` from the config_dict
        #   because we're no longer a draft self
        _ = config_dict.pop("_is_draft_config", None)

        return self.model_validate(config_dict, strict=strict)

    @override
    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)

        # This fixes the issue w/ `copy.deepcopy` not working properly when
        #   the object was created using `cls.model_construct`.
        if not hasattr(self, "__pydantic_private__"):
            object.__setattr__(self, "__pydantic_private__", None)

        # If we're not in a draft, call __post_init__
        if not self._is_draft_config:
            self.__post_init__()

    # region MutableMapping implementation
    if not TYPE_CHECKING:
        # This is mainly so the config can be used with lightning's hparams
        #   transparently and without any issues.

        @property
        def _ll_dict(self):
            return self.model_dump()

        # We need to make sure every config class
        #   is a MutableMapping[str, Any] so that it can be used
        #   with lightning's hparams.
        @override
        def __getitem__(self, key: str):
            # Key can be of the format "a.b.c"
            #   so we need to split it into a list of keys.
            [first_key, *rest_keys] = key.split(".")
            value = self._ll_dict[first_key]

            for key in rest_keys:
                if isinstance(value, Mapping):
                    value = value[key]
                else:
                    value = getattr(value, key)

            return value

        @override
        def __setitem__(self, key: str, value: Any):
            # Key can be of the format "a.b.c"
            #   so we need to split it into a list of keys.
            [first_key, *rest_keys] = key.split(".")
            if len(rest_keys) == 0:
                self._ll_dict[first_key] = value
                return

            # We need to traverse the keys until we reach the last key
            #   and then set the value
            current_value = self._ll_dict[first_key]
            for key in rest_keys[:-1]:
                if isinstance(current_value, Mapping):
                    current_value = current_value[key]
                else:
                    current_value = getattr(current_value, key)

            # Set the value
            if isinstance(current_value, MutableMapping):
                current_value[rest_keys[-1]] = value
            else:
                setattr(current_value, rest_keys[-1], value)

        @override
        def __delitem__(self, key: str):
            # This is unsupported for this class
            raise NotImplementedError

        @override
        def __iter__(self):
            return iter(self._ll_dict)

        @override
        def __len__(self):
            return len(self._ll_dict)

    # endregion


__all__ = ["TypedConfig", "Field"]
