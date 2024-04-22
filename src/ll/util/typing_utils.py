from typing import TYPE_CHECKING

from typing_extensions import TypeVar

TBase = TypeVar("TBase")


def mixin_base_type(base_class: type[TBase]) -> type[TBase]:
    """
    Useful function to make mixins with baseclass typehint

    ```
    class ReadonlyMixin(mixin_base_type(BaseAdmin))):
        ...
    ```
    """
    if TYPE_CHECKING:
        return base_class
    return object
