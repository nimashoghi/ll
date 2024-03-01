from collections.abc import Sequence
from logging import getLogger
from typing import Any

import torch
from jaxtyping._storage import get_shape_memo, shape_str
from lovely_tensors import lovely
from typing_extensions import TypeVar

log = getLogger(__name__)


def typecheck_modules(modules: Sequence[str]):
    """
    Typecheck the given modules using `jaxtyping`.

    Args:
        modules: Modules to typecheck.
    """
    # Install the jaxtyping import hook for this module.
    from jaxtyping import install_import_hook

    install_import_hook(modules, "beartype.beartype")

    log.critical(f"Type checking the following modules: {modules}")


def typecheck_this_module(
    additional_modules: Sequence[str] = (),
):
    """
    Typecheck the calling module and any additional modules using `jaxtyping`.

    Args:
        additional_modules: Additional modules to typecheck.
    """
    # Get the calling module's name.
    # Here, we can just use beartype's internal implementation behind
    # `beartype_this_package`.
    from beartype._util.func.utilfuncframe import get_frame, get_frame_package_name

    # Get the calling module's name.
    assert get_frame is not None, "get_frame is None"
    frame = get_frame(1)
    assert frame is not None, "frame is None"
    calling_module_name = get_frame_package_name(frame)

    # Typecheck the calling module + any additional modules.
    typecheck_modules((calling_module_name, *additional_modules))


def _make_error_str(input: Any, t: Any) -> str:
    error_components: list[str] = []
    error_components.append("Type checking error:")
    if hasattr(t, "__instancecheck_str__"):
        error_components.append(t.__instancecheck_str__(input))
    if torch.is_tensor(input):
        error_components.append(repr(lovely(input)))
    error_components.append(shape_str(get_shape_memo()))

    return "\n".join(error_components)


T = TypeVar("T", infer_variance=True)


def tassert(t: Any, input: T) -> T:
    """
    Typecheck the input against the given type.

    Args:
        t: Type to check against.
        input: Input to check.
    """

    assert isinstance(input, t), _make_error_str(input, t)
    return input
