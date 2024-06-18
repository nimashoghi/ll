import copy
from collections.abc import Callable
from typing import Any, cast

from typing_extensions import TypeVar

TDataset = TypeVar("TDataset", infer_variance=True)


def transform(
    dataset: TDataset,
    transform: Callable[[Any], Any],
    *,
    deepcopy: bool = False,
) -> TDataset:
    """
    Wraps a dataset with a transform function.

    Args:
        dataset: The dataset to wrap.
        transform: The transform function to apply to each item.
        deepcopy: Whether to deep copy each item before applying the transform.
    """

    import wrapt

    class _TransformedDataset(wrapt.ObjectProxy):
        def __getitem__(self, idx):
            nonlocal deepcopy, transform

            data = self.__wrapped__.__getitem__(idx)
            if deepcopy:
                data = copy.deepcopy(data)
            data = transform(data)
            return data

    return cast(TDataset, _TransformedDataset(dataset))


def transform_with_index(
    dataset: TDataset,
    transform: Callable[[Any, int], Any],
    *,
    deepcopy: bool = False,
) -> TDataset:
    """
    Wraps a dataset with a transform function that takes an index, in addition to the item.

    Args:
        dataset: The dataset to wrap.
        transform: The transform function to apply to each item.
        deepcopy: Whether to deep copy each item before applying the transform.
    """

    import wrapt

    class _TransformedWithIndexDataset(wrapt.ObjectProxy):
        def __getitem__(self, idx: int):
            nonlocal deepcopy, transform

            data = self.__wrapped__.__getitem__(idx)
            if deepcopy:
                data = copy.deepcopy(data)
            data = transform(data, idx)
            return data

    return cast(TDataset, _TransformedWithIndexDataset(dataset))
