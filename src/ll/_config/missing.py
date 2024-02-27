from dataclasses import dataclass
import types
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Never,
    TypeAlias,
    cast,
    get_args,
    get_origin,
)

from pydantic import BaseModel, GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic_core import core_schema
from typing_extensions import TypeVar

MISSING = cast(Any, None)


@dataclass
class _MissingType:
    pass


class _MissingTypeAnnotation:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        return core_schema.json_or_python_schema(
            json_schema=core_schema.literal_schema([MISSING]),
            python_schema=core_schema.literal_schema([MISSING]),
            serialization=core_schema.plain_serializer_function_ser_schema(lambda x: x),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        _core_schema: core_schema.CoreSchema,
        handler: GetJsonSchemaHandler,
    ):
        # Use the same schema that would be used for `int`
        return handler(core_schema.literal_schema([MISSING]))


T = TypeVar("T", infer_variance=True)
if TYPE_CHECKING:
    AllowMissing: TypeAlias = T | Annotated[Never, _MissingTypeAnnotation()]
else:
    AllowMissing: TypeAlias = T | Annotated[types.NoneType, _MissingTypeAnnotation()]


def _validate_field(annotation: Any):
    # Resolve the origin and args of the field's annotation
    # (e.g. `List[int]` -> origin=`List`, args=`(int,)`)
    if (origin := get_origin(annotation)) is None:
        return
    args = get_args(annotation)

    match origin, args:
        case _:
            pass


def validate_no_missing_values(model: BaseModel):
    return
    for name, field in model.model_fields.items():
        _validate_field(field.annotation)
