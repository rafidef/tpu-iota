import base64
import json
from typing import Any
from pydantic import BaseModel


# Cannonisation helpers if we want to sign a json
def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj,
        default=_json_default,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def payload_base64_from_obj(obj: Any) -> str:
    return base64.b64encode(canonical_json_bytes(obj)).decode("ascii")


def _json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
