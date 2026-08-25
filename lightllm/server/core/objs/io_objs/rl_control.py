from dataclasses import dataclass, field
from typing import Any


@dataclass
class RLControlRequest:
    op_id: str
    operation: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class RLControlResponse:
    op_id: str
    consumer: str
    success: bool
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
