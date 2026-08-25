from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .api_models import Message


class InitWeightsUpdateGroupRequest(BaseModel):
    master_address: str
    master_port: int = Field(gt=0, lt=65536)
    world_size: int = Field(gt=1)
    group_name: str = "weight_update_group"
    backend: str = "nccl"
    master_ports: dict[Literal["language", "vision", "x2v"], int] | None = None


class DistributedWeightsRequest(BaseModel):
    names: list[str]
    dtypes: list[str]
    shapes: list[list[int]]
    checksums: list[str]
    assignments: dict[str, list[Literal["language", "vision", "x2v"]]] = Field(default_factory=dict)
    required: dict[str, list[str]] = Field(default_factory=dict)
    policy_version: str
    group_name: str = "weight_update_group"
    buckets: list[dict] | None = None

    @model_validator(mode="after")
    def validate_columns(self):
        size = len(self.names)
        if not size or not (size == len(self.dtypes) == len(self.shapes) == len(self.checksums)):
            raise ValueError("weight manifest columns must have the same non-zero length")
        if len(set(self.names)) != size:
            raise ValueError("weight manifest contains duplicate names")
        return self


class DestroyWeightsUpdateGroupRequest(BaseModel):
    group_name: str = "weight_update_group"


class TensorWeightsRequest(BaseModel):
    serialized_safetensors: str
    checksums: dict[str, str]
    assignments: dict[str, list[Literal["language", "vision", "x2v"]]] = Field(default_factory=dict)
    required: dict[str, list[str]] = Field(default_factory=dict)
    policy_version: str


class ImagePolicyConfig(BaseModel):
    image_size: str | None = None
    height: int | None = Field(default=None, gt=0)
    width: int | None = Field(default=None, gt=0)
    image_steps: int = Field(default=50, gt=0)
    timestep_shift: float = 3.0
    t_eps: float = 0.02
    image_noise_level: float = Field(default=0.7, gt=0)
    sde_window_start: int = Field(default=0, ge=0)
    sde_window_end: int | None = Field(default=None, gt=0)
    sde_selected_steps: int | None = Field(default=None, gt=0)
    sde_indices: list[int] | None = None


class RLRolloutRequest(BaseModel):
    expected_policy_version: str
    modality: Literal["ti2t", "ti2ti"]
    messages: list[Message]
    seeds: list[int] = Field(min_length=1)
    max_new_tokens: int = Field(default=4096, gt=0)
    max_images: int = Field(default=1, ge=0)
    temperature: float = 1.0
    top_p: float = 1.0
    image_policy: ImagePolicyConfig | None = None

    @model_validator(mode="after")
    def validate_replay_contract(self):
        if self.temperature != 1.0 or self.top_p != 1.0:
            raise ValueError("RL replay currently requires temperature=1 and top_p=1")
        if self.modality == "ti2t" and self.max_images != 0:
            self.max_images = 0
        if self.modality == "ti2ti" and self.image_policy is None:
            self.image_policy = ImagePolicyConfig()
        return self
