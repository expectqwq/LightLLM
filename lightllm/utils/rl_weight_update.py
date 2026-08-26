"""Shared NCCL receiver for atomic RL policy updates."""

from __future__ import annotations

import base64
import os

import torch
import torch.distributed as dist
from safetensors.torch import load as load_safetensors

from lightllm.utils.dist_utils import init_custom_process_group

_DTYPES = {
    "torch.float32": torch.float32,
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.int64": torch.int64,
    "torch.int32": torch.int32,
    "torch.int8": torch.int8,
    "torch.uint8": torch.uint8,
    "torch.bool": torch.bool,
}


class DistributedWeightReceiver:
    def __init__(self, consumer: str, device: torch.device):
        self.consumer = consumer
        self.device = torch.device(device)
        self.groups: dict[str, tuple[object, str]] = {}

    def init_group(self, payload: dict, rank: int) -> dict:
        name = payload.get("group_name", "weight_update_group")
        if name in self.groups:
            raise RuntimeError(f"weight group {name} already exists")
        backend = payload.get("backend", "nccl")
        default_bound = (
            dist.group.WORLD.bound_device_id if dist.is_initialized() else None
        )
        print(
            "RL_WEIGHT_GROUP_INIT_BEGIN "
            f"pid={os.getpid()} consumer={self.consumer} name={name} "
            f"rank={int(rank)} world_size={int(payload['world_size'])} "
            f"port={int(payload['master_port'])} backend={backend} "
            f"device={self.device} default_bound={default_bound}",
            flush=True,
        )
        group = init_custom_process_group(
            backend=backend,
            init_method=f"tcp://{payload['master_address']}:{int(payload['master_port'])}",
            world_size=int(payload["world_size"]),
            rank=int(rank),
            group_name=name,
            # External publishers are not ranks in the serving default group.
            # Passing device_id enables PyTorch's device-bound split path and
            # makes NCCL wait for ranks that can never join this communicator.
            device_id=None,
        )
        print(
            f"RL_WEIGHT_GROUP_CREATED pid={os.getpid()} consumer={self.consumer} "
            f"name={name}",
            flush=True,
        )
        self.groups[name] = (group, backend)
        warmed_up = bool(payload.get("warmup", False))
        if warmed_up:
            transport_device = self.device if backend == "nccl" else torch.device("cpu")
            probe = torch.empty(1, dtype=torch.int64, device=transport_device)
            print(
                f"RL_WEIGHT_GROUP_WARMUP_BEGIN pid={os.getpid()} "
                f"consumer={self.consumer} name={name} device={transport_device}",
                flush=True,
            )
            dist.broadcast(probe, src=0, group=group)
            if int(probe.item()) != 20260827:
                raise RuntimeError(f"weight group {name} failed its warmup probe")
            print(
                f"RL_WEIGHT_GROUP_WARMUP_DONE pid={os.getpid()} "
                f"consumer={self.consumer} name={name}",
                flush=True,
            )
        return {
            "group_name": name,
            "rank": int(rank),
            "backend": backend,
            "warmed_up": warmed_up,
        }

    def destroy_group(self, name: str = "weight_update_group") -> dict:
        entry = self.groups.pop(name, None)
        if entry is not None:
            dist.destroy_process_group(entry[0])
        return {"group_name": name, "destroyed": entry is not None}

    def receive(self, payload: dict) -> tuple[dict[str, torch.Tensor], dict]:
        name = payload.get("group_name", "weight_update_group")
        if name not in self.groups:
            raise RuntimeError(f"weight group {name} is not initialized")
        group, backend = self.groups[name]
        transport_device = self.device if backend == "nccl" else torch.device("cpu")
        names = payload["names"]
        dtypes = payload["dtypes"]
        shapes = payload["shapes"]
        assignments = payload.get("assignments", {})
        if not (len(names) == len(dtypes) == len(shapes)):
            raise ValueError("weight manifest columns have different lengths")
        if payload.get("buckets"):
            return self._receive_buckets(payload, group, backend)
        selected: dict[str, torch.Tensor] = {}
        received: list[str] = []
        for tensor_name, dtype_name, shape in zip(names, dtypes, shapes):
            if dtype_name not in _DTYPES:
                raise ValueError(f"unsupported tensor dtype {dtype_name}")
            tensor = torch.empty(tuple(shape), dtype=_DTYPES[dtype_name], device=transport_device)
            dist.broadcast(tensor, src=0, group=group)
            owners = assignments.get(tensor_name, [])
            if not owners or self.consumer in owners:
                selected[tensor_name] = tensor.to(self.device)
                received.append(tensor_name)
        required = set(payload.get("required", {}).get(self.consumer, []))
        missing = sorted(required - set(selected))
        if missing:
            raise ValueError(f"consumer closure missing tensors: {missing[:5]}")
        return selected, {
            "consumer": self.consumer,
            "received_names": sorted(received),
            "closure_size": len(required),
        }

    def _receive_buckets(self, payload: dict, group, backend: str):
        names = payload["names"]
        dtypes = payload["dtypes"]
        shapes = payload["shapes"]
        assignments = payload.get("assignments", {})
        transport_device = self.device if backend == "nccl" else torch.device("cpu")
        selected = {}
        received = []
        received_bucket_count = 0
        for bucket in payload["buckets"]:
            consumers = bucket.get("consumers")
            if consumers and self.consumer not in consumers:
                continue
            dtype_name = bucket["dtype"]
            if dtype_name not in _DTYPES:
                raise ValueError(f"unsupported tensor dtype {dtype_name}")
            flat = torch.empty(int(bucket["numel"]), dtype=_DTYPES[dtype_name], device=transport_device)
            dist.broadcast(flat, src=0, group=group)
            received_bucket_count += 1
            offset = 0
            for entry_index in bucket["entry_indices"]:
                entry_index = int(entry_index)
                name = names[entry_index]
                numel = 1
                for dimension in shapes[entry_index]:
                    numel *= int(dimension)
                tensor = flat[offset : offset + numel].reshape(tuple(shapes[entry_index]))
                offset += numel
                owners = assignments.get(name, [])
                if not owners or self.consumer in owners:
                    selected[name] = tensor.to(self.device).clone()
                    received.append(name)
            if offset != flat.numel():
                raise ValueError(f"bucket geometry mismatch for {bucket['id']}")
        required = set(payload.get("required", {}).get(self.consumer, []))
        missing = sorted(required - set(selected))
        if missing:
            raise ValueError(f"consumer closure missing tensors: {missing[:5]}")
        return selected, {
            "consumer": self.consumer,
            "received_names": sorted(received),
            "closure_size": len(required),
            "bucket_count": received_bucket_count,
        }

    def decode_bundle(self, payload: dict) -> tuple[dict[str, torch.Tensor], dict]:
        tensors = load_safetensors(base64.b64decode(payload["serialized_safetensors"]))
        assignments = payload.get("assignments", {})
        selected = {}
        received = []
        for name, tensor in tensors.items():
            owners = assignments.get(name, [])
            if not owners or self.consumer in owners:
                selected[name] = tensor.to(self.device)
                received.append(name)
        required = set(payload.get("required", {}).get(self.consumer, []))
        missing = sorted(required - set(selected))
        if missing:
            raise ValueError(f"consumer closure missing tensors: {missing[:5]}")
        return selected, {
            "consumer": self.consumer,
            "received_names": sorted(received),
            "closure_size": len(required),
        }
