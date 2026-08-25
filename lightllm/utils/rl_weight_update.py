"""Shared NCCL receiver for atomic RL policy updates."""

from __future__ import annotations

import hashlib
import base64

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


def tensor_checksum(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


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
        group = init_custom_process_group(
            backend=backend,
            init_method=f"tcp://{payload['master_address']}:{int(payload['master_port'])}",
            world_size=int(payload["world_size"]),
            rank=int(rank),
            group_name=name,
            device_id=self.device if backend == "nccl" else None,
        )
        self.groups[name] = (group, backend)
        return {"group_name": name, "rank": int(rank), "backend": backend}

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
        checksums = payload["checksums"]
        assignments = payload.get("assignments", {})
        if not (len(names) == len(dtypes) == len(shapes) == len(checksums)):
            raise ValueError("weight manifest columns have different lengths")
        if payload.get("buckets"):
            return self._receive_buckets(payload, group, backend)
        selected: dict[str, torch.Tensor] = {}
        received: dict[str, str] = {}
        for tensor_name, dtype_name, shape, expected in zip(names, dtypes, shapes, checksums):
            if dtype_name not in _DTYPES:
                raise ValueError(f"unsupported tensor dtype {dtype_name}")
            tensor = torch.empty(tuple(shape), dtype=_DTYPES[dtype_name], device=transport_device)
            dist.broadcast(tensor, src=0, group=group)
            actual = tensor_checksum(tensor)
            if actual != expected:
                raise ValueError(f"checksum mismatch for {tensor_name}")
            owners = assignments.get(tensor_name, [])
            if not owners or self.consumer in owners:
                selected[tensor_name] = tensor.to(self.device)
                received[tensor_name] = actual
        required = set(payload.get("required", {}).get(self.consumer, []))
        missing = sorted(required - set(selected))
        if missing:
            raise ValueError(f"consumer closure missing tensors: {missing[:5]}")
        return selected, {
            "consumer": self.consumer,
            "received_names": sorted(received),
            "checksums": received,
            "closure_size": len(required),
        }

    def _receive_buckets(self, payload: dict, group, backend: str):
        names = payload["names"]
        dtypes = payload["dtypes"]
        shapes = payload["shapes"]
        checksums = payload["checksums"]
        assignments = payload.get("assignments", {})
        transport_device = self.device if backend == "nccl" else torch.device("cpu")
        selected = {}
        received = {}
        for bucket in payload["buckets"]:
            dtype_name = bucket["dtype"]
            if dtype_name not in _DTYPES:
                raise ValueError(f"unsupported tensor dtype {dtype_name}")
            flat = torch.empty(int(bucket["numel"]), dtype=_DTYPES[dtype_name], device=transport_device)
            dist.broadcast(flat, src=0, group=group)
            if tensor_checksum(flat) != bucket["checksum"]:
                raise ValueError(f"bucket checksum mismatch for {bucket['id']}")
            offset = 0
            for entry_index in bucket["entry_indices"]:
                entry_index = int(entry_index)
                name = names[entry_index]
                numel = 1
                for dimension in shapes[entry_index]:
                    numel *= int(dimension)
                tensor = flat[offset : offset + numel].reshape(tuple(shapes[entry_index]))
                offset += numel
                if tensor_checksum(tensor) != checksums[entry_index]:
                    raise ValueError(f"checksum mismatch for {name}")
                owners = assignments.get(name, [])
                if not owners or self.consumer in owners:
                    selected[name] = tensor.to(self.device).clone()
                    received[name] = checksums[entry_index]
            if offset != flat.numel():
                raise ValueError(f"bucket geometry mismatch for {bucket['id']}")
        required = set(payload.get("required", {}).get(self.consumer, []))
        missing = sorted(required - set(selected))
        if missing:
            raise ValueError(f"consumer closure missing tensors: {missing[:5]}")
        return selected, {
            "consumer": self.consumer,
            "received_names": sorted(received),
            "checksums": received,
            "closure_size": len(required),
            "bucket_count": len(payload["buckets"]),
        }

    def decode_bundle(self, payload: dict) -> tuple[dict[str, torch.Tensor], dict]:
        tensors = load_safetensors(base64.b64decode(payload["serialized_safetensors"]))
        checksums = payload["checksums"]
        assignments = payload.get("assignments", {})
        selected = {}
        received = {}
        for name, tensor in tensors.items():
            if name not in checksums or tensor_checksum(tensor) != checksums[name]:
                raise ValueError(f"checksum mismatch for {name}")
            owners = assignments.get(name, [])
            if not owners or self.consumer in owners:
                selected[name] = tensor.to(self.device)
                received[name] = checksums[name]
        required = set(payload.get("required", {}).get(self.consumer, []))
        missing = sorted(required - set(selected))
        if missing:
            raise ValueError(f"consumer closure missing tensors: {missing[:5]}")
        return selected, {
            "consumer": self.consumer,
            "received_names": sorted(received),
            "checksums": received,
            "closure_size": len(required),
        }
