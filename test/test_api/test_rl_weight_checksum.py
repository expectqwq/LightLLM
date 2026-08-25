import hashlib

import torch

from lightllm.utils.rl_weight_update import tensor_checksum


def test_tensor_checksum_supports_scalar_bfloat16():
    tensor = torch.tensor(1.5, dtype=torch.bfloat16)
    expected = hashlib.sha256(
        tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    ).hexdigest()

    assert tensor_checksum(tensor) == expected
