from __future__ import annotations

import pytest
import torch

from vllm_omni.model_executor.models.raon.raon_speaker_encoder import (
    EcapaSpeakerBackend,
    EcapaTdnnEncoder,
    _ECAPA_CONFIG,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_ecapa_state_dict_includes_speechbrain_batch_norm_keys():
    state_keys = set(EcapaTdnnEncoder(_ECAPA_CONFIG).state_dict().keys())

    assert "blocks.0.norm.norm.weight" in state_keys
    assert "blocks.1.tdnn1.norm.norm.weight" in state_keys
    assert "asp_bn.norm.weight" in state_keys


def test_backend_normalize_uses_valid_frame_lengths_only():
    feats = torch.tensor(
        [
            [[1.0], [3.0], [100.0], [100.0]],
            [[2.0], [4.0], [6.0], [8.0]],
        ],
        dtype=torch.float32,
    )
    feat_lengths = torch.tensor([2, 4], dtype=torch.long)

    normalized = EcapaSpeakerBackend._normalize(feats, feat_lengths)

    assert torch.allclose(normalized[0, :2].mean(dim=0), torch.zeros(1), atol=1e-6)
    assert torch.allclose(normalized[1, :4].mean(dim=0), torch.zeros(1), atol=1e-6)
