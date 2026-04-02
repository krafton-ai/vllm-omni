# SPDX-License-Identifier: Apache-2.0
"""Shared utility functions for the Raon module."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _first_module_buffer(module: nn.Module) -> torch.Tensor | None:
    for _, buffer in module.named_buffers(recurse=True):
        return buffer
    return None


def _unwrap_singleton_tensor(value: Any) -> Any | None:
    if not isinstance(value, torch.Tensor):
        return value
    if value.numel() != 1:
        return None
    return value.item()


def module_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        buffer = _first_module_buffer(module)
        if buffer is not None:
            return buffer.device
        return torch.device("cpu")


def module_dtype(module: nn.Module) -> torch.dtype:
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        buffer = _first_module_buffer(module)
        if buffer is not None:
            return buffer.dtype
        return torch.float32


def unwrap_singleton_list(value: Any) -> Any:
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value


def collapse_exact_repeated_codec_snapshot(codes: torch.Tensor) -> torch.Tensor:
    """Collapse exact repeated [T, G] snapshots into one segment."""
    if codes.ndim != 2:
        return codes

    total_rows = int(codes.shape[0])
    if total_rows < 2:
        return codes

    max_repeats = min(total_rows, 16)
    for repeats in range(max_repeats, 1, -1):
        if total_rows % repeats != 0:
            continue
        seg_len = total_rows // repeats
        if seg_len <= 0:
            continue
        segment = codes[:seg_len]
        if all(torch.equal(segment, codes[i * seg_len : (i + 1) * seg_len]) for i in range(1, repeats)):
            return segment

    return codes


def coerce_optional_int(value: Any) -> int | None:
    value = unwrap_singleton_list(value)
    value = _unwrap_singleton_tensor(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_runtime_request_id(value: Any) -> str | None:
    value = unwrap_singleton_list(value)
    value = _unwrap_singleton_tensor(value)
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    req_id = str(value).strip()
    return req_id if req_id else None
