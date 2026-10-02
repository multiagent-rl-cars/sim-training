"""Small contracts shared by tensor kernels, without simulator imports."""

import math

import torch


def positive_scalar(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def floating_tensor(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point torch.Tensor")


def finite_tensor(value: torch.Tensor, name: str) -> None:
    """Debug/preflight validation; checking GPU values synchronizes the device."""
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")
