"""Device management for CPU/GPU support."""

import numpy as np
from typing import Union, Optional

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


class Device:
    """Device abstraction for CPU/GPU operations."""

    def __init__(self, device_type: str = "cpu"):
        """
        Initialize device.

        Args:
            device_type: Either "cpu" or "cuda"
        """
        if device_type not in ["cpu", "cuda"]:
            raise ValueError(f"Invalid device type: {device_type}. Must be 'cpu' or 'cuda'")

        if device_type == "cuda" and not CUPY_AVAILABLE:
            raise RuntimeError("CuPy is not installed. Install it for GPU support.")

        self.type = device_type
        self.xp = cp if device_type == "cuda" else np

    def __str__(self):
        return self.type

    def __repr__(self):
        return f"Device('{self.type}')"

    def __eq__(self, other):
        if isinstance(other, str):
            return self.type == other
        elif isinstance(other, Device):
            return self.type == other.type
        return False

    @property
    def is_cuda(self):
        """Check if device is CUDA."""
        return self.type == "cuda"

    @property
    def is_cpu(self):
        """Check if device is CPU."""
        return self.type == "cpu"


def get_array_module(arr):
    """Get the appropriate array module (numpy or cupy) for an array."""
    if CUPY_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp
    return np


def to_device(arr, device: Union[str, Device]):
    """Move array to specified device."""
    if isinstance(device, str):
        device = Device(device)

    xp_src = get_array_module(arr)

    if device.is_cuda:
        if xp_src == np:
            return cp.asarray(arr)
        return arr
    else:  # CPU
        if xp_src == cp:
            return cp.asnumpy(arr)
        return arr


# Global default device
_default_device = Device("cpu")


def get_default_device() -> Device:
    """Get the default device."""
    return _default_device


def set_default_device(device: Union[str, Device]):
    """Set the default device."""
    global _default_device
    if isinstance(device, str):
        device = Device(device)
    _default_device = device


def cuda_is_available() -> bool:
    """Check if CUDA is available."""
    return CUPY_AVAILABLE
