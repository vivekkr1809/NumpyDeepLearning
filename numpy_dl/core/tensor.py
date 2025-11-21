"""Tensor class with automatic differentiation support."""

import numpy as np
from typing import Union, Tuple, Optional, List
from numpy_dl.utils.device import Device, get_array_module, to_device, get_default_device


class Tensor:
    """
    Tensor class with automatic differentiation.

    Attributes:
        data: The underlying numpy/cupy array
        grad: Gradient tensor
        requires_grad: Whether to track gradients
        device: Device where tensor resides
    """

    def __init__(
        self,
        data,
        requires_grad: bool = False,
        device: Optional[Union[str, Device]] = None,
        _children: Tuple['Tensor', ...] = (),
        _op: str = '',
    ):
        """
        Initialize a Tensor.

        Args:
            data: Array-like data
            requires_grad: Whether to compute gradients
            device: Device to place tensor on
            _children: Parent tensors in computation graph
            _op: Operation that created this tensor
        """
        if device is None:
            device = get_default_device()
        elif isinstance(device, str):
            device = Device(device)

        self.device = device
        xp = device.xp

        if isinstance(data, Tensor):
            self.data = to_device(data.data, device)
        else:
            self.data = to_device(xp.asarray(data), device)

        self.requires_grad = requires_grad
        self.grad = None

        # For autograd
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @property
    def shape(self):
        """Get tensor shape."""
        return self.data.shape

    @property
    def dtype(self):
        """Get tensor dtype."""
        return self.data.dtype

    @property
    def ndim(self):
        """Get number of dimensions."""
        return self.data.ndim

    @property
    def size(self):
        """Get total number of elements."""
        return self.data.size

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad}, device={self.device})"

    def __str__(self):
        return str(self.data)

    # ============ Tensor Operations ============

    def __add__(self, other):
        """Addition."""
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            device=self.device,
            _children=(self, other),
            _op='+'
        )

        def _backward():
            if self.requires_grad:
                grad = out.grad
                # Handle broadcasting
                ndims_added = grad.ndim - self.data.ndim
                for _ in range(ndims_added):
                    grad = grad.sum(axis=0)
                for i, dim in enumerate(self.shape):
                    if dim == 1:
                        grad = grad.sum(axis=i, keepdims=True)
                self.grad = grad if self.grad is None else self.grad + grad

            if other.requires_grad:
                grad = out.grad
                # Handle broadcasting
                ndims_added = grad.ndim - other.data.ndim
                for _ in range(ndims_added):
                    grad = grad.sum(axis=0)
                for i, dim in enumerate(other.shape):
                    if dim == 1:
                        grad = grad.sum(axis=i, keepdims=True)
                other.grad = grad if other.grad is None else other.grad + grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        """Element-wise multiplication."""
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            device=self.device,
            _children=(self, other),
            _op='*'
        )

        def _backward():
            if self.requires_grad:
                grad = other.data * out.grad
                # Handle broadcasting
                ndims_added = grad.ndim - self.data.ndim
                for _ in range(ndims_added):
                    grad = grad.sum(axis=0)
                for i, dim in enumerate(self.shape):
                    if dim == 1:
                        grad = grad.sum(axis=i, keepdims=True)
                self.grad = grad if self.grad is None else self.grad + grad

            if other.requires_grad:
                grad = self.data * out.grad
                # Handle broadcasting
                ndims_added = grad.ndim - other.data.ndim
                for _ in range(ndims_added):
                    grad = grad.sum(axis=0)
                for i, dim in enumerate(other.shape):
                    if dim == 1:
                        grad = grad.sum(axis=i, keepdims=True)
                other.grad = grad if other.grad is None else other.grad + grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        """Power operation."""
        assert isinstance(other, (int, float)), "Only support int/float powers for now"
        out = Tensor(
            self.data ** other,
            requires_grad=self.requires_grad,
            device=self.device,
            _children=(self,),
            _op=f'**{other}'
        )

        def _backward():
            if self.requires_grad:
                grad = (other * self.data ** (other - 1)) * out.grad
                self.grad = grad if self.grad is None else self.grad + grad

        out._backward = _backward
        return out

    def __neg__(self):
        """Negation."""
        return self * -1

    def __sub__(self, other):
        """Subtraction."""
        return self + (-other)

    def __truediv__(self, other):
        """Division."""
        return self * (other ** -1)

    def __radd__(self, other):
        """Right addition."""
        return self + other

    def __rsub__(self, other):
        """Right subtraction."""
        return other + (-self)

    def __rmul__(self, other):
        """Right multiplication."""
        return self * other

    def __rtruediv__(self, other):
        """Right division."""
        return other * (self ** -1)

    def __matmul__(self, other):
        """Matrix multiplication."""
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        xp = get_array_module(self.data)
        out = Tensor(
            xp.matmul(self.data, other.data),
            requires_grad=self.requires_grad or other.requires_grad,
            device=self.device,
            _children=(self, other),
            _op='@'
        )

        def _backward():
            if self.requires_grad:
                grad = xp.matmul(out.grad, other.data.swapaxes(-2, -1))
                self.grad = grad if self.grad is None else self.grad + grad

            if other.requires_grad:
                grad = xp.matmul(self.data.swapaxes(-2, -1), out.grad)
                other.grad = grad if other.grad is None else other.grad + grad

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        """Sum elements."""
        xp = get_array_module(self.data)
        out = Tensor(
            xp.sum(self.data, axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            device=self.device,
            _children=(self,),
            _op='sum'
        )

        def _backward():
            if self.requires_grad:
                grad = out.grad
                if axis is not None:
                    if not keepdims:
                        if isinstance(axis, int):
                            grad = xp.expand_dims(grad, axis=axis)
                        else:
                            for ax in sorted(axis):
                                grad = xp.expand_dims(grad, axis=ax)
                grad = xp.broadcast_to(grad, self.shape)
                self.grad = grad if self.grad is None else self.grad + grad

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        """Mean of elements."""
        xp = get_array_module(self.data)
        n = self.size if axis is None else np.prod([self.shape[i] for i in (axis if isinstance(axis, tuple) else (axis,))])
        out = Tensor(
            xp.mean(self.data, axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            device=self.device,
            _children=(self,),
            _op='mean'
        )

        def _backward():
            if self.requires_grad:
                grad = out.grad / n
                if axis is not None:
                    if not keepdims:
                        if isinstance(axis, int):
                            grad = xp.expand_dims(grad, axis=axis)
                        else:
                            for ax in sorted(axis):
                                grad = xp.expand_dims(grad, axis=ax)
                grad = xp.broadcast_to(grad, self.shape)
                self.grad = grad if self.grad is None else self.grad + grad

        out._backward = _backward
        return out

    def reshape(self, *shape):
        """Reshape tensor."""
        xp = get_array_module(self.data)
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]

        out = Tensor(
            xp.reshape(self.data, shape),
            requires_grad=self.requires_grad,
            device=self.device,
            _children=(self,),
            _op='reshape'
        )

        def _backward():
            if self.requires_grad:
                grad = xp.reshape(out.grad, self.shape)
                self.grad = grad if self.grad is None else self.grad + grad

        out._backward = _backward
        return out

    def transpose(self, *axes):
        """Transpose tensor."""
        xp = get_array_module(self.data)
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = axes[0]

        out = Tensor(
            xp.transpose(self.data, axes=axes),
            requires_grad=self.requires_grad,
            device=self.device,
            _children=(self,),
            _op='transpose'
        )

        def _backward():
            if self.requires_grad:
                if axes is None:
                    inv_axes = None
                else:
                    inv_axes = tuple(np.argsort(axes))
                grad = xp.transpose(out.grad, axes=inv_axes)
                self.grad = grad if self.grad is None else self.grad + grad

        out._backward = _backward
        return out

    @property
    def T(self):
        """Transpose (2D only)."""
        return self.transpose()

    def exp(self):
        """Exponential."""
        xp = get_array_module(self.data)
        out = Tensor(
            xp.exp(self.data),
            requires_grad=self.requires_grad,
            device=self.device,
            _children=(self,),
            _op='exp'
        )

        def _backward():
            if self.requires_grad:
                grad = out.data * out.grad
                self.grad = grad if self.grad is None else self.grad + grad

        out._backward = _backward
        return out

    def log(self):
        """Natural logarithm."""
        xp = get_array_module(self.data)
        out = Tensor(
            xp.log(self.data),
            requires_grad=self.requires_grad,
            device=self.device,
            _children=(self,),
            _op='log'
        )

        def _backward():
            if self.requires_grad:
                grad = (1.0 / self.data) * out.grad
                self.grad = grad if self.grad is None else self.grad + grad

        out._backward = _backward
        return out

    def sqrt(self):
        """Square root."""
        return self ** 0.5

    def clip(self, min_val=None, max_val=None):
        """Clip values."""
        xp = get_array_module(self.data)
        out = Tensor(
            xp.clip(self.data, min_val, max_val),
            requires_grad=self.requires_grad,
            device=self.device,
            _children=(self,),
            _op='clip'
        )

        def _backward():
            if self.requires_grad:
                mask = xp.ones_like(self.data)
                if min_val is not None:
                    mask = mask * (self.data >= min_val)
                if max_val is not None:
                    mask = mask * (self.data <= max_val)
                grad = mask * out.grad
                self.grad = grad if self.grad is None else self.grad + grad

        out._backward = _backward
        return out

    # ============ Gradient Operations ============

    def backward(self):
        """Compute gradients using backpropagation."""
        if not self.requires_grad:
            raise RuntimeError("Tensor does not require grad")

        xp = get_array_module(self.data)
        # Initialize gradient
        if self.grad is None:
            self.grad = xp.ones_like(self.data)

        # Topological sort
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Backpropagate
        for node in reversed(topo):
            node._backward()

    def zero_grad(self):
        """Zero out gradients."""
        self.grad = None

    # ============ Utility Methods ============

    def to(self, device: Union[str, Device]) -> 'Tensor':
        """Move tensor to device."""
        if isinstance(device, str):
            device = Device(device)

        if device == self.device:
            return self

        return Tensor(
            to_device(self.data, device),
            requires_grad=self.requires_grad,
            device=device
        )

    def cpu(self) -> 'Tensor':
        """Move tensor to CPU."""
        return self.to('cpu')

    def cuda(self) -> 'Tensor':
        """Move tensor to CUDA."""
        return self.to('cuda')

    def detach(self) -> 'Tensor':
        """Detach from computation graph."""
        return Tensor(self.data, requires_grad=False, device=self.device)

    def numpy(self):
        """Convert to numpy array."""
        xp = get_array_module(self.data)
        if xp.__name__ == 'cupy':
            return xp.asnumpy(self.data)
        return self.data

    def item(self):
        """Get scalar value."""
        return self.data.item()

    # ============ In-place Operations ============

    def __getitem__(self, idx):
        """Indexing."""
        xp = get_array_module(self.data)
        out = Tensor(
            self.data[idx],
            requires_grad=self.requires_grad,
            device=self.device,
            _children=(self,),
            _op='getitem'
        )

        def _backward():
            if self.requires_grad:
                grad = xp.zeros_like(self.data)
                grad[idx] = out.grad
                self.grad = grad if self.grad is None else self.grad + grad

        out._backward = _backward
        return out


def tensor(data, requires_grad=False, device=None, dtype=None):
    """Create a tensor."""
    if device is None:
        device = get_default_device()
    elif isinstance(device, str):
        device = Device(device)

    xp = device.xp
    if dtype is not None:
        data = xp.asarray(data, dtype=dtype)

    return Tensor(data, requires_grad=requires_grad, device=device)


def zeros(*shape, requires_grad=False, device=None, dtype=np.float32):
    """Create a tensor of zeros."""
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]

    if device is None:
        device = get_default_device()
    elif isinstance(device, str):
        device = Device(device)

    xp = device.xp
    return Tensor(xp.zeros(shape, dtype=dtype), requires_grad=requires_grad, device=device)


def ones(*shape, requires_grad=False, device=None, dtype=np.float32):
    """Create a tensor of ones."""
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]

    if device is None:
        device = get_default_device()
    elif isinstance(device, str):
        device = Device(device)

    xp = device.xp
    return Tensor(xp.ones(shape, dtype=dtype), requires_grad=requires_grad, device=device)


def randn(*shape, requires_grad=False, device=None, dtype=np.float32):
    """Create a tensor with random normal values."""
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]

    if device is None:
        device = get_default_device()
    elif isinstance(device, str):
        device = Device(device)

    xp = device.xp
    return Tensor(xp.random.randn(*shape).astype(dtype), requires_grad=requires_grad, device=device)


def rand(*shape, requires_grad=False, device=None, dtype=np.float32):
    """Create a tensor with random uniform values."""
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]

    if device is None:
        device = get_default_device()
    elif isinstance(device, str):
        device = Device(device)

    xp = device.xp
    return Tensor(xp.random.rand(*shape).astype(dtype), requires_grad=requires_grad, device=device)
