"""Functional operations for neural networks."""

import numpy as np
from numpy_dl.core.tensor import Tensor
from numpy_dl.utils.device import get_array_module
from typing import Tuple, Union, Optional


def relu(x: Tensor) -> Tensor:
    """ReLU activation function."""
    xp = get_array_module(x.data)
    out = Tensor(
        xp.maximum(0, x.data),
        requires_grad=x.requires_grad,
        device=x.device,
        _children=(x,),
        _op='relu'
    )

    def _backward():
        if x.requires_grad:
            grad = (x.data > 0).astype(x.dtype) * out.grad
            x.grad = grad if x.grad is None else x.grad + grad

    out._backward = _backward
    return out


def leaky_relu(x: Tensor, negative_slope: float = 0.01) -> Tensor:
    """Leaky ReLU activation function."""
    xp = get_array_module(x.data)
    out = Tensor(
        xp.where(x.data > 0, x.data, negative_slope * x.data),
        requires_grad=x.requires_grad,
        device=x.device,
        _children=(x,),
        _op='leaky_relu'
    )

    def _backward():
        if x.requires_grad:
            grad = xp.where(x.data > 0, 1.0, negative_slope) * out.grad
            x.grad = grad if x.grad is None else x.grad + grad

    out._backward = _backward
    return out


def sigmoid(x: Tensor) -> Tensor:
    """Sigmoid activation function."""
    xp = get_array_module(x.data)
    out_data = 1.0 / (1.0 + xp.exp(-x.data))
    out = Tensor(
        out_data,
        requires_grad=x.requires_grad,
        device=x.device,
        _children=(x,),
        _op='sigmoid'
    )

    def _backward():
        if x.requires_grad:
            grad = out.data * (1 - out.data) * out.grad
            x.grad = grad if x.grad is None else x.grad + grad

    out._backward = _backward
    return out


def tanh(x: Tensor) -> Tensor:
    """Tanh activation function."""
    xp = get_array_module(x.data)
    out_data = xp.tanh(x.data)
    out = Tensor(
        out_data,
        requires_grad=x.requires_grad,
        device=x.device,
        _children=(x,),
        _op='tanh'
    )

    def _backward():
        if x.requires_grad:
            grad = (1 - out.data ** 2) * out.grad
            x.grad = grad if x.grad is None else x.grad + grad

    out._backward = _backward
    return out


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Softmax activation function."""
    xp = get_array_module(x.data)
    # Numerical stability
    exp_x = xp.exp(x.data - xp.max(x.data, axis=axis, keepdims=True))
    out_data = exp_x / xp.sum(exp_x, axis=axis, keepdims=True)
    out = Tensor(
        out_data,
        requires_grad=x.requires_grad,
        device=x.device,
        _children=(x,),
        _op='softmax'
    )

    def _backward():
        if x.requires_grad:
            # Jacobian computation for softmax
            s = out.data
            grad_out = out.grad
            grad = s * (grad_out - xp.sum(s * grad_out, axis=axis, keepdims=True))
            x.grad = grad if x.grad is None else x.grad + grad

    out._backward = _backward
    return out


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Log-softmax activation function."""
    xp = get_array_module(x.data)
    # Numerical stability
    max_x = xp.max(x.data, axis=axis, keepdims=True)
    shifted = x.data - max_x
    out_data = shifted - xp.log(xp.sum(xp.exp(shifted), axis=axis, keepdims=True))
    out = Tensor(
        out_data,
        requires_grad=x.requires_grad,
        device=x.device,
        _children=(x,),
        _op='log_softmax'
    )

    def _backward():
        if x.requires_grad:
            exp_out = xp.exp(out.data)
            grad = out.grad - exp_out * xp.sum(out.grad, axis=axis, keepdims=True)
            x.grad = grad if x.grad is None else x.grad + grad

    out._backward = _backward
    return out


def dropout(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """Dropout regularization."""
    if not training or p == 0:
        return x

    xp = get_array_module(x.data)
    mask = (xp.random.rand(*x.shape) > p).astype(x.dtype) / (1 - p)
    out = Tensor(
        x.data * mask,
        requires_grad=x.requires_grad,
        device=x.device,
        _children=(x,),
        _op='dropout'
    )

    def _backward():
        if x.requires_grad:
            grad = mask * out.grad
            x.grad = grad if x.grad is None else x.grad + grad

    out._backward = _backward
    return out


def conv2d(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
) -> Tensor:
    """2D convolution operation."""
    xp = get_array_module(x.data)

    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)

    # Input: (batch, in_channels, height, width)
    # Weight: (out_channels, in_channels, kernel_h, kernel_w)
    batch_size, in_channels, in_h, in_w = x.shape
    out_channels, _, kernel_h, kernel_w = weight.shape

    # Apply padding
    if padding[0] > 0 or padding[1] > 0:
        x_padded = xp.pad(
            x.data,
            ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])),
            mode='constant'
        )
    else:
        x_padded = x.data

    # Calculate output dimensions
    out_h = (in_h + 2 * padding[0] - kernel_h) // stride[0] + 1
    out_w = (in_w + 2 * padding[1] - kernel_w) // stride[1] + 1

    # Im2col transformation
    col = xp.zeros((batch_size, in_channels, kernel_h, kernel_w, out_h, out_w), dtype=x.dtype)
    for y in range(kernel_h):
        y_max = y + stride[0] * out_h
        for x_idx in range(kernel_w):
            x_max = x_idx + stride[1] * out_w
            col[:, :, y, x_idx, :, :] = x_padded[:, :, y:y_max:stride[0], x_idx:x_max:stride[1]]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(batch_size * out_h * out_w, -1)
    weight_col = weight.data.reshape(out_channels, -1).T

    # Convolution as matrix multiplication
    out_data = xp.dot(col, weight_col)
    out_data = out_data.reshape(batch_size, out_h, out_w, out_channels).transpose(0, 3, 1, 2)

    if bias is not None:
        out_data += bias.data.reshape(1, -1, 1, 1)

    children = (x, weight) if bias is None else (x, weight, bias)
    out = Tensor(
        out_data,
        requires_grad=x.requires_grad or weight.requires_grad or (bias is not None and bias.requires_grad),
        device=x.device,
        _children=children,
        _op='conv2d'
    )

    def _backward():
        if weight.requires_grad:
            # Gradient w.r.t. weight
            grad_out = out.grad.transpose(0, 2, 3, 1).reshape(batch_size * out_h * out_w, out_channels)
            grad_weight = xp.dot(col.T, grad_out)
            grad_weight = grad_weight.T.reshape(weight.shape)
            weight.grad = grad_weight if weight.grad is None else weight.grad + grad_weight

        if bias is not None and bias.requires_grad:
            # Gradient w.r.t. bias
            grad_bias = xp.sum(out.grad, axis=(0, 2, 3))
            bias.grad = grad_bias if bias.grad is None else bias.grad + grad_bias

        if x.requires_grad:
            # Gradient w.r.t. input (would need col2im - simplified for now)
            grad_out = out.grad.transpose(0, 2, 3, 1).reshape(batch_size * out_h * out_w, out_channels)
            grad_col = xp.dot(grad_out, weight_col.T)

            # Col2im transformation (simplified)
            grad_x = xp.zeros_like(x_padded)
            grad_col = grad_col.reshape(batch_size, out_h, out_w, in_channels, kernel_h, kernel_w).transpose(0, 3, 4, 5, 1, 2)

            for y in range(kernel_h):
                y_max = y + stride[0] * out_h
                for x_idx in range(kernel_w):
                    x_max = x_idx + stride[1] * out_w
                    grad_x[:, :, y:y_max:stride[0], x_idx:x_max:stride[1]] += grad_col[:, :, y, x_idx, :, :]

            # Remove padding
            if padding[0] > 0 or padding[1] > 0:
                grad_x = grad_x[:, :, padding[0]:-padding[0] if padding[0] > 0 else None,
                                padding[1]:-padding[1] if padding[1] > 0 else None]

            x.grad = grad_x if x.grad is None else x.grad + grad_x

    out._backward = _backward
    return out


def max_pool2d(
    x: Tensor,
    kernel_size: Union[int, Tuple[int, int]] = 2,
    stride: Optional[Union[int, Tuple[int, int]]] = None,
) -> Tensor:
    """2D max pooling operation."""
    xp = get_array_module(x.data)

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if stride is None:
        stride = kernel_size
    elif isinstance(stride, int):
        stride = (stride, stride)

    batch_size, channels, in_h, in_w = x.shape
    kernel_h, kernel_w = kernel_size

    out_h = (in_h - kernel_h) // stride[0] + 1
    out_w = (in_w - kernel_w) // stride[1] + 1

    # Create output
    out_data = xp.zeros((batch_size, channels, out_h, out_w), dtype=x.dtype)
    mask = xp.zeros_like(x.data)

    for i in range(out_h):
        for j in range(out_w):
            h_start = i * stride[0]
            h_end = h_start + kernel_h
            w_start = j * stride[1]
            w_end = w_start + kernel_w

            window = x.data[:, :, h_start:h_end, w_start:w_end]
            out_data[:, :, i, j] = xp.max(window, axis=(2, 3))

            # Create mask for backprop
            window_shape = window.shape
            window_flat = window.reshape(batch_size, channels, -1)
            max_indices = xp.argmax(window_flat, axis=2)

            # Create mask in window
            window_mask = xp.zeros_like(window_flat)
            batch_indices = xp.arange(batch_size)[:, None]
            channel_indices = xp.arange(channels)[None, :]
            window_mask[batch_indices, channel_indices, max_indices] = 1
            window_mask = window_mask.reshape(window_shape)

            mask[:, :, h_start:h_end, w_start:w_end] += window_mask

    out = Tensor(
        out_data,
        requires_grad=x.requires_grad,
        device=x.device,
        _children=(x,),
        _op='max_pool2d'
    )

    def _backward():
        if x.requires_grad:
            grad_x = xp.zeros_like(x.data)
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * stride[0]
                    h_end = h_start + kernel_h
                    w_start = j * stride[1]
                    w_end = w_start + kernel_w

                    window_mask = mask[:, :, h_start:h_end, w_start:w_end]
                    grad_window = out.grad[:, :, i:i+1, j:j+1] * window_mask
                    grad_x[:, :, h_start:h_end, w_start:w_end] += grad_window

            x.grad = grad_x if x.grad is None else x.grad + grad_x

    out._backward = _backward
    return out


def avg_pool2d(
    x: Tensor,
    kernel_size: Union[int, Tuple[int, int]] = 2,
    stride: Optional[Union[int, Tuple[int, int]]] = None,
) -> Tensor:
    """2D average pooling operation."""
    xp = get_array_module(x.data)

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if stride is None:
        stride = kernel_size
    elif isinstance(stride, int):
        stride = (stride, stride)

    batch_size, channels, in_h, in_w = x.shape
    kernel_h, kernel_w = kernel_size

    out_h = (in_h - kernel_h) // stride[0] + 1
    out_w = (in_w - kernel_w) // stride[1] + 1

    out_data = xp.zeros((batch_size, channels, out_h, out_w), dtype=x.dtype)

    for i in range(out_h):
        for j in range(out_w):
            h_start = i * stride[0]
            h_end = h_start + kernel_h
            w_start = j * stride[1]
            w_end = w_start + kernel_w

            window = x.data[:, :, h_start:h_end, w_start:w_end]
            out_data[:, :, i, j] = xp.mean(window, axis=(2, 3))

    out = Tensor(
        out_data,
        requires_grad=x.requires_grad,
        device=x.device,
        _children=(x,),
        _op='avg_pool2d'
    )

    def _backward():
        if x.requires_grad:
            grad_x = xp.zeros_like(x.data)
            grad_value = 1.0 / (kernel_h * kernel_w)

            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * stride[0]
                    h_end = h_start + kernel_h
                    w_start = j * stride[1]
                    w_end = w_start + kernel_w

                    grad_x[:, :, h_start:h_end, w_start:w_end] += (
                        out.grad[:, :, i:i+1, j:j+1] * grad_value
                    )

            x.grad = grad_x if x.grad is None else x.grad + grad_x

    out._backward = _backward
    return out


def conv2d_transpose(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    output_padding: Union[int, Tuple[int, int]] = 0,
) -> Tensor:
    """2D transposed convolution (deconvolution) operation."""
    xp = get_array_module(x.data)

    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(output_padding, int):
        output_padding = (output_padding, output_padding)

    # Input: (batch, in_channels, height, width)
    # Weight: (in_channels, out_channels, kernel_h, kernel_w)
    batch_size, in_channels, in_h, in_w = x.shape
    _, out_channels, kernel_h, kernel_w = weight.shape

    # Calculate output dimensions
    out_h = (in_h - 1) * stride[0] - 2 * padding[0] + kernel_h + output_padding[0]
    out_w = (in_w - 1) * stride[1] - 2 * padding[1] + kernel_w + output_padding[1]

    # Initialize output
    out_data = xp.zeros((batch_size, out_channels, out_h, out_w), dtype=x.dtype)

    # Perform transposed convolution
    for b in range(batch_size):
        for oc in range(out_channels):
            for ic in range(in_channels):
                for i in range(in_h):
                    for j in range(in_w):
                        h_start = i * stride[0] - padding[0]
                        w_start = j * stride[1] - padding[1]
                        h_end = h_start + kernel_h
                        w_end = w_start + kernel_w

                        # Bounds checking
                        h_start_valid = max(0, h_start)
                        w_start_valid = max(0, w_start)
                        h_end_valid = min(out_h, h_end)
                        w_end_valid = min(out_w, w_end)

                        if h_start_valid < h_end_valid and w_start_valid < w_end_valid:
                            kernel_h_start = h_start_valid - h_start
                            kernel_w_start = w_start_valid - w_start
                            kernel_h_end = kernel_h_start + (h_end_valid - h_start_valid)
                            kernel_w_end = kernel_w_start + (w_end_valid - w_start_valid)

                            out_data[b, oc, h_start_valid:h_end_valid, w_start_valid:w_end_valid] += (
                                x.data[b, ic, i, j] * weight.data[ic, oc, kernel_h_start:kernel_h_end, kernel_w_start:kernel_w_end]
                            )

    if bias is not None:
        out_data += bias.data.reshape(1, -1, 1, 1)

    children = (x, weight) if bias is None else (x, weight, bias)
    out = Tensor(
        out_data,
        requires_grad=x.requires_grad or weight.requires_grad or (bias is not None and bias.requires_grad),
        device=x.device,
        _children=children,
        _op='conv2d_transpose'
    )

    def _backward():
        if x.requires_grad:
            # Gradient w.r.t. input is a regular convolution
            grad_x = xp.zeros_like(x.data)
            for b in range(batch_size):
                for ic in range(in_channels):
                    for oc in range(out_channels):
                        for i in range(in_h):
                            for j in range(in_w):
                                h_start = i * stride[0] - padding[0]
                                w_start = j * stride[1] - padding[1]
                                h_end = h_start + kernel_h
                                w_end = w_start + kernel_w

                                h_start_valid = max(0, h_start)
                                w_start_valid = max(0, w_start)
                                h_end_valid = min(out_h, h_end)
                                w_end_valid = min(out_w, w_end)

                                if h_start_valid < h_end_valid and w_start_valid < w_end_valid:
                                    kernel_h_start = h_start_valid - h_start
                                    kernel_w_start = w_start_valid - w_start
                                    kernel_h_end = kernel_h_start + (h_end_valid - h_start_valid)
                                    kernel_w_end = kernel_w_start + (w_end_valid - w_start_valid)

                                    grad_x[b, ic, i, j] += xp.sum(
                                        out.grad[b, oc, h_start_valid:h_end_valid, w_start_valid:w_end_valid] *
                                        weight.data[ic, oc, kernel_h_start:kernel_h_end, kernel_w_start:kernel_w_end]
                                    )

            x.grad = grad_x if x.grad is None else x.grad + grad_x

        if weight.requires_grad:
            # Gradient w.r.t. weight
            grad_weight = xp.zeros_like(weight.data)
            for ic in range(in_channels):
                for oc in range(out_channels):
                    for b in range(batch_size):
                        for i in range(in_h):
                            for j in range(in_w):
                                h_start = i * stride[0] - padding[0]
                                w_start = j * stride[1] - padding[1]
                                h_end = h_start + kernel_h
                                w_end = w_start + kernel_w

                                h_start_valid = max(0, h_start)
                                w_start_valid = max(0, w_start)
                                h_end_valid = min(out_h, h_end)
                                w_end_valid = min(out_w, w_end)

                                if h_start_valid < h_end_valid and w_start_valid < w_end_valid:
                                    kernel_h_start = h_start_valid - h_start
                                    kernel_w_start = w_start_valid - w_start
                                    kernel_h_end = kernel_h_start + (h_end_valid - h_start_valid)
                                    kernel_w_end = kernel_w_start + (w_end_valid - w_start_valid)

                                    grad_weight[ic, oc, kernel_h_start:kernel_h_end, kernel_w_start:kernel_w_end] += (
                                        x.data[b, ic, i, j] *
                                        out.grad[b, oc, h_start_valid:h_end_valid, w_start_valid:w_end_valid]
                                    )

            weight.grad = grad_weight if weight.grad is None else weight.grad + grad_weight

        if bias is not None and bias.requires_grad:
            # Gradient w.r.t. bias
            grad_bias = xp.sum(out.grad, axis=(0, 2, 3))
            bias.grad = grad_bias if bias.grad is None else bias.grad + grad_bias

    out._backward = _backward
    return out


def batch_norm(
    x: Tensor,
    gamma: Tensor,
    beta: Tensor,
    running_mean: Optional[Tensor] = None,
    running_var: Optional[Tensor] = None,
    training: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
    """Batch normalization operation."""
    xp = get_array_module(x.data)

    if training:
        # Compute batch statistics
        axes = tuple(range(x.ndim - 1))  # All axes except channel axis
        mean = xp.mean(x.data, axis=axes, keepdims=True)
        var = xp.var(x.data, axis=axes, keepdims=True)

        # Update running statistics
        if running_mean is not None:
            running_mean.data = (1 - momentum) * running_mean.data + momentum * mean.squeeze()
        if running_var is not None:
            running_var.data = (1 - momentum) * running_var.data + momentum * var.squeeze()
    else:
        # Use running statistics
        mean = running_mean.data.reshape(1, -1) if running_mean is not None else 0
        var = running_var.data.reshape(1, -1) if running_var is not None else 1

    # Normalize
    x_norm = (x.data - mean) / xp.sqrt(var + eps)
    out_data = gamma.data.reshape(1, -1) * x_norm + beta.data.reshape(1, -1)

    out = Tensor(
        out_data,
        requires_grad=x.requires_grad or gamma.requires_grad or beta.requires_grad,
        device=x.device,
        _children=(x, gamma, beta),
        _op='batch_norm'
    )

    def _backward():
        N = x.data.size // x.shape[-1]  # Number of elements per channel

        if gamma.requires_grad:
            grad_gamma = xp.sum(out.grad * x_norm, axis=tuple(range(x.ndim - 1)))
            gamma.grad = grad_gamma if gamma.grad is None else gamma.grad + grad_gamma

        if beta.requires_grad:
            grad_beta = xp.sum(out.grad, axis=tuple(range(x.ndim - 1)))
            beta.grad = grad_beta if beta.grad is None else beta.grad + grad_beta

        if x.requires_grad and training:
            dx_norm = out.grad * gamma.data.reshape(1, -1)
            dvar = xp.sum(dx_norm * (x.data - mean) * -0.5 * (var + eps) ** (-1.5), axis=tuple(range(x.ndim - 1)), keepdims=True)
            dmean = xp.sum(dx_norm * -1 / xp.sqrt(var + eps), axis=tuple(range(x.ndim - 1)), keepdims=True) + dvar * xp.sum(-2 * (x.data - mean), axis=tuple(range(x.ndim - 1)), keepdims=True) / N

            grad_x = dx_norm / xp.sqrt(var + eps) + dvar * 2 * (x.data - mean) / N + dmean / N
            x.grad = grad_x if x.grad is None else x.grad + grad_x

    out._backward = _backward
    return out
