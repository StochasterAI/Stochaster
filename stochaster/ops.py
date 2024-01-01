import numpy as _np
from stochaster.tensor import Function
from typing import Tuple, Type
from numpy.typing import ArrayLike
from stochaster.buffer import Buffer
import math

# Reshape/Move Operations

class Expand(Function):
  def forward(self, tensor: Buffer, shape: Tuple[int, ...]) -> Buffer:
    assert len(shape) >= tensor.ndim, "shape cannot have less dims than tensor.shape"
    self.unsqueezed = len(shape) - tensor.ndim
    self.prev_shape = tensor.shape
    self.expanded = tuple(i for i, (a, b) in enumerate(zip(self.prev_shape, shape[self.unsqueezed:]))
                     if a != b)
    return _np.broadcast_to(tensor, shape)

  def backward(self, grad: Buffer) -> Buffer:
    input_grad = grad
    for i in range(self.unsqueezed): # return dims to normal
      input_grad = _np.sum(input_grad, axis=0)
      if (input_grad.ndim > 1):
        input_grad = _np.squeeze(input_grad, axis=0)
    for i in reversed(self.expanded):
      input_grad = _np.sum(input_grad, axis=i, keepdims=True)
    return input_grad

class Reshape(Function):
  def forward(self, tensor: Buffer, shape: Tuple[int, ...]) -> Buffer:
    self.prev_shape = tensor.shape
    return _np.reshape(tensor, shape)

  def backward(self, grad: Buffer) -> Buffer:
    return _np.reshape(grad, self.prev_shape)

class Permute(Function):
  def forward(self, tensor: Buffer, axis: Tuple[int, ...]) -> Buffer:
    self.prev_axis = tensor.shape
    return _np.transpose(tensor, axis)

  def backward(self, grad: Buffer) -> Buffer:
    inverse_axis = _np.argsort(self.prev_axis) # inverse permutation
    return _np.transpose(grad, inverse_axis)

class Pad(Function):
  def forward(self, tensor: Buffer, shape: Tuple[Tuple[int, int], ...]) -> Buffer:
    self.prev_shape = shape
    return _np.pad(tensor, shape, mode='constant', constant_values=0)

  def backward(self, grad: Buffer) -> Buffer:
    slices = tuple(slice(self.shape[i][0], grad.shape[i] - self.shape[i][1])
              for i in range(len(grad.shape)))
    return grad[slices]

class Shrink(Function):
  def forward(self, tensor: Buffer, shape: Tuple[int, ...]) -> Buffer:
    self.prev_shape = tensor.shape
    slices = tuple(slice(0, dim) for dim in shape)
    return tensor[slices]

  def backward(self, grad: Buffer) -> Buffer:
    pad_width = [(0, max(0, self.prev_shape[i] - grad.shape[i]))
                 for i in range(len(self.prev_shape))]
    return _np.pad(grad, pad_width, mode='constant', constant_values=0)

class Flip(Function):
  def forward(self, tensor: Buffer, axis: Tuple[int, ...]) -> Buffer:
    self.axis = axis
    return _np.flip(tensor, axis=axis)

  def backward(self, grad: Buffer) -> Buffer:
    return _np.flip(grad, axis=self.axis)

# Unary Operations

class Zero(Function):
  def forward(self, tensor: Buffer) -> Buffer:
    return _np.zeros_like(tensor)

  def backward(self, grad: Buffer) -> Buffer:
    return _np.zeros_like(grad)

class Negate(Function):
  def forward(self, tensor: Buffer) -> Buffer:
    return _np.negative(tensor)

  def backward(self, grad: Buffer) -> Buffer:
    return _np.negative(grad)

class Sin(Function):
  def forward(self, tensor: Buffer) -> Buffer:
    self.input_tensor = tensor
    return _np.sin(tensor)

  def backward(self, grad: Buffer) -> Buffer:
    tensor, = self.saved_tensors
    return grad * _np.cos(self.input_tensor)

class Tanh(Function):
  def forward(self, tensor: Buffer) -> Buffer:
    self.input_tensor = tensor
    return _np.tanh(tensor)

  def backward(self, grad: Buffer) -> Buffer:
    tensor, = self.saved_tensors
    return grad * (1 - _np.square(_np.tanh(self.input_tensor)))

class Relu(Function):
  def forward(self, tensor: Buffer) -> Buffer:
    self.input_tensor = tensor
    return _np.maximum(0, tensor)

  def backward(self, grad: Buffer) -> Buffer:
    grad_output = grad.copy()
    grad_output[self.input_tensor < 0] = 0
    return grad_output

class Log(Function):
  def forward(self, tensor: Buffer) -> Buffer:
    self.input_tensor = tensor
    return _np.log(tensor)
  
  def backward(self, grad: Buffer) -> Buffer:
    return grad / self.x

class Exp(Function):
  def forward(self, tensor: Buffer) -> Buffer:
    result = _np.exp(tensor)
    self.forward_result = result
    return result

  def backward(self, grad: Buffer) -> Buffer:
    return grad * self.forward_result

class Sqrt(Function):
  def forward(self, tensor: Buffer) -> Buffer:
    self.input_tensor = tensor
    return _np.sqrt(tensor)

  def backward(self, grad: Buffer) -> Buffer:
    return grad / (2 * _np.sqrt(self.input_tensor))

class Sigmoid(Function):
  def forward(self, tensor: Buffer) -> Buffer:
    result = 1 / (1 + _np.exp(-tensor))
    self.forward_result = result
    return result

  def backward(self, grad: Buffer) -> Buffer:
    return grad * self.forward_result * (1 - self.forward_result)
