from __future__ import annotations
import numpy as _np
from numpy.typing import ArrayLike
from typing import Union, Type
from stochaster.buffer import Buffer, buffer
from stochaster.ops import Multiply


class Function:
  '''
  Function is the base layer for all function/operation layers
  Functions do not apply inplace
  
  Example of applying function on Tensor:

  import numpy as np
  x = Tensor(np.ones((6, 8)))
  x, ops.Sin.apply(x) # x will remain the same after this line
  '''
  
  def __init__(self, *tensors: Tensor):
    # check if we need to keep parents for grad
    self.input_requires_grad = [t.requires_grad for t in tensors]
    self.requires_grad = True if any(self.input_requires_grad) else False

    self.parents = tensors if self.requires_grad else None
  
  def forward(self, *args, **kwargs):
    raise f"type {type(self)} does not have a forward method"
  def backward(self, *args, **kwargs):
    # backward function is just taking a derivative during backpropagation
    raise f"type {type(self)} does not have a backward method"
  
  @classmethod
  def apply(fn: Type[Function], *input: Tensor, **kwargs):
    # applies `fn` on `input`s and returns the output
    context = fn(*input)
    output = Tensor(context.forward(*[t.data for t in input], **kwargs),
                    requires_grad=context.requires_grad)
    
    if (context.requires_grad):
      output.context = context
    
    return output


class Tensor:

  def __init__(self, data: ArrayLike[Union[int, float]], requires_grad=False):
    self.data: Buffer = buffer.array(data)
    self.requires_grad: bool = requires_grad
    self.grad: Buffer = None if not requires_grad else buffer.ones_like(self.data)
    self._context: Function = None

  def backward(self, allow_fill=True):
    if self._context is None:
      return
    return
  
  def dot(self, x: Tensor) -> Tensor:
    return Multiply.apply(self, x)


  # Numpy Properties
  
  @property
  def shape(self):
    return self.data.shape

  @property
  def ndim(self):
    return self.data.ndim

  @property
  def size(self):
    return self.data.size

  @property
  def dtype(self):
    return self.data.dtype

  @property
  def T(self):
    return Tensor(self.data.T, self.requires_grad)

  def astype(self, dtype):
    return Tensor(self.data.astype(dtype), self.requires_grad)

  def __getitem__(self, item):
    return Tensor(self.data[item], self.requires_grad)

  def __setitem__(self, key, value):
    self.data[key] = value

  def __str__(self):
    return str(self.data)

  def __repr__(self):
    return "Tensor(" + str(self.data) + ", requires_grad=" + str(self.requires_grad) + ")"


