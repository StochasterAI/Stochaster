from typing import Any, Tuple
from numpy import *
from numpy._typing import NDArray

class Buffer(ndarray):
  def __new__(self, *args, **kwargs):
    return super().__new__(Buffer, *args, **kwargs)

def rand(shape: Tuple, generator=None, **kwargs) -> Buffer:
  return random.default_rng(generator).random(shape)
