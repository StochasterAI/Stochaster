from typing import Optional, Tuple
from numpy import *

class Buffer(ndarray):
  def __new__(self, *args, **kwargs):
    return super().__new__(Buffer, *args, **kwargs)

def rand(shape: Tuple, generator: Optional[int]) -> Buffer:
  print(f"shape is {shape}")
  return random.default_rng(generator).random(shape)
