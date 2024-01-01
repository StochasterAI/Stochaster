import numpy as _np
from numpy.typing import ArrayLike
from typing import Union

buffer = _np

class Buffer(_np.ndarray):
  def __init__(self, *args, **kwargs):
    super().__init__()
