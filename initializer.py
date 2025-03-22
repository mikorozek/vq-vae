import abc
import math
import torch
from typing import Optional, Tuple, Union

class Initializer(abc.ABC):
  """Initializer base class, all initializers must implement a call method."""

  @abc.abstractmethod
  def __call__(self, shape, dtype=torch.float32) -> float:
    pass

class CodeBookInitializer(Initializer):
    def __init__(self,
                scale: float = 1.0,
                mode: str = "fan_in",
                seed: Optional[int] = None):
        if scale <= 0.:
              raise ValueError("`scale` must be positive float.")
        if mode not in {"fan_in", "fan_out", "fan_avg"}:
              raise ValueError("Invalid `mode` argument:", mode)
        self.scale = scale
        self.mode = mode
        self.seed = seed

    def __call__(self, shape, dtype=torch.float32) -> float:
        scale = self.scale
        if self.mode == "fan_in":
            scale /= max(1., shape[0])
        elif self.mode == "fan_out":
            scale /= max(1., shape[1])
        else:
            scale /= max(1., (shape[0] + shape[1]) / 2.)

        limit = math.sqrt(3.0 * scale)
        return limit

