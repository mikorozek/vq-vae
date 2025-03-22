import abc
import math
import torch
from typing import Optional, Tuple, Union

ShapeLike = Union[Tuple[int, ...], torch.Size]

class Initializer(abc.ABC):
  """Initializer base class, all initializers must implement a call method."""

  @abc.abstractmethod
  def __call__(self, shape: ShapeLike, dtype=torch.float32) -> torch.Tensor:
    pass

class VarianceScaling(Initializer):
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

    def __call__(self, shape: ShapeLike, dtype=torch.float32) -> torch.Tensor:
        scale = self.scale
        fan_in, fan_out = _compute_fans(shape)
        if self.mode == "fan_in":
            scale /= max(1., fan_in)
        elif self.mode == "fan_out":
            scale /= max(1., fan_out)
        else:
            scale /= max(1., (fan_in + fan_out) / 2.)

        limit = math.sqrt(3.0 * scale)
        return torch.empty(shape, dtype=dtype).uniform_(-limit, limit)


def _compute_fans(shape: ShapeLike):
  if len(shape) < 1:  # Just to avoid errors for constants.
    fan_in = fan_out = 1
  elif len(shape) == 1:
    fan_in = fan_out = shape[0]
  elif len(shape) == 2:
    fan_in = shape[0]
    fan_out = shape[1]
  else:
    # Assuming convolution kernels (2D, 3D, or more).
    # kernel shape: (..., input_depth, depth)
    receptive_field_size = 1.
    for dim in shape[:-2]:
      receptive_field_size *= dim
    fan_in = shape[-2] * receptive_field_size
    fan_out = shape[-1] * receptive_field_size
  return fan_in, fan_out
