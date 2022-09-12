from typing import Union, Optional
import numpy as np

class IIRFilter(object):
    """
    Implements a multidimensional IIR filter.
    """
    def __init__(self, alpha: Union[np.ndarray, float]) -> None:
        """
        alpha - (n,) array of alpha gains or single float, in [0,1]. 1 is no smoothing, 0 is completely damped.
        """
        self._alpha = alpha
        self._x = None

    def append(self, x: Union[np.ndarray, float]) -> None:
        if self._x is None:
            self._x = x
        else:
            self._x = self._alpha * x + (1. - self._alpha) * self._x

    def get(self) -> Optional[Union[np.ndarray, float]]:
        """
        Returns None if no data is in the filter.
        """
        return self._x

    def reset(self) -> None:
        self._x = None
