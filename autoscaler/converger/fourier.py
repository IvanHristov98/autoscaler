from typing import List

import numpy as np
import numpy.fft as fft
import numpy.polynomial.polynomial as poly


class FourierExtrapolator:
    _NUM_HARMONICS = 100
    
    _y: np.ndarray
    _y_mean: float
    _period_scale: float
    _y_freqdom: np.ndarray
    _freqs: np.ndarray
    # The weights of the frequencies sorted by importance.
    _indices: List[int]
    _is_taught: bool
    
    def __init__(self) -> None:
        self._is_taught = False

    def is_taught(self) -> bool:
        return self._is_taught

    # from start to end inclusively
    def predict(self, start: int, end: int):
        t = np.arange(start, end + 1)
        restored_signal = np.zeros(t.size)
        
        for i in self._indices[: 1 + 2 * self._NUM_HARMONICS]:
            ampli = 2 * np.absolute(self._y_freqdom[i]) / self._y.size
            phase = np.angle(self._y_freqdom[i])

            restored_signal += ampli * np.cos(2.0 * np.pi * self._freqs[i] * t * self._period_scale + phase)

        return t, restored_signal + self._y_mean # + self._trend[0] * t + self._trend[1]

    def forget(self) -> None:
        self._is_taught = False

    def learn(self, y: np.ndarray, period_scale: float):
        self._is_taught = True

        self._y_mean = np.mean(y)
        self._y = y - self._y_mean
        self._period_scale = period_scale

        detrended_y = self._y# - self._trend[0] * t #- self._trend[1]
        self._y_freqdom = fft.rfft(detrended_y)
        self._freqs = fft.rfftfreq(self._y.size)

        self._indices = list(range(self._y_freqdom.size))
        self._indices.sort(key=lambda i: np.absolute(self._y_freqdom[i]), reverse=True)
