from typing import List

import numpy as np
import numpy.fft as fft
import numpy.polynomial.polynomial as poly


class FourierExtrapolator:
    _NUM_HARMONICS = 10
    
    _y: np.ndarray
    _polynomial: poly.Polynomial
    _period_scale: float
    _trend: np.ndarray
    _y_freqdom: np.ndarray
    _freqs: np.ndarray
    # The weights of the frequencies sorted by importance.
    _indices: List[int]

    def __init__(self, y: np.ndarray, polynomial: poly.Polynomial, period_scale: float) -> None:
        self._y = y
        self._polynomial = polynomial
        self._period_scale = period_scale
        
        self._learn()

    # from start to end inclusively
    def predict(self, start: int, end: int):
        t = np.arange(start, end + 1)
        restored_signal = np.zeros(t.size)
        
        for i in self._indices[: 1 + 2 * self._NUM_HARMONICS]:
            ampli = 2 * np.absolute(self._y_freqdom[i]) / self._y.size
            phase = np.angle(self._y_freqdom[i])

            restored_signal += ampli * np.cos(2.0 * np.pi * self._freqs[i] * t * self._period_scale + phase)

        return t, restored_signal # + self._trend[0] * t + self._trend[1]

    def _learn(self):
        t = np.arange(0, self._y.size)
        self._trend = self._polynomial.fit(t, self._y, deg=1).convert().coef
        
        detrended_y = self._y# - self._trend[0] * t #- self._trend[1]
        self._y_freqdom = fft.rfft(detrended_y)
        self._freqs = fft.rfftfreq(self._y.size)

        self._indices = list(range(self._y_freqdom.size))
        self._indices.sort(key=lambda i: np.absolute(self._y_freqdom[i]), reverse=True)
