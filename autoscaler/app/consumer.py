import abc

import numpy as np


# N is the number of data points per period.
# This is the num of data points per season.
N = 1000
# PERIOD is the length between two wave extremums. Just check your maths.
PERIOD = 100
# ANGULAR_FREQ is just the coeff used to achieve a length of PERIOD.
ANGULAR_FREQ = 2.0 * np.pi / PERIOD
# Period scale is just used to receive proper values from the FourierExtrapolator.
PERIOD_SCALE=N/PERIOD


# Consumer should be returning data points with a seasonality that could be
# captured by forward Fast Fourier Transform.
# The data points should be wave like and should have a seasonality.
# The season consists of 1000 data points and then the pattern should be repeated.
# It is because this is how Fourier works.
class Consumer(abc.ABC):
    # Should return the amount of required resrc at time t.
    def required_resrc(self, t: int) -> float:
        raise NotImplementedError("")


class CosConsumer(Consumer):
    _ampli: float
    _freq: float
    _phase: float
    
    def __init__(self, ampli: float, freq: float, phase: float) -> None:
        super().__init__()
        
        self._ampli = ampli
        self._freq = freq
        self._phase = phase

    def required_resrc(self, t: int) -> float:
        return self._ampli + self._ampli * np.cos(self._freq * ANGULAR_FREQ * (t % N) + self._phase)


# TODO: Fix code repetition in some way... but I don't really care for now.
class SinConsumer(Consumer):
    _ampli: float
    _freq: float
    _phase: float
    
    def __init__(self, ampli: float, freq: float, phase: float) -> None:
        super().__init__()
        
        self._ampli = ampli
        self._freq = freq
        self._phase = phase

    def required_resrc(self, t: int) -> float:
        return self._ampli + self._ampli * np.sin(self._freq * ANGULAR_FREQ * (t % N) + self._phase)
