import logging
import numpy as np

import autoscaler.converger.fourier as fourier
import autoscaler.app as app


class Converger(fourier.FourierExtrapolator):
    _FORESEEABLE_BATCH_SIZE = 512
    
    _desired_resrc_ratio: float
    _extr: fourier.FourierExtrapolator
    _app: app.App
    # closed interval (i.e. inclusive)
    _pred_start: int
    _known_until_t: int
    _predicted_load: np.ndarray

    def __init__(self, desired_resrc_ratio: float, extr: fourier.FourierExtrapolator, app: app.App) -> None:
        self._desired_resrc_ratio = desired_resrc_ratio
        self._extr = extr
        self._app = app
        self._known_until_t = -1
        self._predicted_load = np.array(0)

    # perform convergence for moment t.
    def converge(self, t: int) -> None:
        if self._extr.is_taught():
            self._converge_with_prediction(t)
            return

        self._converge_with_reaction(t)
        
    def _converge_with_reaction(self, t: int) -> None:
        if t == 0:
            return
        
        prev_used_ratio = self._app.used_resrc_ratio(t-1)
        prev_used = self._app.used_resrc(t-1)
        
        desired_resrc = (prev_used / prev_used_ratio) * self._desired_resrc_ratio
        self._app.set_resrc(desired_resrc)

    def _converge_with_prediction(self, t: int) -> None:
        if self._known_until_t < t:
            self._foresee_batch(t)

        load = self._predicted_load[t - self._pred_start]
        
        desired_resrc = (100.0 * load) / self._desired_resrc_ratio
        self._app.set_resrc(desired_resrc)
        
        logging.debug(f"set desired_resrc to {desired_resrc}")

    def _foresee_batch(self, start_t: int) -> None:
        logging.info(f"foreseeing batch of {self._FORESEEABLE_BATCH_SIZE}")

        _, self._predicted_load  = self._extr.predict(start_t, start_t + self._FORESEEABLE_BATCH_SIZE)
        
        self._pred_start = start_t
        self._known_until_t = start_t + self._FORESEEABLE_BATCH_SIZE
