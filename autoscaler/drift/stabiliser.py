import enum
import logging

import autoscaler.app as app
import autoscaler.converger as cvg
import autoscaler.drift.psi as psi
import autoscaler.drift.ks as ks


class DriftMetric(enum.Enum):
    PSI = "PSI"
    KS = "KS"


class Stabiliser:
    _metri: app.MetriCollector
    _extr: cvg.FourierExtrapolator
    _metric: DriftMetric
    _num_drifts: int
    
    def __init__(
        self, 
        metri: app.MetriCollector,
        extr: cvg.FourierExtrapolator,
        metric: DriftMetric = DriftMetric.PSI,
    ) -> None:
        self._metri = metri
        self._extr = extr
        self._metric = metric
        
        # starting from -1 because the initial learning isn't a drift.
        self._num_drifts = -1

    def stabilise(self, season: int) -> None:
        if not self._should_remodel(season):
            return

        logging.info("concept drift encountered")

        win = self._metri.window((season - 1) * app.N, season * app.N)
        self._extr.learn(win.used_resrc_vals, period_scale=1)

        self._num_drifts += 1

    def num_drifts(self) -> int:
        return self._num_drifts

    def _should_remodel(self, season: int) -> bool:
        if season == 0:
            return False

        if not self._extr.is_taught():
            return True

        win = self._metri.window((season - 1) * app.N, season * app.N)
        _, pred = self._extr.predict((season - 1) * app.N, season * app.N - 1)
        
        if self._metric == DriftMetric.PSI and psi.has_drift(pred, win.used_resrc_vals, buckets=25):
            return True

        if self._metric == DriftMetric.KS and ks.has_drift(pred, win.used_resrc_vals):
            return True

        return False
