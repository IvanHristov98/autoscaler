from typing import List, NamedTuple

import numpy as np

import autoscaler.app.app as app


class WindowBundle(NamedTuple):
    used_resrc_vals: np.ndarray
    used_resrc_ratios: np.ndarray


class MetriCollector:
    _a: app.App
    _used_resrc_vals: List[int]
    _used_resrc_ratios: List[float]
    _curr_time: int
    
    def __init__(self, a: app.App) -> None:
        self._used_resrc_vals = []
        self._used_resrc_ratios = []
        self._a = a
        self._curr_time = -1

    def collect(self, t: int) -> None:
        # Rebuild history through this cycle if needed
        for i in range(self._curr_time+1, t+1):
            used_resrc = self._a.used_resrc(i)
            self._used_resrc_vals.append(used_resrc)
            
            used_resrc_ratio = self._a.used_resrc_ratio(i)
            self._used_resrc_ratios.append(used_resrc_ratio)

        self._curr_time = t

    # the window is [start; end)
    def window(self, start: int, end: int) -> WindowBundle:
        vals = np.zeros(end-start)
        ratios = np.zeros(end-start)
        
        if len(self._used_resrc_vals) < end:
            raise Exception(f"fetching uncollected window [{start};{end})")

        for t in range(start, end):
            vals[t] = self._used_resrc_vals[t]
            ratios[t] = self._used_resrc_ratios[t]

        return WindowBundle(
            used_resrc_vals=vals,
            used_resrc_ratios=ratios,
        )
