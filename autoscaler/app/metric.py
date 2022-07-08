from typing import List

import numpy as np

import autoscaler.app.app as app


class MetriCollector:
    _a: app.App
    _used_resrc_vals: List[int]
    _curr_time: int
    
    def __init__(self, a: app.App) -> None:
        self._used_resrc_vals = []
        self._a = a
        self._curr_time = -1

    def collect(self, t: int) -> None:
        # Rebuild history through this cycle if needed
        for i in range(self._curr_time+1, t+1):
            used_resrc = self._a.used_resrc(i)
            self._used_resrc_vals.append(used_resrc)

        self._curr_time = t

    # the window is [start; end)
    def window(self, start: int, end: int) -> np.ndarray:
        vals = np.zeros(end-start)
        
        if len(self._used_resrc_vals) < end:
            raise Exception(f"fetching uncollected window [{start};{end})")

        for t in range(start, end):
            vals[t] = self._used_resrc_vals[t]

        return vals
