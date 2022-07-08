import logging
import random
from typing import List, Tuple

import numpy as np

import autoscaler.app as app
import autoscaler.converger as cvg
import autoscaler.drift as drift


class Metrics:
    concept_drift_detection_ratio: float
    relative_loss: float


class Simulator:
    _LOW_AMPLI = 0.1
    _HIGH_AMPLI = 2
    _LOW_FREQ = 0.02
    _HIGH_FREQ = 2
    _LOW_PHASE = 0.0
    _HIGH_PHASE = np.pi * 2
    
    _a: app.App
    _converger: cvg.Converger
    _metri: app.MetriCollector
    _stabiliser: drift.Stabiliser
    _cons_count: int
    _moment: int
    
    def __init__(
        self, 
        a: app.App, 
        converger: cvg.Converger,
        metri: app.MetriCollector,
        stabiliser: drift.Stabiliser,
    ) -> None:
        self._a = a
        self._converger = converger
        self._metri = metri
        self._stabiliser = stabiliser

        self._cons_count = 0
        self._moment = 0
        
        self._add_consumers(upto=25)

    def run(self, num_seasons: int) -> None:
        for season in range(num_seasons):
            logging.info(f"running season {season}")

            self._stabiliser.stabilise(season)
            self._run_season(season)

        logging.info(f"drift counts {self._stabiliser.num_drifts()}")

    def _run_season(self, season: int) -> None:
        # optionally remove and add consumers.
        if season % 3 == 0:
            logging.info(f"concept actually drifted")
            self._rem_consumers()
            self._add_consumers()
            
        for _ in range(app.N):
            self._converger.converge(self._moment)
            self._metri.collect(self._moment)

            self._moment += 1

    def _rem_consumers(self, upto: int = 5) -> None:
        num = random.randint(0, upto)
        names = set(self._a.consumer_names())

        for _ in range(num):
            if len(names) <= 0:
                break
            
            names.remove(random.choice(tuple(names)))            

    def _add_consumers(self, upto: int = 5) -> None:
        num = random.randint(0, upto)
        consumers = self._gen_consumers(num)

        for consumer in consumers:
            self._a.add_consumer(f"{self._cons_count}", consumer)
            self._cons_count += 1

    def _gen_consumers(self, num: int) -> List[app.Consumer]:
        consumers = [None] * num
        
        for i in range(num):
            consumers[i] = self._gen_consumer()

        return consumers

    def _gen_consumer(self) -> app.Consumer:
        side = random.randint(0, 1)
        
        if side == 0:
            return self._gen_sin_consumer()

        return self._gen_cos_consumer()

    def _gen_sin_consumer(self) -> app.Consumer:
        ampli, freq, phase = self._generate_random_consumer_cfg()

        return app.SinConsumer(ampli, freq, phase)

    def _gen_cos_consumer(self) -> app.Consumer:
        ampli, freq, phase = self._generate_random_consumer_cfg()
        
        return app.CosConsumer(ampli, freq, phase)

    def _generate_random_consumer_cfg(self) -> Tuple[float, float, float]:
        ampli = self._LOW_AMPLI + random.random() * (self._HIGH_AMPLI - self._LOW_AMPLI)
        freq = self._LOW_FREQ + random.random() * (self._HIGH_FREQ - self._LOW_FREQ)
        phase = self._LOW_PHASE + random.random() * (self._HIGH_PHASE - self._LOW_PHASE)

        return ampli, freq, phase
