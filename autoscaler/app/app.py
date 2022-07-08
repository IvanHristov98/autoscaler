from typing import Dict, List

import autoscaler.app.consumer as cons


# App is an abstraction of an application.
# It should probably contain a couple of consumers that it should ping for moment t
# to learn what their available resource are.
class App:
    _INITIAL_RESRC_CAPACITY = 10.0

    _resrc_capacity: float
    _consumers: Dict[str, cons.Consumer]

    def __init__(self) -> None:
        self._resrc_capacity = self._INITIAL_RESRC_CAPACITY
        self._consumers = dict()

    def add_consumer(self, name: str, consumer: cons.Consumer) -> None:
        self._consumers[name] = consumer

    def rem_consumer(self, name: str) -> None:
        del self._consumers[name]

    def consumer_names(self) -> List[str]:
        return self._consumers.keys()

    # used_resrc should return the amount of used rsc in % for moment t.
    def used_resrc_ratio(self, t: int) -> float:        
        return self.used_resrc(t) / self._resrc_capacity

    def used_resrc(self, t: int) -> float:
        used = 0.0
        
        for consumer in self._consumers.values():
            used += abs(consumer.required_resrc(t))

        return min(used, self._resrc_capacity)

    # set_resrc sets the available resource for the app.
    def set_resrc_capacity(self, resrc: float) -> None:
        self._resrc_capacity = resrc
