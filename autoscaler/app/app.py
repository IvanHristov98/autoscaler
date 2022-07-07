# App is an abstraction of an application.
# It should probably contain a couple of consumers that it should ping for moment t
# to learn what their available resource are.
class App:
    def __init__(self) -> None:
        pass

    # used_resrc should return the amount of used rsc in % for moment t.
    def used_resrc_ratio(self, t: int) -> float:
        return 0

    def used_resrc(self, t: int) -> float:
        return 0

    # set_resrc sets the available resource for the app.
    def set_resrc(self, resrc: float) -> None:
        pass
