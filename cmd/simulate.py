import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly

import autoscaler.converger as cvg
import autoscaler.app as app


def main():
    extr = cvg.FourierExtrapolator()
    a = app.App()
    
    converger = cvg.Converger(0.6, extr, a)

    converger.converge(0)
    a.used_resrc_ratio(0)


if __name__ == "__main__":
    main()
