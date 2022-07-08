import logging

import matplotlib.pyplot as plt
import numpy as np

import autoscaler.converger as cvg
import autoscaler.app as app
import autoscaler.simulator as simulator


def main():
    logging.basicConfig(level=logging.INFO)

    extr = cvg.FourierExtrapolator()    
    a = app.App()

    # cons1 = app.SinConsumer(0.5, 3.0, 2.0)
    # cons2 = app.CosConsumer(10, 5, 0.2)
    # cons2 = app.CosConsumer(3, 2, 5)

    # a.add_consumer("cons1", cons1)
    # a.add_consumer("cons2", cons2)
    
    converger = cvg.Converger(0.6, extr, a)
    metri = app.MetriCollector(a)
    
    sim = simulator.Simulator(a, converger, metri)
    sim.run(3)

    win = metri.window(0, app.N * 3)
    ts = np.arange(0, app.N * 3)

    plt.figure(1)
    plt.title("Desired state deviation")
    plt.plot(ts, win.used_resrc_ratios, color="red", label="y", alpha=0.7)
    plt.legend()
    
    plt.show()


if __name__ == "__main__":
    main()
