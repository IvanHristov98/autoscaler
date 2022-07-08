import logging

import matplotlib.pyplot as plt
import numpy as np

import autoscaler.app as app
import autoscaler.drift as drift
import autoscaler.converger as cvg
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
    stabiliser = drift.Stabiliser(metri, extr)
    
    num_seasons = 6
    
    sim = simulator.Simulator(a, converger, metri, stabiliser)
    sim.run(num_seasons)

    win = metri.window(0, app.N * num_seasons)
    ts = np.arange(0, app.N * num_seasons)

    plt.figure(1)
    plt.title("Desired state deviation")
    plt.plot(ts, win.used_resrc_ratios, color="red", label="y", alpha=0.7)
    plt.legend()
    
    win = metri.window(4000, app.N * 6)
    predicted_t, predicted_y = extr.predict(4000, 5999)
    
    plt.figure(2)
    plt.title("Fourier prediction")
    
    plt.plot(predicted_t, win.used_resrc_vals, color="red", label="y", alpha=0.7)
    plt.plot(predicted_t, predicted_y, color="green", label="predicted y", alpha=0.7)
    plt.legend()

    plt.show()
    
    plt.show()


if __name__ == "__main__":
    main()
