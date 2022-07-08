import logging

import matplotlib.pyplot as plt
import numpy as np

import autoscaler.converger as cvg
import autoscaler.app as app


def main():
    logging.basicConfig(level=logging.INFO)

    extr = cvg.FourierExtrapolator()    
    a = app.App()

    cons1 = app.SinConsumer(0.5, 3.0, 2.0)
    cons2 = app.CosConsumer(10, 5, 0.2)
    cons2 = app.CosConsumer(3, 2, 5)

    a.add_consumer("cons1", cons1)
    a.add_consumer("cons2", cons2)
    
    converger = cvg.Converger(0.6, extr, a)
    metri = app.MetriCollector(a)
    
    for i in range(2000):
        converger.converge(i)
        metri.collect(i)

    win = metri.window(0, 2000)
    ts = np.arange(0, 2000)
    
    plt.figure(1)
    plt.title("Desired state deviation")
    plt.plot(ts, win, color="red", label="y", alpha=0.7)
    plt.legend()
    
    plt.show()
    


if __name__ == "__main__":
    main()
