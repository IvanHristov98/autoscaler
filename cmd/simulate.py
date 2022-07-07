import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly

import autoscaler.fourier as fourier


def main() -> None:
    # time setup
    n = 2000 # points per period
    period = 100
    angular_freq = 2.0 * np.pi / period
    
    # generation of signals
    t = np.linspace(0, period, n)
    y1 = 1.0 * np.cos(5.0 * angular_freq * t + 5)
    y2 = 1.0 * np.sin(10.0 * angular_freq * t)
    y3 = 0.5 * np.sin(20.0 * angular_freq * t)

    y = y1 + y2 + y3
    
    extr = fourier.FourierExtrapolator(y, poly.Polynomial((1, 1)), period_scale=n/period)
    predicted_t, predicted_y = extr.predict(0, 150)

    plt.figure(1)
    plt.title("Fourier prediction")
    plt.plot(t, y, color="red", label="y", alpha=0.7)
    plt.plot(predicted_t, predicted_y, color="green", label="predicted y", alpha=0.7)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
