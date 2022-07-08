import logging
import autoscaler.converger as cvg
import autoscaler.app as app


def main():
    logging.basicConfig(level=logging.DEBUG)

    extr = cvg.FourierExtrapolator()    
    a = app.App()

    cons1 = app.SinConsumer(0.5, 3.0, 2.0)
    cons2 = app.CosConsumer(0.2, 3.5, 0.2)

    a.add_consumer("cons1", cons1)
    a.add_consumer("cons2", cons2)
    
    converger = cvg.Converger(0.6, extr, a)
    
    for i in range(1000):
        converger.converge(i)
        print(a.used_resrc_ratio(i))


if __name__ == "__main__":
    main()
