import numpy


class InitialDataControlSine:
    def __init__(self, coefficients):
        self.coefficients = coefficients

    def __call__(self, x):
        u = numpy.zeros_like(x)

        for k, coefficient in enumerate(self.coefficients):
            u += coefficient * numpy.sin(k*numpy.pi*x)

        return u
