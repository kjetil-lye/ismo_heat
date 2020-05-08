import numpy


class InitialDataControlSine:
    def __init__(self, coefficients):
        self.coefficients = coefficients

    def __call__(self, x):
        u = numpy.zeros_like(x)

        for k, coefficient in enumerate(self.coefficients):
            u += coefficient * numpy.exp((k * numpy.pi) ** 2) * numpy.sin(k * numpy.pi * x)

        return u

    def exact_solution(self, x, t, q=1):
        return sum(coefficient * numpy.exp((k * numpy.pi) ** 2) * numpy.exp(-q * (k * numpy.pi) ** 2 * t) * numpy.sin(
            k * numpy.pi * x) for k, coefficient in enumerate(self.coefficients))
