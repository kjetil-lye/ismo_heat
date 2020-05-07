import numpy as np


class Objective(object):
    def __init__(self, coefficients=[1, 2, 3, 4, 5], q=1.0, end_time=1.0,
                 control_points=[0.125, 0.25, 0.5, 0.75, 0.825]):
        self.coefficients = coefficients
        self.end_time = end_time
        self.q = q
        self.control_points = control_points

    def exact_solution(self, x):
        exact_solution = sum(
            coefficient * np.exp(-self.q * (k * np.pi) ** 2 * self.end_time) * np.sin(k * np.pi * x) for k, coefficient
            in
            enumerate(self.coefficients))

        return exact_solution

    def __call__(self, solution):

        assert (len(solution) == len(self.control_points))

        error = 0
        for control_point_index, control_point in enumerate(self.control_points):
            exact_solution = self.exact_solution(control_point)

            error += (exact_solution - solution[control_point_index]) ** 2

        return 0.5 * error

    def grad(self, solution):
        gradient = np.zeros(len(self.control_points))
        for control_point_index, control_point in enumerate(self.control_points):
            exact_solution = self.exact_solution(control_point)

            gradient[control_point_index] = (solution[control_point_index] - exact_solution)

        return gradient
