import unittest
import heat
import numpy as np


class TestHeatEquation(unittest.TestCase):
    def test_zero(self):
        initial_data = lambda x: 0
        dt = 1 / 1024.
        dx = dt

        end_time = 1.25

        solution_to_heat_equation = heat.solve_heat_equation(initial_data, dt, dx, end_time)

        self.assertEqual(int(1 / dt), solution_to_heat_equation.shape[0])
        self.assertTrue(np.all(solution_to_heat_equation == np.zeros_like(solution_to_heat_equation)))

    def test_sine_single(self):
        # we do a quick convergence test to make sure it is indeed second order
        initial_data = lambda x: np.sin(np.pi * x)

        resolutions = 2.0 ** np.arange(-5, -12, -1)
        errors = []

        end_time = 1.25

        for dx in resolutions:
            dt = dx

            solution_to_heat_equation = heat.solve_heat_equation(initial_data, dt, dx, end_time)

            self.assertEqual(int(1 / dx), solution_to_heat_equation.shape[0])

            x = np.arange(0, 1, dx)

            exact_solution = np.exp(-np.pi ** 2 * end_time) * initial_data(x)

            difference_in_l2_norm = np.linalg.norm((exact_solution - solution_to_heat_equation) * dx, ord=2)

            errors.append(difference_in_l2_norm)

        convergence_rate = np.polyfit(np.log(resolutions), np.log(errors), 1)[0]

        self.assertGreaterEqual(convergence_rate, 2)

    def test_sine_three_modes(self):
        # we do a quick convergence test to make sure it is indeed second order
        resolutions = 2.0 ** np.arange(-5, -12, -1)
        errors = []

        end_time = 1.25

        coefficients = [0.4, 0.2, 0.7]
        for dx in resolutions:
            dt = dx

            initial_data = heat.InitialDataControlSine(coefficients)

            solution_to_heat_equation = heat.solve_heat_equation(initial_data, dt, dx, end_time)

            self.assertEqual(int(1 / dx), solution_to_heat_equation.shape[0])

            x = np.arange(0, 1, dx)

            exact_solution = initial_data.exact_solution(x, end_time)

            difference_in_l2_norm = np.linalg.norm((exact_solution - solution_to_heat_equation) * dx, ord=2)

            errors.append(difference_in_l2_norm)

        convergence_rate = np.polyfit(np.log(resolutions), np.log(errors), 1)[0]

        self.assertGreaterEqual(convergence_rate, 2)

    def test_sine_single_different_coefficient(self):
        # we do a quick convergence test to make sure it is indeed second order
        initial_data = lambda x: np.sin(np.pi * x)

        resolutions = 2.0 ** np.arange(-5, -12, -1)
        errors = []

        end_time = 1.25

        q = 0.8
        for dx in resolutions:
            dt = dx

            solution_to_heat_equation = heat.solve_heat_equation(initial_data, dt, dx, end_time, q=q)

            self.assertEqual(int(1 / dx), solution_to_heat_equation.shape[0])

            x = np.arange(0, 1, dx)

            exact_solution = np.exp(-q * np.pi ** 2 * end_time) * initial_data(x)

            difference_in_l2_norm = np.linalg.norm((exact_solution - solution_to_heat_equation) * dx, ord=2)

            errors.append(difference_in_l2_norm)

        convergence_rate = np.polyfit(np.log(resolutions), np.log(errors), 1)[0]

        self.assertGreaterEqual(convergence_rate, 1.9)

    def test_sine_three_modes_different_coefficient(self):
        # we do a quick convergence test to make sure it is indeed second order
        resolutions = 2.0 ** np.arange(-5, -12, -1)
        errors = []

        end_time = 1.25

        coefficients = [0.4, 0.2, 0.7]

        q = 1.3
        for dx in resolutions:
            dt = dx

            initial_data = heat.InitialDataControlSine(coefficients)

            solution_to_heat_equation = heat.solve_heat_equation(initial_data, dt, dx, end_time, q=q)

            self.assertEqual(int(1 / dx), solution_to_heat_equation.shape[0])

            x = np.arange(0, 1, dx)

            exact_solution = initial_data.exact_solution(x, end_time, q)

            difference_in_l2_norm = np.linalg.norm((exact_solution - solution_to_heat_equation) * dx, ord=2)

            errors.append(difference_in_l2_norm)

        convergence_rate = np.polyfit(np.log(resolutions), np.log(errors), 1)[0]

        self.assertGreaterEqual(convergence_rate, 1.9)


if __name__ == '__main__':
    unittest.main()
