import matplotlib.pyplot as plt
import json
import numpy
import sys
import objective
import heat
import plot_info
with open('objective_parameters.json') as f:
    objective_parameters = json.load(f)

objective_function = objective.Objective(**objective_parameters)


initial_data = objective_function.initial_data

dt = 1.0/2048
dx = 1.0/2048
end_time = objective_parameters['end_time']
solution = heat.solve_heat_equation(initial_data, dt, dx, end_time)
x = numpy.arange(0,1, dx)
plt.plot(x, initial_data(x), label='initial')

plt.plot(x, solution, '*', label='numerical')
plt.plot(x, initial_data.exact_solution(x, end_time), label='exact')
plt.legend()
plot_info.showAndSave("heat_exact_solution")
