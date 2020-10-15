import matplotlib.pyplot as plt
import json
import numpy as np
import sys
import objective
import heat
import plot_info
with open('objective_parameters.json') as f:
    objective_parameters = json.load(f)

coefficients_filename = sys.argv[1]
coefficients_per_iteration = np.loadtxt(coefficients_filename)

coefficients_true = objective_parameters['coefficients']

per_plot = 3

number_of_coefficients = len(coefficients_true)
number_of_plots = (number_of_coefficients + per_plot - 1) // per_plot
fig, axes = plt.subplots(1, number_of_plots + 1, sharey = True, figsize=(16, 8))
for coefficient_index in range(len(coefficients_true)):
    plot_index = coefficient_index // per_plot

    iterations = np.arange(0, coefficients_per_iteration.shape[0])
    plot_ref = axes[plot_index].plot(iterations, coefficients_per_iteration[:, coefficient_index + 1], '-o',
        label=f'$a_{{{coefficient_index}}}$')
    axes[plot_index].plot(iterations, coefficients_true[coefficient_index]*np.ones_like(coefficients_per_iteration[:, coefficient_index]),
        '--', color=plot_ref[0].get_color())
    axes[plot_index].grid(True)
    axes[plot_index].legend()

    axes[plot_index].set_xlabel("Iteration")
plot_ref = axes[number_of_plots].plot(iterations, coefficients_per_iteration[:,0], '-o', label='q')
axes[number_of_plots].plot(iterations, np.zeros_like(coefficients_per_iteration[:,0])*objective_parameters['q'], '--', label='q',
    color=plot_ref[0].get_color())
axes[number_of_plots].grid(True)
axes[number_of_plots].legend()

axes[number_of_plots].set_xlabel("Iteration")
plot_info.showAndSave("coefficients")

dx = 1.0/2048
x = np.arange(0,1, dx)

objective_function = objective.Objective(**objective_parameters)
initial_data = objective_function.initial_data


end_time = objective_parameters['end_time']
#plt.plot(x, initial_data.exact_solution(x, end_time), label='exact true')

objective_function_approximated = objective.Objective(end_time=end_time, coefficients=coefficients_per_iteration[-1,1:],
    q=coefficients_per_iteration[-1,0], control_points=objective_parameters['control_points'])

plt.plot(x, objective_function_approximated.initial_data(x), '--', label='Initial data')
plt.plot(x, objective_function_approximated.initial_data.exact_solution(x, end_time), label='Evolved data')
plt.plot(objective_parameters['control_points'],
    initial_data.exact_solution(np.array(objective_parameters['control_points']), end_time),
     '*', label='Control points', markersize=15)
plt.legend()
plot_info.showAndSave("exact_with_coefficients_from_ismo")
