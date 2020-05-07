import json

import numpy as np
import os.path
import sys
from heat import solve_heat_equation, InitialDataControlSine

if __name__ == '__main__':
    print(f"Command line: {' '.join(sys.argv)}")
    import argparse
    from mpi4py import MPI

    parser = argparse.ArgumentParser(description="""
Runs some complicated function on the input parameters
    """)

    parser.add_argument('--input_parameters_file', type=str, required=True,
                        help='Input filename for the parameters (readable by np.loadtxt)')

    parser.add_argument('--output_values_files', type=str, required=True, nargs="+",
                        help='Output filename for the values (will be written by np.savetxt)')

    parser.add_argument('--starting_sample', type=int, required=True,
                        help='The starting id of the first sample')

    parser.add_argument('--iteration_number', type=int, required=True,
                        help='The iteration number')

    parser.add_argument('--start', type=int, default=0,
                        help='Starting index to read out of the parameter file, by default reads from start of file')

    parser.add_argument('--end', type=int, default=-1,
                        help='Ending index (exclusive) to read out of the parameter file, by default reads to end of file')

    parser.add_argument('--output_append', action='store_true',
                        help='Append output to end of file')

    args = parser.parse_args()

    starting_sample_id = args.starting_sample
    iteration_number = args.iteration_number

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    number_of_processes = comm.Get_size()

    if args.end != -1:
        parameters = np.loadtxt(args.input_parameters_file)[args.start:args.end]
    else:
        parameters = np.loadtxt(args.input_parameters_file)[args.start:]

    assert (parameters.shape[0] >= number_of_processes)

    # Load configuration so we can get the control points
    with open("objective_parameters.json") as objective_parameters_file:
        objective_parameters = json.load(objective_parameters_file)

    control_points = objective_parameters['control_points']
    end_time = objective_parameters['end_time']
    dx = 1.0 / 2048
    dt = dx

    values = np.zeros((parameters.shape[0], len(control_points)))

    number_of_parameters_per_process = (parameters.shape[0] + number_of_processes - 1) // number_of_processes
    start_parameter = rank * number_of_parameters_per_process
    end_parameter = min(parameters.shape[0], (rank + 1) * number_of_parameters_per_process)

    x = np.linspace(0, 1, int(1 / dx))
    control_point_indices = [abs(control_point - x).argmin() for control_point in control_points]

    for parameter_index in range(start_parameter, end_parameter):
        q = parameters[parameter_index, 0]
        coefficients = parameters[parameter_index, 1:]

        initial_data = InitialDataControlSine(coefficients)

        solution = solve_heat_equation(initial_data, dt, dx, end_time, q=q)

        for output_index, (control_point_index, control_point) in enumerate(zip(control_point_indices, control_points)):
            values[parameter_index, output_index] = solution[control_point_index]
    comm.Barrier()

    if rank == 0:
        for k in range(values.shape[1]):
            values_to_write = values[:, k]
            if args.output_append:
                if os.path.exists(args.output_values_files[k]):
                    previous_values = np.loadtxt(args.output_values_files[k])

                    new_values = np.zeros((values.shape[0] + previous_values.shape[0]))

                    new_values[:previous_values.shape[0]] = previous_values
                    new_values[previous_values.shape[0]:] = values[:,k]

                    values_to_write = new_values
            np.savetxt(args.output_values_files[k], values_to_write)
