import numpy
import scipy.sparse
import scipy.sparse.linalg


def solve_heat_equation(initial_data: callable(numpy.ndarray), dt: float, dx: float, end_time: float, a: float = 0.0,
                        b: float = 1):
    """
    Solves the heat equation on the domain [a, b]
    :param b: end point of domain
    :param a: start point of domain
    :param initial_data: a function that can compute the initial data
    :param dt: time step size
    :param dx: spatial step size
    :param end_time: final time
    :return: solution to the heat equation at end_time
    """
    x = numpy.arange(a, b, dx)
    number_of_spatial_points = x.shape[0]

    # for boundary conditions
    u = numpy.zeros(number_of_spatial_points + 2)
    u[1:-1] = initial_data(x)
    h = dt / dx ** 2

    # Time loop
    t = 0
    while t < end_time:

        # Create sparse matrix
        A = scipy.sparse.lil_matrix((number_of_spatial_points, number_of_spatial_points))

        # Loop over matrix entries
        for j in range(1, number_of_spatial_points + 1):  # j = 1,..., N
            A[j - 1, j - 1] = 1 + h

            if j > 1:
                A[j - 1, j - 2] = -h / 2
            if j < number_of_spatial_points:
                A[j - 1, j] = -h / 2

        # Our sparse solver likes the CSR format better
        A = A.tocsr()

        # Bulid RHS
        F = h / 2 * u[:-2] + h / 2 * u[2:] + (1 - h) * u[1:-1]

        # Solve matrix system

        u[1:-1] = scipy.sparse.linal.spsolve(A, F)

        # Boundary conditions are handled automatically since
        # U is zero everywhere in the beginning

        # update time
        t += dt

    return u
