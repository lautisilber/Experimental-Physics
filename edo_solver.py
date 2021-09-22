import sys
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def solve_edo(system, ti, tf, y0, *args):
    '''
        def system(t, y, ...)
    '''
    sol = solve_ivp(fun=system, t_span=(ti, tf), y0=y0, args=args, dense_output=True)
    return sol.t, sol.y, sol.sol


def plot_sol(sol_t, sol_y, sol_sol, density=500):
    plt.scatter(sol_t, sol_y, label='solution')
    ts = np.linspace(sol_t[0], sol_t[-1], density)
    plt.plot(ts, sol_sol(ts), label='dense')
    plt.legend()
    plt.show()