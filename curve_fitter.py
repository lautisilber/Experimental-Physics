import numpy as np
from scipy.optimize import curve_fit


def fit_curve(fun, data_x, data_y, guess, iterations=10):
    popt = guess
    for _ in range(iterations):
        popt, pcov = curve_fit(fun, data_x, data_y, p0=popt)
    fit_func = lambda x: fun(x, *popt)
    return popt, pcov, fit_func


def fit_lorentz(data_x, data_y, guess=[1, 1, 1, 1], iterations=10):
    ''' A lorentzian peak with:
       Constant Background          : p[0]
       Peak height above background : p[1]
       Central value                : p[2]
       Full Width at Half Maximum   : p[3]
    '''
    def lorentzian_1d(x, *p):
        return np.abs(p[0])+(p[1])/(1.0+((x-p[2])/p[3])**2)

    return fit_curve(lorentzian_1d, data_x, data_y, guess, iterations)


def fit_gauss(data_x, data_y, guess=[1, 1, 1], iterations=10):
    ''' A lorentzian peak with:
       Amplitude : A
       Center    : x0
       Sigma     : sigma
    '''
    def gauss(x, A, x0, sigma):
        return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
    
    return fit_curve(gauss, data_x, data_y, guess, iterations)


def poly_fit(data_x, data_y, degree=1):
    return np.polyfit(data_x, data_y, degree)


def linear_fit_to_origin(data_x, data_y, guess=[1], iterations=10):
    def linear_to_origin(x, m):
        return m*x

    return fit_curve(linear_to_origin, data_x, data_y, guess, iterations)
    
