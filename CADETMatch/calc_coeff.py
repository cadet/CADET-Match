import numpy


def linear_coeff(x1, y1, x2, y2):
    "return paramters that fit y = a*x+b"
    a = (y2 - y1) / (x2 - x1)
    b = y2 - a * x2
    return a, b


def exponential_coeff(x1, y1, x2, y2):
    "return paramaters that fit y = b*exp(m*x)"
    b = (numpy.log(y2) - numpy.log(y1)) / (x2 - x1)
    a = y1 * numpy.exp(-b * x1)
    return a, b


def exponential(x, a, b):
    return a * numpy.exp(b * x)


def linear(x, a, b):
    return a * x + b
