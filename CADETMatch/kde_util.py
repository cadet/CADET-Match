import itertools
import multiprocessing
import time

import numpy
import SALib.sample.sobol_sequence
from sklearn import preprocessing
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KernelDensity

import CADETMatch.util as util


def fit(x):
    bw, kde_params, scores = x
    kde = KernelDensity(**kde_params)
    kde.set_params(bandwidth=bw)
    cross_val = cross_val_score(kde, scores, cv=20)
    return cross_val, bw

def mirror_double(data):
    #mirror below and above data to catch boundaries better
    data_max = numpy.max(data)
    data_min = numpy.min(data)

    data_temp = data_max - data
    data_mask = numpy.ma.masked_equal(data_temp, 0.0, copy=False)
    delta_max_value = data_mask.min()

    data_temp = data - data_min
    data_mask = numpy.ma.masked_equal(data_temp, 0.0, copy=False)
    delta_min_value = data_mask.min()

    data_mirror_lb = numpy.copy(data)[::-1] - data_max - delta_min_value
    
    data_mirror_ub = numpy.copy(data)[::-1] + data_max + delta_max_value

    full_data = numpy.vstack([data_mirror_lb, data, data_mirror_ub])

    return full_data

def plot_kde(data, mirror=True):
    "take a 1d data set and return x and probability, handles scaling bandwidth etc"
    data = data.reshape(-1,1)
    scaler = preprocessing.RobustScaler().fit(data)

    x = numpy.linspace(min(data), max(data), 1000)

    kde = KernelDensity(kernel="gaussian")

    if mirror:
        data_mirror = mirror_double(data)
        mirror_mult = 3
    else:
        data_mirror = data
        mirror_mult = 1

    data_scaled = scaler.transform(data_mirror)
    x_scaled = scaler.transform(x)

    kde, bandwidth, store = get_bandwidth(kde, data_scaled)

    kde.fit(data_scaled)

    prob = mirror_mult * numpy.exp(kde.score_samples(x_scaled))

    return x, prob



def get_bandwidth(kde, scores):
    start = time.time()
    map = util.getMapFunction()

    bw = numpy.logspace(-5, 0, 80)

    results = list(
        map(fit, zip(bw, itertools.repeat(kde.get_params()), itertools.repeat(scores)))
    )

    cross_val, bw = zip(*results)

    cross_val = numpy.array(cross_val)
    bw = numpy.array(bw)

    sort_index = numpy.argsort(bw)

    bw = bw[sort_index]
    cross_val = cross_val[sort_index, :]

    scaler = preprocessing.MinMaxScaler()
    shape = cross_val.shape
    cross_val = scaler.fit_transform(cross_val.reshape(-1, 1))
    cross_val = cross_val.reshape(shape)

    mean_value = numpy.mean(cross_val, 1)

    bandwidth = interp_bw(bw, mean_value, cross_val)

    multiprocessing.get_logger().info(
        "Bandwidth %s found in %s seconds", bandwidth, time.time() - start
    )

    store = numpy.array([bw, mean_value]).T

    kde_final = clone(kde)
    kde_final.set_params(bandwidth=bandwidth)

    return kde_final, bandwidth, store


def interp_bw(params, mean_value, values):
    index = numpy.argmax(mean_value)
    params = numpy.array(params)
    mean_value = numpy.array(mean_value)

    indexes = numpy.array([index - 1, index, index + 1])
    x = params[indexes]

    X = numpy.tile(x.reshape(-1, 1), values.shape[1])
    Y = values[indexes, :]

    X = numpy.squeeze(X.reshape(-1, 1))
    Y = numpy.squeeze(Y.reshape(-1, 1))

    poly, res = numpy.polynomial.Polynomial.fit(X, Y, 2, full=True)

    bw = poly.deriv().roots()[0]

    return bw
