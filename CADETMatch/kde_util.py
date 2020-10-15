from sklearn.base import clone
import numpy
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KernelDensity
import SALib.sample.sobol_sequence
import CADETMatch.util as util
from sklearn import preprocessing
import multiprocessing
import itertools
import time

def fit(x):
    bw, kde_params, scores = x
    kde = KernelDensity(**kde_params)
    kde.set_params(bandwidth=bw)
    cross_val = cross_val_score(kde, scores, cv=20)
    return cross_val, bw

def get_bandwidth(kde, scores):
    start = time.time()
    map = util.getMapFunction()
    
    bw = numpy.logspace(-5, 0, 80)
    
    results = list(map(fit, zip(bw, itertools.repeat(kde.get_params()), itertools.repeat(scores))))

    cross_val, bw = zip(*results)

    cross_val = numpy.array(cross_val)
    bw = numpy.array(bw)

    sort_index = numpy.argsort(bw)

    bw = bw[sort_index]
    cross_val = cross_val[sort_index, :]

    scaler = preprocessing.MinMaxScaler()
    shape = cross_val.shape
    cross_val = scaler.fit_transform(cross_val.reshape(-1,1))
    cross_val = cross_val.reshape(shape)

    mean_value = numpy.mean(cross_val, 1)

    bandwidth = interp_bw(bw, mean_value, cross_val)

    multiprocessing.get_logger().info("Bandwidth %s found in %s seconds", bandwidth, time.time() - start)

    store = numpy.array([bw, mean_value]).T

    kde_final = clone(kde)
    kde_final.set_params(bandwidth=bandwidth)

    return kde_final, bandwidth, store

def interp_bw(params, mean_value, values):
    index = numpy.argmax(mean_value)
    params = numpy.array(params)
    mean_value = numpy.array(mean_value)
    
    indexes = numpy.array([index-1, index, index+1])
    x = params[indexes]
    
    X = numpy.tile(x.reshape(-1,1), values.shape[1])
    Y = values[indexes,:]

    X = numpy.squeeze(X.reshape(-1,1))
    Y = numpy.squeeze(Y.reshape(-1,1))
    
    poly, res = numpy.polynomial.Polynomial.fit(X, Y,2, full=True)
    
    bw = poly.deriv().roots()[0]
    
    return bw