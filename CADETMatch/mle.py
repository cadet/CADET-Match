import numpy
from sklearn import preprocessing
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import cross_val_score
import scipy
import scoop

import nsga2_simple

def reduce_data(data, size):
    lb, ub = numpy.percentile(data, [5, 95], 0)
    selected = (data >= lb) & (data <= ub)

    selected = numpy.all(selected,1)

    data = data[selected,:]

    shape = data.shape

    if size < data.shape[0]:
        indexes = numpy.random.choice(data.shape[0], size, replace=False)
        data_reduced = data[indexes]
    else:
        data_reduced = data

    scaler = preprocessing.StandardScaler().fit(data)
    data = scaler.transform(data)

    data_reduced = scaler.transform(data_reduced)

    return data, data_reduced, scaler

def bandwidth_score(data, kernel, atol, bw):
    bandwidth = 10**bw[0]
    kde_bw = KernelDensity(kernel=kernel, bandwidth=bandwidth, atol=atol)
    scores = cross_val_score(kde_bw, data, cv=3)
    return [-max(scores),]

def goal_kde(kde, x):
    test_value = numpy.array(x).reshape(1, -1)
    score = kde.score_samples(test_value)
    return [-score[0],]

def get_mle(data):
    atol = 1e-4
    kernel = 'gaussian'

    data, data_reduced, scaler = reduce_data(data, 4000)

    BOUND_LOW_num = numpy.min(data, 0)
    BOUND_UP_num = numpy.max(data, 0)

    BOUND_LOW_trans = list(BOUND_LOW_num)
    BOUND_UP_trans = list(BOUND_UP_num)    

    BOUND_LOW_real = [0.0] * len(BOUND_LOW_trans)
    BOUND_UP_real = [1.0] * len(BOUND_UP_trans)

    result_bw = nsga2_simple.nsga2_min(bandwidth_score, [-3.0,], [1.0,], args=(data_reduced, kernel, atol), mu=16)

    bw = 10**result_bw['x'][0]

    kde_ga = KernelDensity(kernel=kernel, bandwidth=bw, atol=atol)
    kde_ga = kde_ga.fit(data)

    result_kde = nsga2_simple.nsga2_min(goal_kde, BOUND_LOW_trans, BOUND_UP_trans, args=(kde_ga,))

    x = list(scaler.inverse_transform(numpy.array(result_kde['x']).reshape(1, -1))[0])

    return x

