import numpy
from sklearn import preprocessing
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import cross_val_score
import scipy
import scoop
from scoop import futures

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

def bandwidth_score(bw, data, kernel, atol):
    bandwidth = 10**bw[0]
    kde_bw = KernelDensity(kernel=kernel, bandwidth=bandwidth, atol=atol)
    scores = cross_val_score(kde_bw, data, cv=3)
    return -max(scores)

def goal_kde(x, kde):
    test_value = numpy.array(x).reshape(1, -1)
    score = kde.score_samples(test_value)
    return -score[0]

def get_mle(data):
    atol = 1e-4
    kernel = 'gaussian'

    scoop.logger.info("setting up scaler and reducing data")
    data, data_reduced, scaler = reduce_data(data, 16000)
    scoop.logger.info("finished setting up scaler and reducing data")

    BOUND_LOW_num = numpy.min(data, 0)
    BOUND_UP_num = numpy.max(data, 0)

    BOUND_LOW_trans = list(BOUND_LOW_num)
    BOUND_UP_trans = list(BOUND_UP_num)    

    BOUND_LOW_real = [0.0] * len(BOUND_LOW_trans)
    BOUND_UP_real = [1.0] * len(BOUND_UP_trans)

    result = scipy.optimize.differential_evolution(bandwidth_score, bounds = [(-3, 1),], 
                                               args = (data_reduced, kernel, atol), updating='deferred', workers=futures.map, disp=True,
                                               popsize=100)
    bw = 10**result.x[0]
    
    scoop.logger.info("mle bandwidth: %.2g", bw)

    kde_ga = KernelDensity(kernel=kernel, bandwidth=bw, atol=atol)
    kde_ga = kde_ga.fit(data)
        
    result_kde = scipy.optimize.differential_evolution(goal_kde, bounds = list(zip(BOUND_LOW_trans, BOUND_UP_trans)), 
                                               args = (kde_ga,), updating='deferred', workers=futures.map, disp=True,
                                               popsize=100)

    x = list(scaler.inverse_transform(numpy.array(result_kde.x).reshape(1, -1))[0])

    return x, kde_ga, scaler

