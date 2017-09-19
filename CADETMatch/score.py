import scipy.stats
import numpy
import scipy.optimize
import scipy.interpolate
import scipy.signal
import numpy.linalg
import util

def logistic(x, a, b):
    return  1.0-1.0/(1.0+numpy.exp(a*(x-b)))

def exponential(x, a, b):
    return a * scipy.exp(b*x)

def linear(x, a, b):
    return a*x+b

def time_function_decay(CV_time, peak_time, diff_input=False):
    x_exp = numpy.array([0, 10.0*CV_time])
    y_exp = numpy.array([1, 0.5])

    args_exp = scipy.optimize.curve_fit(exponential, x_exp, y_exp, [1, -0.1])[0]

    def wrapper(x):

        if diff_input:
            diff = x
        else:
            diff = numpy.abs(x - peak_time)

        value = exponential(diff, *args_exp)


        #value = float(fun(diff))

        #clip values
        #value = min(value, 1.0)
        #value = max(value, 0.0)

        return value

    return wrapper


def time_function(CV_time, peak_time, diff_input=False):
    #x = numpy.array([0, CV_time/2, 2*CV_time, 5*CV_time, 8*CV_time, 12*CV_time])
    #y = numpy.array([1.0, 0.97, 0.5, 0.15, 0.01, 0])
    #fun = scipy.interpolate.UnivariateSpline(x,y, s=1e-6, ext=1)

    #args = scipy.optimize.curve_fit(logistic, x, y, [-0.1,2.0*CV_time])[0]
    x_exp = numpy.array([CV_time/2.0, 10.0*CV_time])
    y_exp = numpy.array([0.97, 0.5])

    x_lin = numpy.array([0, CV_time/2.0])
    y_lin = numpy.array([1, 0.97])

    args_exp = scipy.optimize.curve_fit(exponential, x_exp, y_exp, [1, -0.1])[0]
    args_lin = scipy.optimize.curve_fit(linear, x_lin, y_lin, [1, -0.1])[0]
    
    #scale = 1.0/logistic(0.0, *args)

    def wrapper(x):

        if diff_input:
            diff = x
        else:
            diff = numpy.abs(x - peak_time)

        if diff < CV_time/2.0:
            value = linear(diff, *args_lin)
        else:
            value = exponential(diff, *args_exp)


        #value = float(fun(diff))

        #clip values
        #value = min(value, 1.0)
        #value = max(value, 0.0)

        return value

    return wrapper

def value_function(peak_height, tolerance=1e-8, bottom_score = 0.01):
    #if the peak height is 0 or less than the tolerance it needs to be treated as a special case to prevent divide by zero problems
    x = numpy.array([0.0, 1.0])
    y = numpy.array([1.0, bottom_score])
    
    args = scipy.optimize.curve_fit(exponential, x, y, [1, -0.1])[0]

    scale = 1.0/exponential(0.0, *args)
    
    if numpy.abs(peak_height) < tolerance:
        def wrapper(x):
            if numpy.abs(x) < tolerance:
                return 1.0
            else:
                diff = numpy.abs(x-tolerance)/numpy.abs(tolerance)
                return exponential(diff, *args) * scale
    else:
        def wrapper(x):
            diff = numpy.abs(x-peak_height)/numpy.abs(peak_height)
            return exponential(diff, *args) * scale

    return wrapper

def pear_corr(cr):
    if cr < 0.5:
        out = 1.0/3.0 * cr + 1.0/3.0
    else:
        out = cr
    return out

def scoreBreakthrough(sim_data, experimental_data, feature):
    "similarity, value, start stop"
    sim_time_values, sim_data_values = util.get_times_values(sim_data['simulation'], feature)

    selected = feature['selected']

    exp_data_values = experimental_data['value'][selected]
    exp_time_values = experimental_data['time'][selected]

    [start, stop] = util.find_breakthrough(exp_time_values, sim_data_values)

    #sim_spline = scipy.interpolate.UnivariateSpline(exp_time_values, sim_data_values, s=util.smoothing_factor(sim_data_values))
    #exp_spline = scipy.interpolate.UnivariateSpline(exp_time_values, exp_data_values, s=util.smoothing_factor(exp_data_values))

    temp = [pear_corr(scipy.stats.pearsonr(sim_data_values, exp_data_values)[0]), 
            feature['value_function'](start[1]), 
            feature['time_function_start'](start[0]),
            feature['time_function_stop'](stop[0])]
    return temp, util.sse(sim_data_values, exp_data_values)

def scoreBreakthroughCross(sim_data, experimental_data, feature):
    "similarity, value, start stop"
    sim_time_values, sim_data_values = util.get_times_values(sim_data['simulation'], feature)

    selected = feature['selected']

    exp_data_values = experimental_data['value'][selected]
    exp_time_values = experimental_data['time'][selected]

    [start, stop] = util.find_breakthrough(exp_time_values, sim_data_values)

    corr = scipy.signal.correlate(sim_data_values, exp_data_values)/(numpy.linalg.norm(sim_data_values) * numpy.linalg.norm(exp_data_values))

    index = numpy.argmax(corr)

    score = corr[index]

    endTime = exp_time_values[-1]

    if index > len(exp_time_values):
        simTime = exp_time_values[-(index - len(exp_time_values))]
    elif index < len(exp_time_values):
        simTime = exp_time_values[-(len(exp_time_values) - index)]
    else:
        simTime = endTime

    diff_time = endTime - simTime

    temp = [score, 
            feature['value_function'](start[1]), 
            feature['time_function'](diff_time)]
    return temp, util.sse(sim_data_values, exp_data_values)

def scoreSimilarity(sim_data, experimental_data, feature):
    "Order is Pearson, Value, Time"
    sim_time_values, sim_data_values = util.get_times_values(sim_data['simulation'], feature)
    selected = feature['selected']

    exp_data_values = experimental_data['value'][selected]
    exp_time_values = experimental_data['time'][selected]

    [high, low] = util.find_peak(exp_time_values, sim_data_values)

    time_high, value_high = high
    
    sim_spline = scipy.interpolate.UnivariateSpline(exp_time_values, sim_data_values, s=util.smoothing_factor(sim_data_values))
    exp_spline = scipy.interpolate.UnivariateSpline(exp_time_values, exp_data_values, s=util.smoothing_factor(exp_data_values))

    temp = [pear_corr(scipy.stats.pearsonr(sim_spline(exp_time_values), exp_spline(exp_time_values))[0]), feature['value_function'](value_high), feature['time_function'](time_high)]
    return temp, util.sse(sim_data_values, exp_data_values)

def scoreSimilarityCrossCorrelate(sim_data, experimental_data, feature):
    "Order is Pearson, Value, Time"
    sim_time_values, sim_data_values = util.get_times_values(sim_data['simulation'], feature)
    selected = feature['selected']

    exp_data_values = experimental_data['value'][selected]
    exp_time_values = experimental_data['time'][selected]

    corr = scipy.signal.correlate(sim_data_values, exp_data_values)/(numpy.linalg.norm(sim_data_values) * numpy.linalg.norm(exp_data_values))

    index = numpy.argmax(corr)

    score = corr[index]

    endTime = exp_time_values[-1]

    if index > len(exp_time_values):
        simTime = exp_time_values[-(index - len(exp_time_values))]
    elif index < len(exp_time_values):
        simTime = exp_time_values[-(len(exp_time_values) - index)]
    else:
        simTime = endTime

    diff_time = endTime - simTime


    [high, low] = util.find_peak(exp_time_values, sim_data_values)

    time_high, value_high = high

    temp = [score, feature['value_function'](value_high), feature['time_function'](diff_time)]
    return temp, util.sse(sim_data_values, exp_data_values)

def scoreDerivativeSimilarity(sim_data, experimental_data, feature):
    "Order is Pearson, Value High, Time High, Value Low, Time Low"
    sim_time_values, sim_data_values = util.get_times_values(sim_data['simulation'], feature)
    selected = feature['selected']

    exp_data_values = experimental_data['value'][selected]
    exp_time_values = experimental_data['time'][selected]

    #spline_exp = scipy.interpolate.splrep(exp_time_values, util.smoothing(exp_time_values, exp_data_values))
    #spline_sim = scipy.interpolate.splrep(exp_time_values, util.smoothing(exp_time_values, sim_data_values))

    sim_spline = scipy.interpolate.UnivariateSpline(exp_time_values, util.smoothing(exp_time_values, sim_data_values), s=util.smoothing_factor(sim_data_values)).derivative(1)
    exp_spline = scipy.interpolate.UnivariateSpline(exp_time_values, util.smoothing(exp_time_values, exp_data_values), s=util.smoothing_factor(exp_data_values)).derivative(1)

    print(type(sim_spline), type(exp_spline))

    #spline_derivative_exp = scipy.interpolate.splev(exp_time_values, spline_exp, der=1)
    #spline_derivative_sim = scipy.interpolate.splev(exp_time_values, spline_sim, der=1)

    [highs, lows] = util.find_peak(exp_time_values, sim_spline(exp_time_values))

    return [pear_corr(scipy.stats.pearsonr(sim_spline(exp_time_values), exp_spline(exp_time_values))[0]), 
            feature['value_function_high'](highs[1]), 
            feature['time_function_high'](highs[0]),
            feature['value_function_low'](lows[1]), 
            feature['time_function_low'](lows[0]),], util.sse(sim_data_values, exp_data_values)

def scoreDerivativeSimilarityCross(sim_data, experimental_data, feature):
    "Order is Pearson, Value High, Time High, Value Low, Time Low"
    sim_time_values, sim_data_values = util.get_times_values(sim_data['simulation'], feature)
    selected = feature['selected']

    exp_data_values = experimental_data['value'][selected]
    exp_time_values = experimental_data['time'][selected]

    sim_spline = scipy.interpolate.UnivariateSpline(exp_time_values, util.smoothing(exp_time_values, sim_data_values)).derivative(1)
    exp_spline = scipy.interpolate.UnivariateSpline(exp_time_values, util.smoothing(exp_time_values, exp_data_values)).derivative(1)

    print(type(sim_spline), type(exp_spline))

    exp_data_values = exp_spline(exp_time_values)
    sim_data_values = sim_spline(exp_time_values)

    corr = scipy.signal.correlate(sim_data_values, exp_data_values)/(numpy.linalg.norm(sim_data_values) * numpy.linalg.norm(exp_data_values))

    index = numpy.argmax(corr)

    score = corr[index]

    endTime = exp_time_values[-1]

    if index > len(exp_time_values):
        simTime = exp_time_values[-(index - len(exp_time_values))]
    elif index < len(exp_time_values):
        simTime = exp_time_values[-(len(exp_time_values) - index)]
    else:
        simTime = endTime

    diff_time = endTime - simTime
    
    [highs, lows] = util.find_peak(exp_time_values, sim_data_values)

    return [pear_corr(scipy.stats.pearsonr(sim_data_values, exp_data_values)[0]), 
            feature['time_function'](diff_time),
            feature['value_function_high'](highs[1]),             
            feature['value_function_low'](lows[1]),], util.sse(sim_data_values, exp_data_values)

def scoreDerivativeSimilarityCrossAlt(sim_data, experimental_data, feature):
    "Order is Pearson, Value High, Time High, Value Low, Time Low"
    sim_time_values, sim_data_values = util.get_times_values(sim_data['simulation'], feature)
    selected = feature['selected']

    exp_data_values = experimental_data['value'][selected]
    exp_time_values = experimental_data['time'][selected]

    sim_spline = scipy.interpolate.UnivariateSpline(exp_time_values, util.smoothing(exp_time_values, sim_data_values)).derivative(1)
    exp_spline = scipy.interpolate.UnivariateSpline(exp_time_values, util.smoothing(exp_time_values, exp_data_values)).derivative(1)

    print(type(sim_spline), type(exp_spline))

    exp_data_values = exp_spline(exp_time_values)
    sim_data_values = sim_spline(exp_time_values)

    corr = scipy.signal.correlate(sim_data_values, exp_data_values)/(numpy.linalg.norm(sim_data_values) * numpy.linalg.norm(exp_data_values))

    index = numpy.argmax(corr)

    score = corr[index]

    endTime = exp_time_values[-1]

    if index > len(exp_time_values):
        simTime = exp_time_values[-(index - len(exp_time_values))]
    elif index < len(exp_time_values):
        simTime = exp_time_values[-(len(exp_time_values) - index)]
    else:
        simTime = endTime

    diff_time = endTime - simTime
    
    return [pear_corr(scipy.stats.pearsonr(sim_data_values, exp_data_values)[0]), 
            feature['time_function'](diff_time),], util.sse(sim_data_values, exp_data_values)

def scoreCurve(sim_data, experimental_data, feature):
    "Just Pearson score"
    sim_time_values, sim_data_values = util.get_times_values(sim_data['simulation'], feature)
    selected = feature['selected']

    exp_data_values = experimental_data['value'][selected]

    return [pear_corr(scipy.stats.pearsonr(sim_data_values, exp_data_values)[0])], util.sse(sim_data_values, exp_data_values)

def scoreDextran(sim_data, experimental_data, feature):
    "special score designed for dextran. This looks at only the front side of the peak up to the maximum slope and pins a value at the elbow in addition to the top"
    #print("feature", feature)
    sim_time_values, sim_data_values = util.get_times_values(sim_data['simulation'], feature)
    selected = feature['origSelected']
    max_time = feature['max_time']

    exp_time_values = experimental_data['time'][selected]
    exp_data_values = experimental_data['value'][selected]

    sim_data_values[sim_data_values < max(sim_data_values)/100.0] = 0

    sim_spline = scipy.interpolate.UnivariateSpline(exp_time_values, sim_data_values, s=util.smoothing_factor(sim_data_values))
    exp_spline = scipy.interpolate.UnivariateSpline(exp_time_values, exp_data_values, s=util.smoothing_factor(exp_data_values))

    sim_spline_derivative = sim_spline.derivative(1)
    exp_spline_derivative = exp_spline.derivative(1)

    expSelected = selected & (experimental_data['time'] <= max_time)
    expTime = experimental_data['time'][expSelected]
    expValues = experimental_data['value'][expSelected]
    expDerivValues = exp_spline_derivative(expSelected)

    values = sim_spline_derivative(exp_time_values)
    sim_max_index = numpy.argmax(values)
    sim_max_time = exp_time_values[sim_max_index]

    simSelected = selected & (experimental_data['time'] <= sim_max_time)
    simTime =  sim_data['time'][simSelected]
    simValues = sim_data['value'][simSelected]
    simDerivValues = sim_spline_derivative(simSelected)

    #this works the same way as matlab  xcorr(simValues, expValues, 'coeff')

    simValues[simValues < max(simValues)/100.0] = 0
    expValues[expValues < max(expValues)/100.0] = 0

    corr = scipy.signal.correlate(simValues, expValues)/(numpy.linalg.norm(simValues) * numpy.linalg.norm(expValues))
    corrDeriv = scipy.signal.correlate(simDerivValues, expDerivValues)/(numpy.linalg.norm(simDerivValues) * numpy.linalg.norm(expDerivValues))

    index = numpy.argmax(corr)

    score = corr[index]
    time = exp_time_values[index]

    indexDeriv = numpy.argmax(corrDeriv)
    scoreDeriv = corrDeriv[indexDeriv]

    if numpy.isnan(score):
        score = 0
        time = 0

    if numpy.isnan(scoreDeriv):
        scoreDeriv = 0

    if score < 0:
        score = 0

    if scoreDeriv < 0:
        scoreDeriv = 0

    return [score, scoreDeriv, feature['maxTimeFunction'](time)], util.sse(sim_data_values, exp_data_values)


def scoreFractionation(sim_data, experimental_data, feature):
    "Just Pearson score"
    simulation = sim_data['simulation']
    funcs = feature['funcs']

    times = simulation.root.output.solution.solution_times
    flow = simulation.root.input.model.connections.switch_000.connections[9]

    scores = []
    sim_values = []
    exp_values = []
   
    graph_sim = {}
    graph_exp = {}
    for (start, stop, component, exp_value, func) in funcs:
        selected = (times >= start) & (times <= stop)

        local_times = times[selected]
        local_values = simulation.root.output.solution.unit_001["solution_outlet_comp_%03d" % component][selected]

        sim_value = numpy.trapz(local_values, local_times) * flow * (stop - start)

        exp_values.append(exp_value)
        sim_values.append(sim_value)
        scores.append(func(sim_value))

        if component not in graph_sim:
            graph_sim[component] = []
            graph_exp[component] = []

        time_center = (start + stop)/2.0
        graph_sim[component].append( (time_center, sim_value) )
        graph_exp[component].append( (time_center, exp_value) )


    #sort lists
    for key,value in graph_sim.items():
        value.sort()
    for key,value in graph_exp.items():
        value.sort()

    sim_data['graph_exp'] = graph_exp
    sim_data['graph_sim'] = graph_sim
    return scores, util.sse(numpy.array(sim_values), numpy.array(exp_values))