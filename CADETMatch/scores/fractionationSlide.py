import CADETMatch.util as util
import CADETMatch.score as score
import numpy
import pandas
import scipy.optimize
from addict import Dict
import multiprocessing

name = "fractionationSlide"


def get_settings(feature):
    settings = Dict()
    settings.adaptive = True
    settings.badScore = 0
    settings.meta_mask = True

    data = pandas.read_csv(feature["fraction_csv"])
    headers = data.columns.values.tolist()
    comps = len(headers[2:])

    settings.count = 3 * comps

    return settings


def goal(offset, frac_exp, sim_data_time, spline, start, stop):
    sim_data_value = spline(sim_data_time - offset)
    frac_sim = util.fractionate(start, stop, sim_data_time, sim_data_value)
    return float(numpy.sum((frac_exp - frac_sim) ** 2))


def searchRange(times, start_frac, stop_frac, CV_time):
    collectionStart = min(start_frac)
    collectionStop = max(stop_frac)

    searchStart = collectionStart - CV_time
    searchStop = collectionStop + CV_time

    searchStart = max(searchStart, times[0])
    searchStop = min(searchStop, times[-1])

    searchIndexStart = numpy.argmax(times[times <= searchStart])
    searchIndexStop = numpy.argmax(times[times <= searchStop])
    return searchIndexStart, searchIndexStop


def run(sim_data, feature):
    simulation = sim_data["simulation"]
    timeFunc = feature["timeFunc"]
    components = feature["components"]
    numComponents = len(components)
    samplesPerComponent = feature["samplesPerComponent"]
    data = feature["data"]
    CV_time = feature["CV_time"]
    start = feature["start"]
    stop = feature["stop"]
    funcs = feature["funcs"]

    time_center = (start + stop) / 2.0

    times = simulation.root.output.solution.solution_times

    scores = []

    sim_values_sse = []
    exp_values_sse = []

    graph_sim = {}
    graph_exp = {}
    graph_sim_offset = {}

    searchIndexStart, searchIndexStop = searchRange(times, start, stop, CV_time)

    for component, value_func in funcs:
        exp_values = numpy.array(data[str(component)])
        selected = numpy.isfinite(exp_values)
        sim_value = simulation.root.output.solution[feature["unit"]]["solution_outlet_comp_%03d" % component]

        spline = scipy.interpolate.InterpolatedUnivariateSpline(times, sim_value, ext=1)

        # get a starting point estimate
        offsets = numpy.linspace(-times[-1], times[-1], 50)
        errors = [goal(offset, exp_values[selected], times, spline, start[selected], stop[selected]) for offset in offsets]
        offset_start = offsets[numpy.argmin(errors)]

        result_powell = scipy.optimize.minimize(
            goal, offset_start, args=(exp_values[selected], times, spline, start[selected], stop[selected]), method="powell"
        )

        time_offset = result_powell.x[0]
        sim_data_value = spline(times - time_offset)

        fracOffset = util.fractionate(start[selected], stop[selected], times, sim_data_value)

        # if the simulation scale and exp scale are too different the estimation of similarity, offset etc is not accurate discard if value max/min > 1e3
        max_exp = max(exp_values[selected])
        max_sim = max(fracOffset)
        if max(max_exp, max_sim) / min(max_exp, max_sim) > 1e3:
            value_score = 0
            pear = 0
            time_score = 0
        else:
            value_score = value_func(max(fracOffset))
            pear = score.pear_corr(scipy.stats.pearsonr(exp_values[selected], fracOffset)[0])
            time_score = timeFunc(abs(time_offset))

        exp_values_sse.extend(exp_values[selected])
        sim_values_sse.extend(fracOffset)

        scores.append(pear)
        scores.append(time_score)
        scores.append(value_score)

        graph_sim[component] = list(zip(start[selected], stop[selected], fracOffset))
        graph_exp[component] = list(zip(start[selected], stop[selected], exp_values[selected]))

        graph_sim_offset[component] = time_offset

    sim_data["graph_exp"] = graph_exp
    sim_data["graph_sim"] = graph_sim
    sim_data["graph_sim_offset"] = graph_sim_offset

    return (
        scores,
        util.sse(numpy.array(sim_values_sse), numpy.array(exp_values_sse)),
        len(sim_values_sse),
        time_center,
        numpy.array(sim_values_sse),
        numpy.array(exp_values_sse),
        [1.0 - i for i in scores],
    )


def setup(sim, feature, selectedTimes, selectedValues, CV_time, abstol, cache):
    temp = {}
    data = pandas.read_csv(feature["fraction_csv"])
    rows, cols = data.shape

    headers = data.columns.values.tolist()

    start = numpy.array(data.iloc[:, 0])
    stop = numpy.array(data.iloc[:, 1])

    smallestTime = min(start - stop)
    abstolFraction = abstol * smallestTime

    funcs = []

    for idx, component in enumerate(headers[2:], 2):
        value = numpy.array(data.iloc[:, idx])
        funcs.append((int(component), score.value_function(max(value), abstolFraction)))

    temp["data"] = data
    temp["start"] = start
    temp["stop"] = stop
    temp["timeFunc"] = score.time_function_decay(feature["time"][-1])
    temp["components"] = [int(i) for i in headers[2:]]
    temp["samplesPerComponent"] = rows
    temp["CV_time"] = CV_time
    temp["funcs"] = funcs
    temp["unit"] = feature["unit_name"]
    temp["peak_max"] = data.iloc[:, 2:].max().min()
    return temp


def headers(experimentName, feature):
    data = pandas.read_csv(feature["fraction_csv"])
    rows, cols = data.shape

    data_headers = data.columns.values.tolist()

    temp = []
    for component in data_headers[2:]:
        temp.append("%s_%s_Component_%s_Similarity" % (experimentName, feature["name"], component))
        temp.append("%s_%s_Component_%s_Time" % (experimentName, feature["name"], component))
        temp.append("%s_%s_Component_%s_Value" % (experimentName, feature["name"], component))
    return temp
