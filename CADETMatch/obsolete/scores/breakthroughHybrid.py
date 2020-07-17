import CADETMatch.util as util
import CADETMatch.score as score
import scipy.stats
from addict import Dict

name = "breakthroughHybrid"
settings = Dict()
settings.adaptive = True
settings.badScore = 0
settings.meta_mask = True
settings.count = 3


def run(sim_data, feature):
    "similarity, value, start stop"
    sim_time_values, sim_data_values = util.get_times_values(sim_data["simulation"], feature)

    selected = feature["selected"]

    exp_data_values = feature["value"][selected]
    exp_time_values = feature["time"][selected]

    [start, stop] = util.find_breakthrough(exp_time_values, sim_data_values)

    score_corr, diff_time = score.cross_correlate(exp_time_values, sim_data_values, exp_data_values)

    temp = [
        score.pear_corr(scipy.stats.pearsonr(sim_data_values, exp_data_values)[0]),
        feature["value_function"](start[1]),
        feature["time_function"](diff_time),
    ]
    return (
        temp,
        util.sse(sim_data_values, exp_data_values),
        len(sim_data_values),
        sim_time_values,
        sim_data_values,
        exp_data_values,
        [1.0 - i for i in temp],
    )


def setup(sim, feature, selectedTimes, selectedValues, CV_time, abstol):
    temp = {}
    temp["break"] = util.find_breakthrough(selectedTimes, selectedValues)
    temp["time_function"] = score.time_function(CV_time, temp["break"][0][0], diff_input=True)
    temp["value_function"] = score.value_function(temp["break"][0][1], abstol)
    return temp


def headers(experimentName, feature):
    name = "%s_%s" % (experimentName, feature["name"])
    temp = ["%s_Similarity" % name, "%s_Value" % name, "%s_Time" % name]
    return temp
