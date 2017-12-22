"Cache module. Used to store per worker information so it does not need to be recalculated"

from pathlib import Path
import json
from cadet import Cadet

import numpy
import scipy
import scipy.interpolate
import pandas
import util
import sys

import score
import plugins

class Cache:
    def __init__(self):
        self.scores = plugins.get_plugins('scores')
        self.search = plugins.get_plugins('search')
        self.settings = None
        self.headers = None
        self.numGoals = None
        self.target= None
        self.MIN_VALUE = None
        self.MAX_VALUE = None
        self.badScore = None
        self.WORST = None
        self.json_path = None
        self.adaptive = None
        self.transform = None
        self.parameter_indexes = None
        self.score_indexes = None
        

    def setup(self, json_path):
        "setup the cache based on the json file being used"
        self.json_path = json_path
        self.setupSettings()
        Cadet.cadet_path = self.settings['CADETPath']
        
        self.setupHeaders()
        self.setupTarget()
        self.setupMinMax()
        
        self.WORST = [self.badScore] * self.numGoals

        self.settings['transform'] = self.transform

        #create used paths in settings, only the root process will make the directories later
        self.settings['resultsDirEvo'] = Path(self.settings['resultsDir']) / "evo"
        self.settings['resultsDirGrad'] = Path(self.settings['resultsDir']) / "grad"
        self.settings['resultsDirMisc'] = Path(self.settings['resultsDir']) / "misc"
        self.settings['resultsDirSpace'] = Path(self.settings['resultsDir']) / "space"
        self.settings['resultsDirBase'] = Path(self.settings['resultsDir'])

    def setupSettings(self):
        settings_file = Path(self.json_path)
        with settings_file.open() as json_data:
            self.settings = json.load(json_data)

            self.settings['population'] = int(self.settings['population'])

            if "bootstrap" in self.settings:
                self.settings['bootstrap']['samples'] = int(self.settings['bootstrap']['samples'])

    def setupHeaders(self):
        self.headers = ['Time','Name', 'Method','Condition Number',]

        self.numGoals = 0
        self.badScore = 0.0
        base = len(self.headers)
        
        for parameter in self.settings['parameters']:
            try:
                comp = parameter['component']
            except KeyError:
                comp = 'None'
            if parameter['transform'] == 'keq':
                location = parameter['location']
                nameKA = location[0].rsplit('/',1)[-1]
                nameKD = location[1].rsplit('/',1)[-1]
                for bound in parameter['bound']:
                    self.headers.append("%s Comp:%s Bound:%s" % (nameKA, comp, bound))
                    self.headers.append("%s Comp:%s Bound:%s" % (nameKD, comp, bound))
                    self.headers.append("%s/%s Comp:%s Bound:%s" % (nameKA, nameKD, comp, bound))
            elif parameter['transform'] == 'log':
                location = parameter['location']
                name = location.rsplit('/',1)[-1]
                for bound in parameter.get('bound', []):
                    self.headers.append("%s Comp:%s Bound:%s" % (name, comp, bound))
                for idx in parameter.get('indexes', []):
                    self.headers.append("%s Comp:%s Index:%s" % (name, comp, idx))

        parameters = len(self.headers)
        self.parameter_indexes = list(range(base, parameters))

        for idx,experiment in enumerate(self.settings['experiments']):
            experimentName = experiment['name']
            experiment['headers'] = []
            for feature in experiment['features']:
                if feature['type'] in ('similarity', 'similarityCross', 'similarityHybrid', 'similarityDecay', 'similarityCrossDecay', 'similarityHybridDecay'):
                    name = "%s_%s" % (experimentName, feature['name'])
                    temp = ["%s_Similarity" % name, "%s_Value" % name, "%s_Time" % name]
                    self.numGoals += 3

                elif feature['type'] == 'derivative_similarity':
                    name = "%s_%s" % (experimentName, feature['name'])
                    temp = ["%s_Derivative_Similarity" % name, "%s_High_Value" % name, "%s_High_Time" % name, "%s_Low_Value" % name, "%s_Low_Time" % name]
                    self.numGoals += 5

                elif feature['type'] == 'derivative_similarity_hybrid':
                    name = "%s_%s" % (experimentName, feature['name'])
                    temp = ["%s_Derivative_Similarity_hybrid" % name, "%s_Time" % name, "%s_High_Value" % name, "%s_Low_Value" % name]
                    self.numGoals += 4

                elif feature['type'] == 'derivative_similarity_cross':
                    name = "%s_%s" % (experimentName, feature['name'])
                    temp = ["%s_Derivative_Similarity_Cross" % name, "%s_Time" % name, "%s_High_Value" % name, "%s_Low_Value" % name]
                    self.numGoals += 4

                elif feature['type'] == 'derivative_similarity_cross_alt':
                    name = "%s_%s" % (experimentName, feature['name'])
                    temp = ["%s_Derivative_Similarity_Cross_Alt" % name, "%s_Time" % name,]
                    self.numGoals += 2

                elif feature['type'] == 'curve':
                    name = "%s_%s" % (experimentName, feature['name'])
                    temp  = ["%s_Similarity" % name]
                    self.numGoals += 1

                elif feature['type'] == 'breakthrough':
                    name = "%s_%s" % (experimentName, feature['name'])
                    temp  = ["%s_Similarity" % name, "%s_Value" % name, "%s_Time_Start" % name, "%s_Time_Stop" % name]
                    self.numGoals += 4

                elif feature['type'] in ('breakthroughCross', 'breakthroughHybrid'):
                    name = "%s_%s" % (experimentName, feature['name'])
                    temp  = ["%s_Similarity" % name, "%s_Value" % name, "%s_Time" % name]
                    self.numGoals += 3

                elif feature['type'] in ('dextran', 'dextranHybrid'):
                    name = "%s_%s" % (experimentName, feature['name'])
                    temp = ["%s_Front_Similarity" % name, "%s_Derivative_Similarity" % name, "%s_Time" % name]
                    self.numGoals += 3

                elif feature['type'] == 'fractionation':
                    data = pandas.read_csv(feature['csv'])
                    rows, cols = data.shape
                    #remove first two columns since those are the start and stop times
                    cols = cols - 2

                    total = rows * cols
                    data_headers = data.columns.values.tolist()

                    temp  = []
                    for sample in range(rows):
                        for component in data_headers[2:]:
                            temp.append('%s_%s_Sample_%s_Component_%s' % (experimentName, feature['name'], sample, component))

                    self.numGoals += len(temp)

                elif feature['type'] == 'fractionationCombine':
                    data = pandas.read_csv(feature['csv'])
                    rows, cols = data.shape
                    #remove first two columns since those are the start and stop times
                    cols = cols - 2

                    total = rows * cols
                    data_headers = data.columns.values.tolist()

                    temp  = []
                    for component in data_headers[2:]:
                        temp.append('%s_%s_Component_%s' % (experimentName, feature['name'], component))
                    self.numGoals += len(temp)

                elif feature['type'] == 'fractionationMeanVariance':
                    data = pandas.read_csv(feature['csv'])
                    rows, cols = data.shape

                    data_headers = data.columns.values.tolist()

                    temp  = []
                    for component in data_headers[2:]:
                        temp.append('%s_%s_Component_%s_time_mean' % (experimentName, feature['name'], component))
                        temp.append('%s_%s_Component_%s_time_var' % (experimentName, feature['name'], component))
                        temp.append('%s_%s_Component_%s_value_mean' % (experimentName, feature['name'], component))
                        temp.append('%s_%s_Component_%s_value_var' % (experimentName, feature['name'], component))
                    self.numGoals += len(temp)

                elif feature['type'] == 'SSE':
                    name = "%s_%s" % (experimentName, feature['name'])
                    temp = ["%s_SSE" % name]
                    self.numGoals += 1
                    self.badScore = -sys.float_info.max

                elif feature['type'] == 'LogSSE':
                    name = "%s_%s" % (experimentName, feature['name'])
                    temp = ["%s_LogSSE" % name]
                    self.numGoals += 1
                    self.badScore = -sys.float_info.max

                self.headers.extend(temp)
                experiment['headers'].extend(temp)
                               
        self.headers.extend(['Product Root Score', 'Min Score', 'Mean Score', 'Norm', 'SSE'])

        scores = len(self.headers)
        self.score_indexes = list(range(parameters, scores))

    def setupTarget(self):
        self.target = {}
        self.adaptive = True

        for experiment in self.settings['experiments']:
            self.target[experiment["name"]] = self.setupExperiment(experiment)
        self.target['bestHumanScores'] = numpy.ones(5) * self.badScore

        #SSE are negative so they sort correctly with better scores being less negative
        self.target['bestHumanScores'][4] = self.badScore;  

        #setup sensitivities
        parms = []
        sensitivityOk = 1
        for parameter in self.settings['parameters']:
            try:
                comp = parameter['component']
            except KeyError:
                sensitivityOk = 0
                break

            transform = parameter['transform']

            if transform == 'keq':
                location = parameter['location']
                nameKA = location[0].rsplit('/',1)[-1]
                nameKD = location[1].rsplit('/',1)[-1]
                unit = int(location[0].split('/')[3].replace('unit_', ''))

                for bound in parameter['bound']:
                    parms.append((nameKA, unit, comp, bound))
                    parms.append((nameKD, unit, comp, bound))

            elif transform == 'log':
                location = parameter['location']
                name = location.rsplit('/',1)[-1]
                try:
                    unit = int(location.split('/')[3].replace('unit_', ''))
                except ValueError:
                    unit = ''
                    sensitivityOk = 0

                for bound in parameter['bound']:
                    parms.append((name, unit, comp, bound))

        if sensitivityOk:
            self.target['sensitivities'] = parms
        else:
            self.target['sensitivities'] = []

    def setupExperiment(self, experiment):
        temp = {}

        sim = Cadet()
        sim.filename = Path(experiment['HDF5'])
        sim.load()

        abstol = sim.root.input.solver.time_integrator.abstol

        #CV needs to be based on superficial velocity not interstitial velocity
        length = float(sim.root.input.model.unit_001.col_length)

        velocity = sim.root.input.model.unit_001.velocity
        if velocity == {}:
            velocity = 1.0
        velocity = float(velocity)

        area = sim.root.input.model.uni_001.cross_section_area
        if area == {}:
            area = 1.0
        area = float(area)

        porosity = sim.root.input.model.unit_001.col_porosity
        if porosity == {}:
            porosity = sim.root.input.model.unit_001.total_porosity
        if porosity == {}:
            porosity = 1.0
        porosity = float(porosity)

        conn = sim.root.input.model.connections.switch_000.connections

        conn = numpy.array(conn)
        conn = numpy.reshape(conn, [-1, 5])

        #find all the entries that connect to the column
        filter = conn[:,1] == 1

        #flow is the sum of all flow rates that connect to this column which is in the last column
        flow = sum(conn[filter, -1])

        if area == 1 and abs(velocity) != 1:
            CV_time = length / velocity
        else:
            CV_time = (area * length) / flow

        if 'CSV' in experiment:
            data = numpy.genfromtxt(experiment['CSV'], delimiter=',')

            temp['time'] = data[:,0]
            temp['value'] = data[:,1]

        for feature in experiment['features']:
            featureName = feature['name']
            featureType = feature['type']
            featureStart = float(feature['start'])
            featureStop = float(feature['stop'])

            temp[featureName] = {}

            if 'CSV' in feature:
                dataLocal = numpy.genfromtxt(feature['CSV'], delimiter=',')
                temp[featureName]['time'] = dataLocal[:,0]
                temp[featureName]['value'] = dataLocal[:,1]
            else:
                temp[featureName]['time'] = data[:,0]
                temp[featureName]['value'] = data[:,1]

            if 'isotherm' in feature:
                temp[featureName]['isotherm'] = feature['isotherm']
            else:
                temp[featureName]['isotherm'] = experiment['isotherm']

            temp[featureName]['selected'] = (temp[featureName]['time'] >= featureStart) & (temp[featureName]['time'] <= featureStop)
            
            selectedTimes = temp[featureName]['time'][temp[featureName]['selected']]
            selectedValues = temp[featureName]['value'][temp[featureName]['selected']]

            if featureType in ('similarity', 'similarityCross', 'similarityHybrid'):
                temp[featureName]['peak'] = util.find_peak(selectedTimes, selectedValues)[0]
                temp[featureName]['time_function'] = score.time_function(CV_time, temp[featureName]['peak'][0], diff_input = True if featureType in ('similarityCross', 'similarityHybrid') else False)
                temp[featureName]['value_function'] = score.value_function(temp[featureName]['peak'][1], abstol)

            if featureType in ('similarityDecay', 'similarityCrossDecay', 'similarityHybridDecay'):
                temp[featureName]['peak'] = util.find_peak(selectedTimes, selectedValues)[0]
                temp[featureName]['time_function'] = score.time_function_decay(CV_time, temp[featureName]['peak'][0], diff_input = True if featureType in ('similarityCrossDecay', 'similarityHybridDecay') else False)
                temp[featureName]['value_function'] = score.value_function(temp[featureName]['peak'][1], abstol)

            if featureType == 'breakthrough':
                temp[featureName]['break'] = util.find_breakthrough(selectedTimes, selectedValues)
                temp[featureName]['time_function_start'] = score.time_function(CV_time, temp[featureName]['break'][0][0])
                temp[featureName]['time_function_stop'] = score.time_function(CV_time, temp[featureName]['break'][1][0])
                temp[featureName]['value_function'] = score.value_function(temp[featureName]['break'][0][1], abstol)

            if featureType in ('breakthroughCross', 'breakthroughHybrid'):
                temp[featureName]['break'] = util.find_breakthrough(selectedTimes, selectedValues)
                temp[featureName]['time_function'] = score.time_function(CV_time, temp[featureName]['break'][0][0], diff_input=True)
                temp[featureName]['value_function'] = score.value_function(temp[featureName]['break'][0][1], abstol)

            if featureType == 'derivative_similarity':
                exp_spline = scipy.interpolate.UnivariateSpline(selectedTimes, util.smoothing(selectedTimes, selectedValues), s=util.smoothing_factor(selectedValues)).derivative(1)

                [high, low] = util.find_peak(selectedTimes, exp_spline(selectedTimes))

                temp[featureName]['peak_high'] = high
                temp[featureName]['peak_low'] = low

                temp[featureName]['time_function_high'] = score.time_function(CV_time, high[0])
                temp[featureName]['value_function_high'] = score.value_function(high[1], abstol, 0.1)
                temp[featureName]['time_function_low'] = score.time_function(CV_time, low[0])
                temp[featureName]['value_function_low'] = score.value_function(low[1], abstol, 0.1)

            if featureType in ('derivative_similarity_hybrid', 'derivative_similarity_cross'):
                exp_spline = scipy.interpolate.UnivariateSpline(selectedTimes, util.smoothing(selectedTimes, selectedValues), s=util.smoothing_factor(selectedValues)).derivative(1)

                [high, low] = util.find_peak(selectedTimes, exp_spline(selectedTimes))

                temp[featureName]['peak_high'] = high
                temp[featureName]['peak_low'] = low

                temp[featureName]['time_function'] = score.time_function(CV_time,high[0], diff_input = True)
                temp[featureName]['value_function_high'] = score.value_function(high[1], abstol, 0.1)
                temp[featureName]['value_function_low'] = score.value_function(low[1], abstol, 0.1)

            if featureType == 'derivative_similarity_cross_alt':
                exp_spline = scipy.interpolate.UnivariateSpline(selectedTimes, util.smoothing(selectedTimes, selectedValues), s=util.smoothing_factor(selectedValues)).derivative(1)

                [high, low] = util.find_peak(selectedTimes, exp_spline(selectedTimes))

                temp[featureName]['peak_high'] = high
                temp[featureName]['peak_low'] = low

                temp[featureName]['time_function'] = score.time_function(CV_time,high[0], diff_input = True)

            if featureType == "dextran":
                #change the stop point to be where the max positive slope is along the searched interval
                exp_spline = scipy.interpolate.UnivariateSpline(selectedTimes, selectedValues, s=util.smoothing_factor(selectedValues), k=1).derivative(1)
                values = exp_spline(selectedTimes)
                #print([i for i in zip(selectedTimes, values)])
                max_index = numpy.argmax(values)
                max_time = selectedTimes[max_index]
                #print(max_time, values[max_index])
            
                temp[featureName]['origSelected'] = temp[featureName]['selected']
                temp[featureName]['selected'] = temp[featureName]['selected'] & (temp[featureName]['time'] <= max_time)
                temp[featureName]['max_time'] = max_time
                temp[featureName]['maxTimeFunction'] = score.time_function_decay(CV_time/10.0, max_time, diff_input=True)

            if featureType == "dextranHybrid":
                #change the stop point to be where the max positive slope is along the searched interval
                exp_spline = scipy.interpolate.UnivariateSpline(selectedTimes, selectedValues, s=util.smoothing_factor(selectedValues), k=1).derivative(1)
                values = exp_spline(selectedTimes)
                max_index = numpy.argmax(values)
                max_time = selectedTimes[max_index]
            
                temp[featureName]['origSelected'] = temp[featureName]['selected']
                temp[featureName]['selected'] = temp[featureName]['selected'] & (temp[featureName]['time'] <= max_time)
                temp[featureName]['max_time'] = max_time
                temp[featureName]['offsetTimeFunction'] = score.time_function_decay(CV_time/10.0, max_time, diff_input=True)

            if featureType == 'fractionation':
                data = pandas.read_csv(feature['csv'])
                rows, cols = data.shape

                flow = sim.root.input.model.connections.switch_000.connections[9]
                smallestTime = min(data['Stop'] - data['Start'])
                abstolFraction = flow * abstol * smallestTime

                print('abstolFraction', abstolFraction)

                headers = data.columns.values.tolist()

                funcs = []

                for sample in range(rows):
                    for component in headers[2:]:
                        start = data['Start'][sample]
                        stop = data['Stop'][sample]
                        value = data[component][sample]
                        func = score.value_function(value, abstolFraction)

                        funcs.append( (start, stop, int(component), value, func) )
                temp[featureName]['funcs'] = funcs

            if featureType == 'fractionationCombine':
                data = pandas.read_csv(feature['csv'])
                rows, cols = data.shape

                headers = data.columns.values.tolist()

                flow = sim.root.input.model.connections.switch_000.connections[9]
                smallestTime = min(data['Stop'] - data['Start'])
                abstolFraction = flow * abstol * smallestTime

                print('abstolFraction', abstolFraction)

                funcs = []

                for sample in range(rows):
                    for component in headers[2:]:
                        start = data['Start'][sample]
                        stop = data['Stop'][sample]
                        value = data[component][sample]
                        func = score.value_function(value, abstolFraction)

                        funcs.append( (start, stop, int(component), value, func) )
                temp[featureName]['funcs'] = funcs
                temp[featureName]['components'] = [int(i) for i in headers[2:]]
                temp[featureName]['samplesPerComponent'] = rows

            if featureType == 'fractionationMeanVariance':
                data = pandas.read_csv(feature['csv'])
                rows, cols = data.shape

                headers = data.columns.values.tolist()

                start = numpy.array(data.iloc[:,0])
                stop = numpy.array(data.iloc[:,1])

                time_center = (start + stop)/2.0

                flow = sim.root.input.model.connections.switch_000.connections[9]
                smallestTime = min(data['Stop'] - data['Start'])
                abstolFraction = flow * abstol * smallestTime

                funcs = []

                for idx, component in enumerate(headers[2:], 2):
                    value = numpy.array(data.iloc[:,idx])

                    mean_time, variance_time, mean_value, variance_value = util.fracStat(time_center, value)

                    func_mean_time = score.time_function(CV_time, mean_time, diff_input = False)
                    func_variance_time = score.value_function(variance_time)

                    func_mean_value = score.value_function(mean_value, abstolFraction)
                    func_variance_value = score.value_function(variance_value, abstolFraction/1e5)

                    funcs.append( (start, stop, int(component), value, func_mean_time, func_variance_time, func_mean_value, func_variance_value) )

                temp[featureName]['funcs'] = funcs
                temp[featureName]['components'] = [int(i) for i in headers[2:]]
                temp[featureName]['samplesPerComponent'] = rows

            if featureType in ('SSE', 'LogSSE'):
                self.adaptive = False
            
        return temp

    def setupMinMax(self):
        "build the minimum and maximum parameter boundaries"
        self.MIN_VALUE = []
        self.MAX_VALUE = []
        self.transform = []

        for parameter in self.settings['parameters']:
            transform = parameter['transform']
            location = parameter['location']

            if transform == 'keq':
                minKA = parameter['minKA']
                maxKA = parameter['maxKA']
                minKEQ = parameter['minKEQ']
                maxKEQ = parameter['maxKEQ']

                minValues = [item for pair in zip(minKA, minKEQ) for item in pair]
                maxValues = [item for pair in zip(maxKA, maxKEQ) for item in pair]

                self.transform.extend([numpy.log for i in minValues])

                minValues = numpy.log(minValues)
                maxValues = numpy.log(maxValues)

            elif transform == 'log':
                minValues = numpy.log(parameter['min'])
                maxValues = numpy.log(parameter['max'])
                self.transform.append(numpy.log)

            self.MIN_VALUE.extend(minValues)
            self.MAX_VALUE.extend(maxValues)

cache = Cache()
