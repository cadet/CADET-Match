import util
import math

name = "norm_diameter"
count = 1
count_extended = 1

def getUnit(location):
    return location.split('/')[3]

def transform(parameter):
    minValue = parameter['min']
    maxValue = parameter['max']

    def trans(i):
        return (i - minValue)/(maxValue-minValue)

    return [trans,]

def untransform(seq, cache, parameter, fullPrecision=False):
    minValue = parameter['min']
    maxValue = parameter['max']

    values_transform = [(maxValue - minValue) * seq[0] + minValue,]

    values = [math.pi * values_transform[0]**2/4.0,]

    if cache.roundParameters is not None and not fullPrecision:
        values = [util.RoundToSigFigs(i, cache.roundParameters) for i in values]

    headerValues = [values[0], values_transform[0]]
    return values, headerValues

def untransform_matrix(matrix, cache, parameter):
    minValue = parameter['min']
    maxValue = parameter['max']

    values = (maxValue - minValue) * matrix + minValue
    values[:,0] = math.pi * matrix[:,0]**2/4.0
    return values

def setSimulation(sim, parameter, seq, cache, experiment, fullPrecision=False):
    values, headerValues = untransform(seq, cache, parameter, fullPrecision)

    if parameter.get('experiments', None) is None or experiment['name'] in parameter['experiments']:
        location = parameter['location']
    
        try:
            comp = parameter['component']
            bound = parameter['bound']
            index = None
        except KeyError:
            index = parameter['index']
            bound = None

        if bound is not None:
            unit = getUnit(location)
            boundOffset = util.getBoundOffset(sim.root.input.model[unit])

            if comp == -1:
                position = ()
                sim[location.lower()] = values[0]
            else:
                position = boundOffset[comp] + bound
                sim[location.lower()][position] = values[0]

        if index is not None:
            sim[location.lower()][index] = values[0]
    return values, headerValues

def setupTarget(parameter):
    location = parameter['location']
    bound = parameter['bound']
    comp = parameter['component']

    name = location.rsplit('/', 1)[-1]
    sensitivityOk = 1

    try:
        unit = int(location.split('/')[3].replace('unit_', ''))
    except ValueError:
        unit = ''
        sensitivityOk = 0

    return [(name, unit, comp, bound),], sensitivityOk

def getBounds(parameter):
    return [0.0,], [1.0,]

def getHeaders(parameter):
    bound = parameter['bound']
    comp = parameter['component']
    
    headers = []
    headers.append("Area Comp:%s Bound:%s" % (comp, bound))
    headers.append("Diameter Comp:%s Bound:%s" % (comp, bound))
    return headers

def getHeadersActual(parameter):
    bound = parameter['bound']
    comp = parameter['component']
    
    headers = []
    headers.append("Area Comp:%s Bound:%s" % (comp, bound))
    return headers

def setBounds(parameter, lb, ub):
    parameter['min'] = lb[0]
    parameter['max'] = ub[0]

