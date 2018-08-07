import util

name = "null"
count = 1

def getUnit(location):
    return location.split('/')[3]

def transform(parameter):
    def trans(i):
        return i

    return [trans,]

def untransform(seq, cache, parameter, fullPrecision=False):
    values = [seq[0],]

    if cache.roundParameters is not None and not fullPrecision:
        values = [util.RoundToSigFigs(i, cache.roundParameters) for i in values]

    headerValues = values
    return values, headerValues

def setSimulation(sim, parameter, seq, cache, fullPrecision=False):
    values, headerValues = untransform(seq, cache, parameter, fullPrecision)

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
    minValue = parameter['min']
    maxValue = parameter['max']

    return [minValue,], [maxValue,]

def getHeaders(parameter):
    location = parameter['location']

    try:
        comp = parameter['component']
    except KeyError:
        comp = 'None'

    name = location.rsplit('/', 1)[-1]
    bound = parameter.get('bound', None)
    index = parameter.get('index', None)
    
    headers = []
    if bound is not None:
        headers.append("%s Comp:%s Bound:%s" % (name, comp, bound))
    if index is not None:
        headers.append("%s Comp:%s Index:%s" % (name, comp, index))
    return headers

def getHeadersActual(parameter):
    return getHeaders(parameter)