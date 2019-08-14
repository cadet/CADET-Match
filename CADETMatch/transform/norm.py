import util

name = "norm"
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

def untransform(seq, cache, parameter):
    minValue = parameter['min']
    maxValue = parameter['max']

    values = [(maxValue - minValue) * seq[0] + minValue,]
    headerValues = values
    return values, headerValues

def untransform_matrix(matrix, cache, parameter):
    minValue = parameter['min']
    maxValue = parameter['max']

    values = (maxValue - minValue) * matrix + minValue
    return values

def setSimulation(sim, parameter, seq, cache, experiment):
    values, headerValues = untransform(seq, cache, parameter)

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

def getGradBounds(parameter):
    return [parameter['min'],], [parameter['max'],]

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

def setBounds(parameter, lb, ub):
    parameter['min'] = lb[0]
    parameter['max'] = ub[0]