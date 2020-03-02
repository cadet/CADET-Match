import CADETMatch.util as util

name = "norm_add"
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

grad_transform = transform

def untransform(seq, cache, parameter):
    minValue = parameter['min']
    maxValue = parameter['max']

    values = [(maxValue - minValue) * seq[0] + minValue,]
    headerValues = values
    return values, headerValues

def grad_untransform(seq, cache, parameter):
    return untransform(seq, cache,parameter)[0]

def untransform_matrix(matrix, cache, parameter):
    minValue = parameter['min']
    maxValue = parameter['max']

    values = (maxValue - minValue) * matrix + minValue
    return values

untransform_matrix_inputorder = untransform_matrix

def getValue(sim, location, bound=None, comp=None, index=None):
    if bound is not None:
        unit = getUnit(location)
        boundOffset = util.getBoundOffset(sim.root.input.model[unit])

        if comp == -1:
            position = ()
            return sim[location.lower()]
        else:
            position = boundOffset[comp] + bound
            return sim[location.lower()][position]

    if index is not None:
        if index == -1:
            return sim[location.lower()]
        else:
            return sim[location.lower()][index]

def setValue(sim, value, location, bound=None, comp=None, index=None):
    if bound is not None:
        unit = getUnit(location)
        boundOffset = util.getBoundOffset(sim.root.input.model[unit])

        if comp == -1:
            position = ()
            sim[location.lower()] = value
        else:
            position = boundOffset[comp] + bound
            sim[location.lower()][position] = value

    if index is not None:
        if index == -1:
            sim[location.lower()] = value
        else:
            sim[location.lower()][index] = value

def setSimulation(sim, parameter, seq, cache, experiment):
    values, headerValues = untransform(seq, cache, parameter)

    if parameter.get('experiments', None) is None or experiment['name'] in parameter['experiments']:    
        locationFrom = parameter['locationFrom']
        locationTo = parameter['locationTo']
    
        try:
            compFrom = parameter['componentFrom']
            boundFrom = parameter['boundFrom']
            indexFrom = None
        except KeyError:
            indexFrom = parameter['indexFrom']
            boundFrom = None
            compFrom = None
        valueFrom = getValue(sim, locationFrom, bound=boundFrom, comp=compFrom, index=indexFrom)

        try:
            compTo = parameter['componentTo']
            boundTo = parameter['boundTo']
            indexTo = None
        except KeyError:
            indexTo = parameter['indexTo']
            boundTo = None
            compTo = None
        setValue(sim, valueFrom + values[0], locationTo, bound=boundTo, comp=compTo, index=indexTo)

    return values, headerValues

def setupTarget(parameter):
    location = parameter['locationTo']
    bound = parameter['boundTo']
    comp = parameter['componentTo']

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
    location = parameter['locationTo']

    try:
        comp = parameter['componentTo']
    except KeyError:
        comp = 'None'

    name = location.rsplit('/', 1)[-1]
    bound = parameter.get('boundTo', None)
    index = parameter.get('indexTo', None)
    
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
