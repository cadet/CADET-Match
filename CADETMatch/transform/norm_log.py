import util
import numpy

name = "norm_log"
count = 1
count_extended = 1

def getUnit(location):
    return location.split('/')[3]

def transform(parameter):
    minValue = numpy.log(parameter['min'])
    maxValue = numpy.log(parameter['max'])

    def trans(i):
        return (numpy.log(i) - minValue)/(maxValue-minValue)

    return [trans,]

grad_transform = transform

def untransform(seq, cache, parameter):
    minValue = numpy.log(parameter['min'])
    maxValue = numpy.log(parameter['max'])

    values = [(maxValue - minValue) * seq[0] + minValue,]

    values = [numpy.exp(values[0]),]
    headerValues = values
    return values, headerValues

def grad_untransform(seq, cache, parameter):
    return untransform(seq, cache,parameter)[0]

def untransform_matrix(matrix, cache, parameter):
    minValue = numpy.log(parameter['min'])
    maxValue = numpy.log(parameter['max'])

    temp = (maxValue - minValue) * matrix + minValue

    values = numpy.exp(temp)

    return values

untransform_matrix_inputorder = untransform_matrix

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