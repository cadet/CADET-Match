import util
import numpy

name = "keq"
count = 2
count_extended = 3

def getUnit(location):
    return location[0].split('/')[3]

def transform(parameter):
    def trans_a(i):
        return numpy.log(i)

    def trans_b(i):
        return numpy.log(i)

    return [trans_a, trans_b]

def untransform(seq, cache, parameter, fullPrecision=False):
    values = [numpy.exp(seq[0]), numpy.exp(seq[0])/(numpy.exp(seq[1]))]

    if cache.roundParameters is not None and not fullPrecision:
        values = [util.RoundToSigFigs(i, cache.roundParameters) for i in values]

    headerValues = [values[0], values[1], values[0]/values[1]]
    return values, headerValues

def untransform_matrix(matrix, cache, parameter):
    values = numpy.exp(matrix)
    values[:,1] = values[:,0] / values[:,1]
 
    return values

def setSimulation(sim, parameter, seq, cache, experiment, fullPrecision=False):
    values, headerValues = untransform(seq, cache, parameter, fullPrecision)

    if parameter.get('experiments', None) is None or experiment['name'] in parameter['experiments']:
        location = parameter['location']
    
        comp = parameter['component']
        bound = parameter['bound']
    
        unit = getUnit(location)
        boundOffset = util.getBoundOffset(sim.root.input.model[unit])

        position = boundOffset[comp] + bound
        sim[location[0].lower()][position] = values[0]
        sim[location[1].lower()][position] = values[1]

    return values, headerValues

def setupTarget(parameter):
    location = parameter['location']
    bound = parameter['bound']
    comp = parameter['component']

    sensitivityOk = 1
    nameKA = location[0].rsplit('/', 1)[-1]
    nameKD = location[1].rsplit('/', 1)[-1]
    unit = int(location[0].split('/')[3].replace('unit_', ''))

    return [(nameKA, unit, comp, bound), (nameKD, unit, comp, bound)], sensitivityOk

def getBounds(parameter):
    minKA = parameter['minKA']
    maxKA = parameter['maxKA']
    minKEQ = parameter['minKEQ']
    maxKEQ = parameter['maxKEQ']

    minValues = numpy.log([minKA, minKEQ])
    maxValues = numpy.log([maxKA, maxKEQ])

    return minValues, maxValues

def getHeaders(parameter):
    location = parameter['location']
    nameKA = location[0].rsplit('/', 1)[-1]
    nameKD = location[1].rsplit('/', 1)[-1]
    bound = parameter['bound']
    comp = parameter['component']
    
    headers = []
    headers.append("%s Comp:%s Bound:%s" % (nameKA, comp, bound))
    headers.append("%s Comp:%s Bound:%s" % (nameKD, comp, bound))
    headers.append("%s/%s Comp:%s Bound:%s" % (nameKA, nameKD, comp, bound))
    return headers

def getHeadersActual(parameter):
    location = parameter['location']
    nameKA = location[0].rsplit('/', 1)[-1]
    nameKD = location[1].rsplit('/', 1)[-1]
    bound = parameter['bound']
    comp = parameter['component']
    
    headers = []
    headers.append("%s Comp:%s Bound:%s" % (nameKA, comp, bound))
    headers.append("%s/%s Comp:%s Bound:%s" % (nameKA, nameKD, comp, bound))
    return headers

def setBounds(parameter, lb, ub):
    parameter['minKA'] = lb[0]
    parameter['maxKA'] = ub[0]
    parameter['minKEQ'] = lb[2]
    parameter['maxKEQ'] = ub[2]

