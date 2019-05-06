import util
import numpy

name = "norm_keq"
count = 2
count_extended = 3

def getUnit(location):
    return location[0].split('/')[3]

def transform(parameter):
    minKA = numpy.log(parameter['minKA'])
    maxKA = numpy.log(parameter['maxKA'])
    minKEQ = numpy.log(parameter['minKEQ'])
    maxKEQ = numpy.log(parameter['maxKEQ'])

    def trans_a(i):
        return (numpy.log(i) - minKA)/(maxKA-minKA)

    def trans_b(i):
        return (numpy.log(i) - minKEQ)/(maxKEQ-minKEQ)

    return [trans_a, trans_b]

def untransform(seq, cache, parameter, fullPrecision=False):
    minKA = parameter['minKA']
    maxKA = parameter['maxKA']
    minKEQ = parameter['minKEQ']
    maxKEQ = parameter['maxKEQ']

    minValues = numpy.log([minKA, minKEQ])
    maxValues = numpy.log([maxKA, maxKEQ])

    values = numpy.array(seq)

    values = (maxValues - minValues) * values + minValues

    values = [numpy.exp(values[0]), numpy.exp(values[0])/(numpy.exp(values[1]))]
    headerValues = [values[0], values[1], values[0]/values[1]]
    return values, headerValues

def untransform_matrix(matrix, cache, parameter):
    minKA = parameter['minKA']
    maxKA = parameter['maxKA']
    minKEQ = parameter['minKEQ']
    maxKEQ = parameter['maxKEQ']

    minValues = numpy.log([minKA, minKEQ])
    maxValues = numpy.log([maxKA, maxKEQ])
    
    values = (maxValues - minValues) * values + minValues

    values = numpy.exp(values)

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
    return [0.0, 0.0], [1.0, 1.0]

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