import util

name = "volume_area"
count = 2
count_extended = 3

def getUnit(location):
    return location.split('/')[3]

def transform(parameter):
    def trans_volume(i):
        return i

    def trans_area(i):
        return i

    return [trans_volume, trans_area]

def untransform(seq, cache, parameter):
    values = [seq[1], seq[0]/seq[1]]
    headerValues = [values[0], values[1], values[0]*values[1]]
    return values, headerValues

def untransform_matrix(matrix, cache, parameter):
    values = numpy.zeros(matrix.shape)
    values[:,1] = matrix[:,0] / matrix[:,1]
    values[:,0] = matrix[:,1]
    return values

def setSimulation(sim, parameter, seq, cache, experiment):
    values, headerValues = untransform(seq, cache, parameter)
    
    if parameter.get('experiments', None) is None or experiment['name'] in parameter['experiments']:
        area_location = parameter['area_location']
        length_location = parameter['length_location']
    
        sim[area_location.lower()] = values[0]
        sim[length_location.lower()] = values[1]

    return values, headerValues

def setupTarget(parameter):
    area_location = parameter['area_location']
    length_location = parameter['length_location']
    bound = parameter['bound']
    comp = parameter['component']

    sensitivityOk = 1
    nameArea = area_location.rsplit('/', 1)[-1]
    nameLength = length_location.rsplit('/', 1)[-1]
    unit = int(area_location.split('/')[3].replace('unit_', ''))

    return [(nameArea, unit, comp, bound), (nameLength, unit, comp, bound)], sensitivityOk

def getBounds(parameter):
    minVolume = parameter['minVolume']
    maxVolume = parameter['maxVolume']
    minArea = parameter['minArea']
    maxArea = parameter['maxArea']

    minValues = [minVolume, minArea]
    maxValues = [maxVolume, maxArea]

    return minValues, maxValues

def getHeaders(parameter):
    bound = parameter['bound']
    comp = parameter['component']
    
    headers = []
    headers.append("Area Comp:%s Bound:%s" % (comp, bound))
    headers.append("Length Comp:%s Bound:%s" % (comp, bound))
    headers.append("Volume Comp:%s Bound:%s" % (comp, bound))
    return headers

def getHeadersActual(parameter):
    bound = parameter['bound']
    comp = parameter['component']
    
    headers = []
    headers.append("Volume Comp:%s Bound:%s" % (comp, bound))
    headers.append("Area Comp:%s Bound:%s" % (comp, bound))
    return headers

def setBounds(parameter, lb, ub):
    parameter['minVolume'] = lb[2]
    parameter['maxVolume'] = ub[2]
    parameter['minArea'] = lb[0]
    parameter['maxArea'] = ub[0]
