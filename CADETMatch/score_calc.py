import numpy

def sse(exp_data, sim_data):
    return numpy.sum( (numpy.array(exp_data) - numpy.array(sim_data))**2 )

def mse(exp_data, sim_data):
    return sse(exp_data, sim_data)/len(exp_data)

def rmse(exp_data, sim_data):
    return numpy.sqrt(mse(exp_data, sim_data))

def sse_norm(exp_data, sim_data):
    return numpy.sum( (numpy.array(exp_data) - numpy.array(sim_data))**2 )/(numpy.max(exp_data)**2)

def mse_norm(exp_data, sim_data):
    return sse_norm(exp_data, sim_data)/len(exp_data)

def norm_rmse(exp_data, sim_data):
    return numpy.sqrt(mse_norm(exp_data, sim_data))

def rmse_combine(exp_datas, sim_datas):
    len_entries = numpy.array([len(exp_data) for exp_data in exp_datas])
    total_entries = numpy.sum(len_entries)
    sse_s = numpy.array([sse_norm(exp_data, sim_data) for exp_data,sim_data in zip(exp_datas, sim_datas)])
    mse_s = sse_s/len_entries
    mean_total = numpy.sum(mse_s)    
    rmse = numpy.sqrt(mean_total)
    return rmse
