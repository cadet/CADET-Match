import shutil
import h5py
import util
from pathlib import Path


def setupTemplates(settings, target):
    parms = []
    for parameter in settings['parameters']:
        comp = parameter['component']
        for location in parameter['location']:
            parts = location.split('/')
            name = parts[-1]
            unit = int(parts[3].replace('unit_', ''))
            for bound in parameter['bound']:
                entry = (name, unit, comp, bound)
                if entry not in parms:
                    parms.append(entry)

    target['sensitivities'] = parms

    for experiment in settings['experiments']:
        HDF5 = experiment['HDF5']
        name = experiment['name']

        template_path = Path(settings['resultsDir'], "template_%s.h5" % name)
        template_path_sens = Path(settings['resultsDir'], "template_%s_sens.h5" % name)

        shutil.copy(bytes(template_path),  bytes(template_path_sens))

        with h5py.File(template_path_sens, 'a') as h5:
            h5['/input/sensitivity/NSENS'][:] = len(parms)

            sensitivity = h5['/input/sensitivity']

            for idx, parm in enumerate(parms):
                name, unit, comp, bound = parm
                sens = sensitivity.create_group('param_%03d' % idx)
        
                util.set_value(sens, 'SENS_UNIT', 'i4', [unit,])
    
                util.set_value_enum(sens, 'SENS_NAME', [name,])

                util.set_value(sens, 'SENS_COMP', 'i4', [comp,])
                util.set_value(sens, 'SENS_REACTION', 'i4', [-1,])
                util.set_value(sens, 'SENS_BOUNDPHASE', 'i4', [bound,])
                util.set_value(sens, 'SENS_SECTION', 'i4', [-1,])

                util.set_value(sens, 'SENS_ABSTOL', 'f8', [1e-6,])
                util.set_value(sens, 'SENS_FACTOR', 'f8', [1.0,])