import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cadet import H5

from CADETMatch.cache import cache

from pathlib import Path
import numpy

def main():
    cache.setup(sys.argv[1])
    
    mcmcDir = Path(cache.settings['resultsDirMCMC'])

    h5_data = H5()
    h5_data.filename = (mcmcDir / "kde_settings.h5").as_posix()
    h5_data.load()

    store = h5_data.root.store

    mcmcDir = Path(cache.settings['resultsDirMCMC'])
    plt.figure(figsize=[10,10])
    plt.scatter(numpy.log10(store[:,0]), numpy.log10(store[:,1]))
    plt.xlabel('bandwidth')
    plt.ylabel('cross_val_score')
    plt.savefig(str(mcmcDir / "log_bandwidth.png" ), bbox_inches='tight')
    plt.close()

    plt.figure(figsize=[10,10])
    plt.scatter(store[:,0], store[:,1])
    plt.xlabel('bandwidth')
    plt.ylabel('cross_val_score')
    plt.savefig(str(mcmcDir / "bandwidth.png" ), bbox_inches='tight')
    plt.close()

    mcmcDir = Path(cache.settings['resultsDirMCMC'])

    h5_data = H5()
    h5_data.filename = (mcmcDir / "kde_settings.h5").as_posix()
    h5_data.load()

    dir_base = cache.settings.get('resultsDirBase')
    file = dir_base / 'kde_data.h5'

    kde_data = H5()
    kde_data.filename = file.as_posix()
    kde_data.load()

    times = {}
    values = {}
    for key in kde_data.root.keys():
        if '_time' in key:
            times[key.replace('_time', '')] = kde_data.root[key]
        if '_unit' in key:
            values[key] = kde_data.root[key]

    error_model = mcmcDir / "error_model"
    error_model.mkdir(parents=True, exist_ok=True)

    for key,value in values.items():
        time = times[key.split('_unit', 1)[0]]
        plt.figure(figsize=[10,10])
        row, col = value.shape
        alpha = 1.0/row
        plt.plot(time, value.T, 'g', alpha=alpha)
        plt.xlabel('time')
        plt.ylabel('concentration')
        plt.savefig((error_model / ("%s.png" % key) ).as_posix(), bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    main()


