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

if __name__ == "__main__":
    main()


