from distutils.core import setup
setup(
  name = 'CADETMatch',         # How you named your package folder (MyLib)
  packages = ['CADETMatch', 'CADETMatch.scores', 'CADETMatch.search', 'CADETMatch.transform'],   # Chose the same as "name"
  version = '0.1',      # Start with a small number and increase it with every change you make
  license='GPL-3.0-only',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'CADETMatch is a parameter estimation and error modeling library for CADET',   # Give a short description about your library
  author = 'William Heymann',                   # Type in your name
  author_email = 'w.heymann@fz-juelich.de',      # Type in your E-Mail
  url = 'https://github.com/modsim/CADET-Match',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/modsim/CADET-Match/archive/v_05.tar.gz',    # I explain this later on
  keywords = ['CADET', 'parameter estimation', 'MCMC'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'joblib',
          'addict',
          'emcee',
          'SAlib',
          'corner',
          'deap',
          'scoop',
          'psutil',
          'openpyxl',
          'numpy',
          'scipy',
          'matplotlib',
          'pandas',
          'h5py'
      ],
  classifiers=[
    'Development Status :: 4 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Scientific/Engineering',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',   # Again, pick a license
    'Programming Language :: Python :: 3.6',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.7',
  ],
)