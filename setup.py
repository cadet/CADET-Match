import setuptools

from CADETMatch import version

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name=version.__title__,
    version=version.__version__,
    author=version.__author__,
    author_email=version.__email__,
    description=version.__summary__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/modsim/CADET-Match",
    packages=setuptools.find_packages(),
    install_requires=[
          'joblib>=0.15.1',
          'addict>=2.2.1',
          'emcee>=3.0.2',
          'SAlib',
          'corner>=2.1.0',
          'deap>=1.3.1',
          'psutil>=5.7.0',
          'openpyxl>=3.0.3',
          'numpy>=1.18.5',
          'scipy>=1.5.0',
          'matplotlib>=3.2.1',
          'pandas>=1.0.5',
          'h5py>=2.10.0',
          'CADET>=0.8',
          'seaborn>=0.10.1',
          'scikit-learn>=0.23.1',
          'importlib-metadata>=1.7.0',
          'jstyleson>=0.0.2',
          'filelock'
      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
) 