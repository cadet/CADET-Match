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
          'joblib>=0.14.0',
          'addict>=2.2.1',
          'emcee>=3',
          'SAlib',
          'corner>=2.0.1',
          'deap>=1.3.0',
          'psutil>=5.6.3',
          'openpyxl>=3.0.0',
          'numpy>=1.16.2',
          'scipy>=1.3.1',
          'matplotlib>=3.1.1',
          'pandas>=0.25.1',
          'h5py>=2.10.0',
          'CADET>=0.2',
          'seaborn>=0.9',
          'sklearn',
          'scikit-learn>=0.21',
      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
) 