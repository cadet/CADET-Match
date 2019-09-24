import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name="CADETMatch",
    version="0.3",
    author="William Heymann",
    author_email="w.heymann@fz-juelich.de",
    description="CADETMatch is a parameter estimation and error modeling library for CADET",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/modsim/CADET-Match",
    packages=setuptools.find_packages(),
    install_requires=[
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
          'h5py',
          'CADET',
          'seaborn'
      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
) 