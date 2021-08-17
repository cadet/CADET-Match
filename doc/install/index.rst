.. _install:

Install
-------

CADET-Match is a parameter estimation and error modeling library for Python. CADET-Match has a number of
dependencies and they can be automatically installed. 

The Python libraries for CADET don't include CADET and it must be installed separately by following the
instructions on `cadet.github.io <https://cadet.github.io/getting_started/installation.html>`_.

We recommend installing `Anaconda <https://www.anaconda.com/>`_.
Anaconda is a high-performance scientific distribution of Python that includes many common packages needed for scientific and engineering work.
Download the installer from their `website <https://www.anaconda.com/>`_ and run it for the local user.

After installing Anaconda open an `Anaconda Shell` and run the following commands:

.. code-block:: bash

    conda config --add channels anaconda-fusion
    conda config --add channels conda-forge

    conda install -c immudzen cadetmatch


If you prefer to install from PyPi you can run

.. code-block:: bash

   pip install cadetmatch

The PyPi packages are updated more frequently than the conda packages are.
