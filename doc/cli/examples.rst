CADETMatch Examples
-------------------

Generating Examples and running them
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CADETMatch can generate a complete set of examples and run them. This can be used to verify that all features used in CADETMatch
are working correctly on your installation and to see how to run CADETMatch in various situations. The examples include variable transforms,
search strategies, and search stratgies. Depending on the hardware available running the complete set of examples can hours to days.

Generate examples
^^^^^^^^^^^^^^^^^

The examples can be generated in any directory you need and only the path to the directory and path to the version of CADET you want to use
needs to be supplied. You can have multiple example directories that are run with different versions of CADET to verify a new installation is
working correctly.

The examples generated can all be run individually and the files used as a template to make your own matches. There are examples for every feature in 
CADETMatch.

.. code-block:: bash

    python -m CADETMatch --generate_examples <example directory> --cadet_examples <path to cadet binary or library>


``--example_population <integer>``   can be added to set the population per parameter

``--example_mcmc_population <integer>`` can be added to set the MCMC population

Run examples
^^^^^^^^^^^^

This will run all the examples and could take hours to days to run depending on the available computing power. It is not recommended to run
this on a laptop.

.. code-block:: bash

    python -m CADETMatch --run_examples <example directory>


Clean examples
^^^^^^^^^^^^^^

This command will clean all the results from the examples and return them to a clean state so that the examples can be run again. This
is especially useful if there is a problem with the CADET install and it needed to be rebuilt.

.. code-block:: bash

    python -m CADETMatch --clean_examples <example directory>

Results examples
^^^^^^^^^^^^^^^^

This command processes all of the run examples and creates a text report of the best matches from every test.

.. code-block:: bash

    python -m CADETMatch --results_examples <example directory>
