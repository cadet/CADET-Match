Running
^^^^^^^

CADETMatch is designed to be used through a command line interface. It is also possible to use CADETMatch using Jupyter notebooks but this
is not recommended for normal usage. CADETMatch has automatic checkpointing and if matching is interrupted it will resume from the last step
completed.

These commands can be run from the Windows command line or from any Linux command line. These commands can also be run through cluster
control software. 

``python -m CADETMatch`` is the basis of all the commands and other options can be added as needed.

``--json <path to json configuration file>``

``-n <int number of parallel processes to use>`` If -n is omitted one process will be created for each core on the machine. 

``--match`` Start parameter estimation or error modeling

``--generate_map`` Find the maximum a posteriori for an MCMC run 


The following options are for genering graphs. CADETMatch generates graphs automatically periodically but these commands
can be used to generate graphs immediately. These commands should be safe to run while matching is proceeding but they
may fail if the required data has not been generated yet.

``--generate_corner`` Generate corner graphs for MCMC

``--generate_graphs`` Generate progress graphs and 2D graphs

``--generate_graphs_all`` Generate progress graphs, 2D graphs, and 3D graphs

``--generate_graphs_autocorr`` Generate auto-correlation graphs

``--generate_mcmc_plot_tube`` Generate MCMC plot tubes (probability plots of chromatograms based on MCMC posterior)

``--generate_mcmc_mixing`` Generate MCMC mixing graphs


Starting a match
----------------

.. code-block:: bash

    python -m CADETMatch --match --json <path to connfiguration file> 


Generate graphs using 6 cores
-----------------------------

.. code-block:: bash

    python -m CADETMatch --generate_graphs --json <path to connfiguration file> -n 6


Running CADETMatch remotely
---------------------------

The most common way CADETMatch is used is by logging into a server with SSH and starting a matching session.
These matches can take hours to days to complete and the following command allows CADETMatch to keep running
once you logout.

.. code-block:: bash

    nohup python -m CADETMatch --match --json <path to connfiguration file> &

.. toctree::
    :glob:
    :hidden:
    
    *
