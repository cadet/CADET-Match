.. _configuration:

Configuration
-------------

CADETMatch uses a JSON based configuration file format. CADETMatch uses `jstyleson <https://github.com/linjackson78/jstyleson>`_ to support JSON with comments.
You can programatically create the JSON in Python using nested dictionaries or `addict <https://github.com/mewwts/addict>`_ and then convert to JSON.

The basic setup is covered in this document. The sidebar has links to variable transforms, scores, searching and setting up experiments.

Basic setup
^^^^^^^^^^^

All paths and directories in the JSON file MUST use / and not \. This means paths on Windows need to be converted from C:\foo\bar to C:/foo/bar

=================== =========== ================ ========== =================================================================================================
 Key                  Values       Default        Required     Description
=================== =========== ================ ========== =================================================================================================
CADETPath             Path       None              Yes       Path to the CADET binary or shared library
baseDir               Path       None              No        If baseDir is given then all other paths are evaluated relative to baseDir
resultsDir            Path       None              Yes       Specifies where the results will be stored
checkpointFile        Path       "check"           No        Specifies the name of the checkpointFile. This is very rarely needed.
CSV                   Path       "results.csv"     No        Species the name of the main results file. 
metaResultsOnly       Boolean    1                 No        Species if the full pareto front simulations should be kept and graphed or only the meta front.
normalizeOutput       Boolean    1                 No        Normalize from 0 to 1 all input data and chromatograms
=================== =========== ================ ========== =================================================================================================

.. toctree::
    :glob:
    :hidden:
    
    *

Graphing
^^^^^^^^

Graphing in CADETMatch is done asyncronously from the normal running however it still takes computing time to generate the graphs. There is a tradeoff
between how frequently graphs are generated to see the progress and how much progress is slowed by generating the graphs. Graphs are generated at the
end of a generation/step during estimation/error modeling. If the time in seconds since the last time graphs where generated at the end of a step than
the specified time then the graphs are generated again and the time is reset.

All times are in seconds.

=================== =========== ================ ========== =================================================================================================
 Key                  Values       Default        Required     Description
=================== =========== ================ ========== =================================================================================================
graphGenerateTime     Integer       3600              No       Generate progress graphs and graphs of the simulations on the pareto front
graphMetaTime         Integer       1200              No       Generate progress graphs only
graphSpearman         Boolean       False             No       Generate Spearman graphs for estimated parameters and metrics
=================== =========== ================ ========== =================================================================================================

Miscellaneous
^^^^^^^^^^^^^

Most of these settings don't really fit with anything else and they should only be set when needed.

======================== =========== ================ ========== ====================================================================================================================================================
 Key                       Values       Default        Required     Description
======================== =========== ================ ========== ====================================================================================================================================================
fullTrainingData           Integer       0              No        This causes CADETMatch to store ALL results of all simulations for machine learning. This can require many gigabytes of storage.
dynamicTolerance           Boolean       False          No        Automatically adapt the tolerance of the simulation based on search method requirements.
abstolFactor               Float         1e-3           No        Set absTol for most searching to max(abstolFactor*smalltest_peak, abstolFactorGradMax*largest_peak)
abstolFactorGrad           Float         1e-7           No        Set absTol for gradient descent to max(abstolFactorGrad*smalltest_peak, abstolFactorGradMax*largest_peak)
abstolFactorGradMax        Float         1e-10          No        Set absTol for gradient descent to max(abstolFactorGrad*smalltest_peak, abstolFactorGradMax*largest_peak)
connectionNumberEntries    Integer       5              No        Set the length of each entry for the connections matrix
gradVector                 Boolean       False          No        If gradVector is set to False gradient descent uses the vector of metrics for minimization. If set to True it uses the smoothed chromatogram data    
======================== =========== ================ ========== ====================================================================================================================================================

Example
^^^^^^^

.. code-block:: json

    {
    "CADETPath": "/home/myhome/bin/cadet-cli",
    "baseDir": "/home/myhome/data",
    "resultsDir": "results1",
    "checkpointFile": "checkpoint",
    "CVS": "bypass_experiment.csv",
    "metaResultsOnly": 1,
    "normalizeOutput": 1
    }