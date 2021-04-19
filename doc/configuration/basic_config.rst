Basic setup
^^^^^^^^^^^

All paths and directories in the JSON file MUST use `/` and not `\\`. 
This means paths on Windows need to be converted from `C:\\foo\\bar` to `C:/foo/bar`.

=================== =========== ================ ========== =================================================================================================
 Key                  Values       Default        Required     Description
=================== =========== ================ ========== =================================================================================================
CADETPath             Path       None              Yes       Path to the CADET binary or shared library
baseDir               Path       None              No        If baseDir is given then all other paths are evaluated relative to baseDir
resultsDir            Path       None              Yes       Specifies where the results will be stored
checkpointFile        Path       "check"           No        Specifies the name of the checkpointFile. This is very rarely needed.
CSV                   Path       "results.csv"     No        Species the name of the main results file. 
metaResultsOnly       Boolean    True              No        Species if the full pareto front simulations should be kept and graphed or only the meta front.
normalizeOutput       Boolean    True              No        Normalize from 0 to 1 all input data and chromatograms
=================== =========== ================ ========== =================================================================================================

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

