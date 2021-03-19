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


