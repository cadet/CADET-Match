Search
------

CADETMatch uses a number of different search strategies for parameter estimation.
Parameters that are common to more than one search method will be repeated.
The choice of search strategy is set by using searchMethod. 

=================== =========== ================ ========== =========================================================================================================
 Key                  Values       Default        Required     Description
=================== =========== ================ ========== =========================================================================================================
searchMethod           String       NSGA3            No       Select the search method to use. (NSGA3, Multistart, GraphSpace, MCMC, AltScore, Gradient, ScoreTest)
=================== =========== ================ ========== =========================================================================================================

Most of the search strategies are population based and capable of running in parallel. 

NSGA3
^^^^^

NSGA3 is a genetic algorithm designed for high-dimensional problems.
If only a single objective is used the system will automatically switch to NSGA2.

=================== =========== ================ ========== =================================================================================================================================
 Key                  Values       Default        Required     Description
=================== =========== ================ ========== =================================================================================================================================
population            Integer       100            No        Set the population size per parameter estimated.
generations           Integer       1000           No        Set the max number of generations to population size * generations.
sobolGeneration       Boolean       True           No        Use a sobol sequence to generate the initial population instead of random initialization
stallGenerations      Integer       10             No        Terminate optimization after this many generations without progress
stallCorrect          Integer       5              No        Increase population size (if allowed) after this many generations without progress
progressCorrect       Integer       5              No        Decrease population size (if allowed) after this many generations with progress
minPopulation         Integer     population       No        Allow CADETMatch to decrease the population to this size if progress is fast
maxPopulation         Integer     population       No        Allow CADETMatch to increase the population to this size if progress stalls
stopAverage           Float         0.0            No        Stop searching when the average value of all metrics is less than or equal to this value
stopBest              Float         0.0            No        Stop searching when the highest metric is less than or equal to this value
stopRMSE              Float         0.0            No        Stop when the Root Mean Square Error is less than or equal to this value
gradCheck             Float         0.0            No        If the geometric mean of the metrics drop below this value run gradient descent to refine the value
finalGradRefinement   Boolean       False          No        Take the final pareto front and refine it with gradient descent
gradFineStop          Float         1e-14          No        set xtol of scipy.optimize.least_squares
localRefine           String        gradient       No        Can be gradient or powell
continueMCMC          Boolean       False          No        Once search finishes start MCMC and carry over useful information automatically
seeds                 List          None           No        Optimization can be seeded with specific values to test
MultiObjectiveSSE     Boolean       False          No        If set to true SSE objectives are split apart and turned into a multi-objective system. This is useful for multiple experiments.
=================== =========== ================ ========== =================================================================================================================================

Multistart
^^^^^^^^^^

This is a simple MultiStart gradient descent search.
If your problem is known to be solveable with gradient descent this can be a good search choice.
While this search algorithm can be used as a precursor for MCMC that is not recomended.   

=================== =========== ================ ========== =================================================================================================================================
 Key                  Values       Default        Required     Description
=================== =========== ================ ========== =================================================================================================================================
population            Integer       100            No        Set the population size per parameter estimated.
sobolGeneration       Boolean       True           No        Use a sobol sequence to generate the initial population instead of random initialization
gradFineStop          Float         1e-14          No        set xtol of scipy.optimize.least_squares
localRefine           String        gradient       No        Can be gradient or powell
continueMCMC          Boolean       False          No        Once search finishes start MCMC and carry over useful information automatically
seeds                 List          None           No        Optimization can be seeded with specific values to test
=================== =========== ================ ========== =================================================================================================================================


GraphSpace
^^^^^^^^^^

The purpose of GraphSpace is to create a map of your problem.
It is designed to be used with an extremely large population and it will just evaluate the goal, generate graphs and terminate.
It will not actually optimize.
What it does tell you is what variable ranges are reasonable to optimize in.
This is a good first step for any new problem where the ranges of the parameters is extremely large and the viable ranges are unknown.

=================== =========== ================ ========== =================================================================================================================================
 Key                  Values       Default        Required     Description
=================== =========== ================ ========== =================================================================================================================================
population            Integer       100            No        Set the population size per parameter estimated.
sobolGeneration       Boolean       True           No        Use a sobol sequence to generate the initial population instead of random initialization
gradFineStop          Float         1e-14          No        set xtol of scipy.optimize.least_squares
localRefine           String        gradient       No        Can be gradient or powell
continueMCMC          Boolean       False          No        Once search finishes start MCMC and carry over useful information automatically
seeds                 List          None           No        Optimization can be seeded with specific values to test
=================== =========== ================ ========== =================================================================================================================================

.. _mcmc-search:

MCMC
^^^^

MCMC is used for error modeling.
It is highly suggested that NSGA3 is used with continueMCMC.
MCMC is not an optimization algorithm and should not be used as one.
In order to run an error model is also required.

=================== =========== ================ ========== =====================================================================================================================================
 Key                  Values       Default        Required     Description
=================== =========== ================ ========== =====================================================================================================================================
MCMCpopulation        Integer      population        No        Set the population size per parameter estimated.
MCMCpopulationSet     Integer       None             No        Set the total population size (overrides MCMCpopulation)
mcmc_h5               Path          None             No        This is the path to the mcmc.h5 file generated in a previous MCMC run to use as a prior. 
PreviousResults       Path          None             No*       MCMC is not an optimization algorithm and should not be directly run. Use NSGA3 and continueMCMC True and this will be auto set.
MCMCTauMult           Integer       50               No        MCMC runs until the chain length/integrated autocorrelation time < MCMCTauMult+2 (2 for burn in)
=================== =========== ================ ========== =====================================================================================================================================

AltScore
^^^^^^^^

AltScore is very rarely used.
What it allows is reading another completed result and will just re-evaluate the entries of the pareto front with a different goal and report the results.
This can be useful for goal design to see the impact of combining different scores and if that would make the problem easier or harder to optimize.

=================== =========== ================ ========== =====================================================================================================================================
 Key                  Values       Default        Required     Description
=================== =========== ================ ========== =====================================================================================================================================
PreviousResults       Path          None             Yes       Look at the previous results and reevaluate the pareto front with a different goal (useful to see the impact of different scores)
=================== =========== ================ ========== =====================================================================================================================================


Gradient
^^^^^^^^

This is a simple test search algorithm that reads seeds and runs gradient descent from them.
This search algorithm has little practical usage and requires good starting points.

=================== =========== ================ ========== =================================================================================================================================
 Key                  Values       Default        Required     Description
=================== =========== ================ ========== =================================================================================================================================
gradFineStop          Float         1e-14          No        set xtol of scipy.optimize.least_squares
localRefine           String        gradient       No        Can be gradient or powell
seeds                 List          None           Yes        Optimization can be seeded with specific values to test
=================== =========== ================ ========== =================================================================================================================================

ScoreTest
^^^^^^^^^

This is the simplest of all the systems and just designed for testing goals.
Given one or more seeds they are simulated, scored and results returned.

=================== =========== ================ ========== =================================================================================================================================
 Key                  Values       Default        Required     Description
=================== =========== ================ ========== =================================================================================================================================
seeds                 List          None           Yes        Optimization can be seeded with specific values to test
=================== =========== ================ ========== =================================================================================================================================

