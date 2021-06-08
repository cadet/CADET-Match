Graphing
^^^^^^^^

Graphing in CADETMatch is done asyncronously from the normal running however it still takes computing time to generate the graphs.
There is a tradeoff between how frequently graphs are generated to see the progress and how much progress is slowed by generating the graphs.
Graphs are generated at the end of a generation/step during estimation/error modeling.
If the time in seconds since the last time graphs where generated at the end of a step than the specified time then the graphs are generated again and the time is reset.

All times are in seconds.

=================== =========== ================ ========== =================================================================================================
 Key                  Values       Default        Required     Description
=================== =========== ================ ========== =================================================================================================
graphGenerateTime     Integer       3600              No       Generate progress graphs and graphs of the simulations on the pareto front
graphMetaTime         Integer       1200              No       Generate progress graphs only
=================== =========== ================ ========== =================================================================================================


