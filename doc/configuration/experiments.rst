Experiments
-----------

CADETMatch supports one or more experiments with one or more components. 
The basic structure is shown below.

.. code-block:: json

    {
    "experiments": [
        {
            
        },
        {
            
        }
    ],
    }

The csv file is used for all scores unless a specific feature has its own csv file.
The csv file needs to have two columns.
The first is the time in seconds and the second is the concentration in mols/m^3.

Output path is the path to where the output is stored in CADET.
If a list is given instead of a single path then all the outputs specified are added together.
This simplifies dealing with fractionation data where often a sum signal and fractionation data is available.

=================== =============== ================ ========== =========================================================================================================
 Key                  Values          Default        Required     Description
=================== =============== ================ ========== =========================================================================================================
name                   String        None             Yes        Name of experiment, each name must be unique
csv                    Path          None             Yes        This is the path to the csv file to match against. 
HDF5                   Path          None             Yes        Path to a runnable HDF5 simulation
output_path          Path or List    None             Yes        Path or list of paths to the output data to match against
scores                 List          None             Yes        List of scores to use
=================== =============== ================ ========== =========================================================================================================

.. code-block:: json

	"experiments": [
		{
			"name": "dextran1",
			"csv": "dextran1.csv",
			"HDF5": "dextran1.h5",
			"output_path": "/output/solution/unit_002/SOLUTION_OUTLET_COMP_000",
			"scores": [
				
			]
		},
		{
			"name": "dextran2",
			"csv": "dextran2.csv",
			"HDF5": "dextran2.h5",
			"output_path": "/output/solution/unit_002/SOLUTION_OUTLET_COMP_000",
			"scores": [
				
			]
		}
		],

