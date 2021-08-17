Error Modeling
--------------

CADET-Match includes error modeling and parameter estimation.
Error modeling in CADET-Match is based on using a fitted simulation and creating an error model by manipulations to the fitted simulation.
Error modeling uses MCMC and can be a slow process which requires a lot of computing time.
Some simple problems can be solved in a few hours on a powerful desktop and others can take weeks on a powerful server.

:ref:`mcmc-search` search settings.

The error model is pretty simple.
The best fit simulation is used as a template.
Variations are made based on the template using the errors supplied.

Pump Delays are implemented using a uniform random distribution.
Any time a new section starts in CADET a pump delay may be applied.
Setting upper and lower bound to 0 disables this error.

Flow rate variations use a normal distribution with a supplied mean and standard deviation.
These numbers can usually be found from a pump manufacturer.
The flow rate in the simulation is multiplied by the pump flow error.
Setting the mean to 0 disables this error.

Loading concentration variations use a normal distribution with a supplied mean and standard deviation.
These numbers normally have to be determined from experiments.
The concentration is multiplied by the concentration error.
Setting the mean to 0 disables this error.

The UV error is modeled as a scale dependent error and a scale indepdennt error so that the total error applied to the chromatogram = signal * uv_noise_norm + uv_noise.
Both of the errors sources are the same length as the chromatogram.
UV noise norm almost always has a mean value of 1.0 and UV noise almost always has a mean noise of 0.0 since they are the multiplicative and additive identities respectively.

=================== =================== ================ ========== =========================================================================================================
 Key                  Values              Default        Required     Description
=================== =================== ================ ========== =========================================================================================================
name                  String                None             Yes       name of the experiment this error model applies to
units                List of Integers       None             Yes       unit numbers that uv noise should be applied to
delay                 [Float, Float]        None             Yes        min and max value of a uniform random distribution for pump delays
flow                  [Float, Float]        None             Yes        mean and standard deviation for a normal distribution
load                  [Float, Float]        None             Yes        mean and standard deviation for a normal distribution 
uv_noise_norm         [Float, Float]        None             No        mean and standard deviation for a normal distribution
uv_noise              [Float, Float]        None             No        mean and standard deviation for a normal distribution
=================== =================== ================ ========== =========================================================================================================


.. code-block:: json

	"errorModel": [
		{
			"file_path": "non.h5",
			"experimental_csv": "non.csv",
			"name": "main",
			"units": [2],
			"delay": [0.0, 2.0],
			"flow": [1.0, 0.001],
			"load": [1.0, 0.001],
			"uv_noise_norm": [1.0, 0.001]
		}
	],

