Scores
------

CADETMatch has many scores available.
Each score is designed to solve a particular type of problem. 
The most common ones will be covered first.
There are a few additional ones available in the code that are retained for backwards compatibility that will not be covered and their usage is not recomended.

.. code-block:: json

    {
    "scores": [
        {
        },
        {
        }
    ],
    }

Common
^^^^^^

A few things are common to nearly all the scores. 

Almost all of them support slicing where a start and stop time can be given and the score will only look in that time slice.
This is really useful if you know that a certain component corresponds to a particular peak.

The scores inherit csv and output_path from the experiment they are part of but these can also be specified on an individual score.
If you know a particular component is tied to a particular peak it can be given a separate output_path and csv file to improve matching.

The derivative of the shape function can be very useful in matching data.
Most chromatography peaks are not symmetrical and the derivative of the shape improves matching.

Another feature common to most of the scores is the idea of decay.
Based on working with experimentalists pumps often have small delays.
Decay controls matching the time offset of the signal.
Setting decay to 1 gives an immediate penalty and setting decay to 0 gives a reduced penalty for small offsets.
It is recomended to use decay = 1 when estimating porosity and setting decay = 0 for most other estimations.

Sum of squared error scores
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sum of squared errors is a common metric used in optimization but it caused problems with chromatography.
Between issues like pump delays and small flow rate variations there is a tendency for the peaks to shift.
A narrowly shifting peak is a particular problem for SSE and it causes the score to favor the peak being in the right place even if the shape is far from correct.
It is not generally recomended to use these scores.

When using multiple scores SSE and non-SSE based scores can't be mixed.
This is due to numerical issues. 

SSE based scores can't be used for error modeling.

Shape
^^^^^

Shape is the most general purpose score for fitting.
It looks at the similarity of the shape between synthetic and experimental data.
The score prioritizes the shape being accurate over the shape eluting at exactly the right time.
The shape is dictated by the physics of the reaction while the position can shift slightly due to things like pump delays.

=================== ============== ================ ========== =========================================================================================================
 Key                  Values          Default        Required     Description
=================== ============== ================ ========== =========================================================================================================
name                  String          None             Yes       name of the score, it must be unique
decay                 Boolean         False            No        set the decay for time offsets
derivative            Boolean         True             No        set using the derivative of the shape
start                 Float           None             No        Slice start in seconds
stop                  Float           None             No        Slice stop in seconds
csv                   Integer         None             No        This is the path to the csv file to match against. 
output_path          Path or List     None             No        Path or list of paths to the output data to match against
=================== ============== ================ ========== =========================================================================================================

.. code-block:: json

    {
		"name": "main_peak",
		"type": "Shape",
        "start": 100.0,
        "stop": 300.0
	}

Shape Front
^^^^^^^^^^^

Shape front is a modification of the shape score that only looks at the front of a peak.
This is especially important with breakthrough curves where the back of the peak may not even exist. 

=================== ============== ================ ========== =========================================================================================================
 Key                  Values          Default        Required     Description
=================== ============== ================ ========== =========================================================================================================
name                  String          None             Yes       name of the score, it must be unique
min_percent           Float           0.02             No        Percent of peak max to identify the bottom of the front of the peak
max_percent           Float           0.98             No        Percent of peak max to identify the top of the front of the peak
decay                 Boolean         False            No        set the decay for time offsets
derivative            Boolean         True             No        set using the derivative of the shape
start                 Float           None             No        Slice start in seconds
stop                  Float           None             No        Slice stop in seconds
csv                   Integer         None             No        This is the path to the csv file to match against. 
output_path          Path or List     None             No        Path or list of paths to the output data to match against
=================== ============== ================ ========== =========================================================================================================

.. code-block:: json

    {
        "name": "peak_front",
        "type": "ShapeFront",
        "min_percent": 0.05,
        "max_percent": 0.95
    }

Shape Back
^^^^^^^^^^

Shape back is a modification of the shape score that only looks at the back of a peak.
This is often used when a system starts saturated and there is no front side of the peak and only a back side exists.

=================== ============== ================ ========== =========================================================================================================
 Key                  Values          Default        Required     Description
=================== ============== ================ ========== =========================================================================================================
name                  String          None             Yes       name of the score, it must be unique
min_percent           Float           0.02             No        Percent of peak max to identify the bottom of the front of the peak
max_percent           Float           0.98             No        Percent of peak max to identify the top of the front of the peak
decay                 Boolean         False            No        set the decay for time offsets
derivative            Boolean         True             No        set using the derivative of the shape
start                 Float           None             No        Slice start in seconds
stop                  Float           None             No        Slice stop in seconds
csv                   Integer         None             No        This is the path to the csv file to match against. 
output_path          Path or List     None             No        Path or list of paths to the output data to match against
=================== ============== ================ ========== =========================================================================================================

.. code-block:: json

    {
        "name": "peak_back",
        "type": "ShapeBack",
        "start": 300,
        "stop": 600
    }

SSE
^^^

SSE is sum of squared errors and this is a typical score used in optimization.
In chromatography the pulses are narrow and have a tendency to shift position which makes this a hard score to optimize with.
For some problems this score may work but in general it is not advised.

This score can't be used with error modeling.

=================== ============== ================ ========== =========================================================================================================
 Key                  Values          Default        Required     Description
=================== ============== ================ ========== =========================================================================================================
name                  String          None             Yes       name of the score, it must be unique
start                 Float           None             No        Slice start in seconds
stop                  Float           None             No        Slice stop in seconds
csv                   Integer         None             No        This is the path to the csv file to match against. 
output_path          Path or List     None             No        Path or list of paths to the output data to match against
=================== ============== ================ ========== =========================================================================================================

.. code-block:: json

    {
        "name": "peak",
        "type": "SSE",
        "start": 300,
        "stop": 600
    }

Dextran Shape
^^^^^^^^^^^^^

Dextran shape is a special score designed to deal with the non-idealities of Dextran in a column.
It can be used with any non-ideal molecule where only part of the front of the peak can be used.
It automatically isolates as much of the front of peak and uses as much of the front of the peak as possible.

=================== ============== ================ ========== =========================================================================================================
 Key                  Values          Default        Required     Description
=================== ============== ================ ========== =========================================================================================================
name                  String          None             Yes       name of the score, it must be unique
start                 Float           None             No        Slice start in seconds
stop                  Float           None             No        Slice stop in seconds
csv                   Integer         None             No        This is the path to the csv file to match against. 
output_path          Path or List     None             No        Path or list of paths to the output data to match against
=================== ============== ================ ========== =========================================================================================================

.. code-block:: json

    {
        "name": "peak",
        "type": "DextranShape",
    }

Dextran SSE
^^^^^^^^^^^

Dextran SSE is uses the same slicing as Dextran shape but uses SSE instead of similarity metrics it used some of squared errors.

This score can't be used with error modeling.

=================== ============== ================ ========== =========================================================================================================
 Key                  Values          Default        Required     Description
=================== ============== ================ ========== =========================================================================================================
name                  String          None             Yes       name of the score, it must be unique
start                 Float           None             No        Slice start in seconds
stop                  Float           None             No        Slice stop in seconds
csv                   Integer         None             No        This is the path to the csv file to match against. 
output_path          Path or List     None             No        Path or list of paths to the output data to match against
=================== ============== ================ ========== =========================================================================================================

.. code-block:: json

    {
        "name": "peak",
        "type": "DextranSSE",
    }

Fractionation Slide
^^^^^^^^^^^^^^^^^^^

This score is used for fractionation.
It requires an additional csv file with the fractionation data. 

The fractionation files has 3 or more columns and it is easier to explain the the example below.
The first colums has a header of Start and the entries in the column are the start times of fractionation in seconds.
The second column has a header of Stop and the entries are the stop times of fractionation.
The times don't have to be continuous and can have gaps.

Each column after the first 2 starts with a header that is the component number and the values in the column are the concentration in mol/m^3 (mM) of the sample.
If there is no data for a sample in a particular fraction the entry can be left blank and it will be handled.  

In the case below 3 samples are collected from 400-450s, 450-500s, and 500-550s on component 0 and component 1.

======    =========  ============  ===========
Start      Stop           0             1
======    =========  ============  ===========
400	       450        0.0051	    0.0054
450        500        0.0178        0.0190
500        550        0.0265        0.0287
======    =========  ============  ===========


=================== ============== ================ ========== =========================================================================================================
 Key                  Values          Default        Required     Description
=================== ============== ================ ========== =========================================================================================================
name                  String          None             Yes       name of the score, it must be unique
unit_name             String          None             Yes       Name of the unit operation that is fractionated (usually outlet representing UV detector)
fraction_csv          Path            None             Yes       csv file with fractionation data
start                 Float           None             No        Slice start in seconds
stop                  Float           None             No        Slice stop in seconds
csv                   Integer         None             No        This is the path to the csv file to match against. 
output_path          Path or List     None             No        Path or list of paths to the output data to match against
=================== ============== ================ ========== =========================================================================================================

.. code-block:: json

    {
        "name": "fractionation",
        "type": "fractionationSlide",
        "unit_name": "unit_002",
        "fraction_csv": "frac.csv"
    }

Fractionation SSE
^^^^^^^^^^^^^^^^^

Fractionation based score using SSE and was written for testing purpose in a paper and is not generally recomended.  

This score can't be used with error modeling.


=================== ============== ================ ========== =========================================================================================================
 Key                  Values          Default        Required     Description
=================== ============== ================ ========== =========================================================================================================
name                  String          None             Yes       name of the score, it must be unique
unit_name             String          None             Yes       Name of the unit operation that is fractionated (usually outlet representing UV detector)
fraction_csv          Path            None             Yes       csv file with fractionation data
start                 Float           None             No        Slice start in seconds
stop                  Float           None             No        Slice stop in seconds
csv                   Integer         None             No        This is the path to the csv file to match against. 
output_path          Path or List     None             No        Path or list of paths to the output data to match against
=================== ============== ================ ========== =========================================================================================================

.. code-block:: json

    {
        "name": "fractionation",
        "type": "fractionationSSE",
        "unit_name": "unit_002",
        "fraction_csv": "frac.csv"
    }

Ceiling
^^^^^^^

Ceiling is a special case score.
It is almost always used with start and stop and is used to ensure nothing is above the defined value in the selected interval.
On some experimental systems that are running very close to overload a fit can be obtained that has a large amount of material coming off during loading.
This can be used as a restriction for that effect.  

=================== ============== ================ ========== =========================================================================================================
 Key                  Values          Default        Required     Description
=================== ============== ================ ========== =========================================================================================================
name                  String          None             Yes       name of the score, it must be unique
max_value             Float           None             Yes       max value that is allowed
start                 Float           None             No        Slice start in seconds
stop                  Float           None             No        Slice stop in seconds
csv                   Integer         None             No        This is the path to the csv file to match against. 
output_path          Path or List     None             No        Path or list of paths to the output data to match against
=================== ============== ================ ========== =========================================================================================================

.. code-block:: json

    {
        "name": "limit",
        "type": "Ceiling",
        "max_value": 0.05,
        "start": 0,
        "stop": 100
    }

Other
^^^^^

There are a few other scores that exist. 

AbsoluteTime and AbsoluteHeight are used by error modeling and can't be used for parameter estimation.
When continueMCMC=1 is set these scores are automatically added when needed.
It is not advisable to add these by hand.

There are many older variations of Shape, ShapeFront and ShapeBack and these are all obsolete now and the same things can be done with the derivative and decay options.

