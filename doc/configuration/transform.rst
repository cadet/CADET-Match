Transform
---------

CADETMatch has many variable transforms available.
The most common ones will be covered first.
There are a few additional ones available in the code that are retained for backwards compatibility that will not be covered and their usage is not recomended.

Each parameter for parameter estimation or error modeling must be listed in the parameters list and a transform selected.

.. code-block:: json

    {
        "parameters": [
            {
            },
            {
            }
        ],
    }

Indexing
^^^^^^^^

An index is used when you want to directly index into the arrays to modify, this is very rarely used and mostly comes up with the MultiState SMA rates between states. 
All transforms use the same method for indexing. 

You either need to specific component and bound or specify index.
Indexing starts at 0.
A value of -1 is used to indicate a value that is not component specific (i.e. a scalar).

Scalar value:index=-1
Index at position n: index=n

Component 0 Bound state 0 would be component=0 bound=0.

Component 1 but independent of bound state would be component=0 bound=-1.

Independent of component or bound state would be component=-1 bound=-1.


Auto
^^^^

Auto is the most common transform.
This transform will convert from the upper and lower boundary to a range of 0 to 1 for the search algorithm.
It will automatically switch between a linear and a log transform as needed for search performance. 

=================== =========== ================ ========== =========================================================================================================
 Key                  Values       Default        Required     Description
=================== =========== ================ ========== =========================================================================================================
min                   Float        None             Yes       minimum allowed value
max                   Float        None             Yes       maximum allowed value
location              String       None             Yes       path inside the simulation to the variable
maxFactor             Float        1000             No        if max/min > maxFactor a log transform is used.
component             Integer      None             Yes*      Index of the component used (starts at 0)
bound                 Integer      None             Yes*      Index of the bound state (starts at 0)
index                 Integer      None             Yes*      Index into the location (starts at 0) Either component + bound or index must be specificed not both.
experiments           String       None             No        If provided this variable is only used for the named experiments
=================== =========== ================ ========== =========================================================================================================

.. code-block:: json

    {
        "transform": "auto",
        "component": 2,
        "bound": 0,
        "location": "/input/model/unit_001/adsorption/SMA_NU",
        "min": 1,
        "max": 50
    }

Auto Inverse
^^^^^^^^^^^^

Auto-inverse works like auto except it transforms used 1/variable.
This transform was designed to deal with coupling of film diffusion and pore diffusion and improves optimization. 

=================== =========== ================ ========== =========================================================================================================
 Key                  Values       Default        Required     Description
=================== =========== ================ ========== =========================================================================================================
min                   Float        None             Yes       minimum allowed value
max                   Float        None             Yes       maximum allowed value
location              String       None             Yes       path inside the simulation to the variable
maxFactor             Float        1000             No        if max/min > maxFactor a log transform is used.
component             Integer      None             Yes*      Index of the component used (starts at 0)
bound                 Integer      None             Yes*      Index of the bound state (starts at 0)
index                 Integer      None             Yes*      Index into the location (starts at 0) Either component + bound or index must be specificed not both.
experiments           String       None             No        If provided this variable is only used for the named experiments
=================== =========== ================ ========== =========================================================================================================

.. code-block:: json

    {
        "transform": "auto_inverse",
        "component": -1,
        "bound": -1,
        "location": "/input/model/unit_001/FILM_DIFFUSION",
        "min": 1e-9,
        "max": 1e-5
    }

Auto kEQ
^^^^^^^^

This transforms convert from kA and kD to kA and kEQ with all the other properties of auto.
In reality kA and kD are coupled and this allows the search algorithm to see the coupling.
There are also some fits where kA and kD are fast enough that a system is effectively in rapid equilibrium.
Without this transform a large number of kA and kD values will be found with equally good results.
With this transform kEQ will have a definite value and there will be a large range of kA values which provides more understanding for the problem.

=================== =========== ================ ========== =========================================================================================================
 Key                  Values       Default        Required     Description
=================== =========== ================ ========== =========================================================================================================
minKA                 Float        None             Yes       minimum allowed value
maxKA                 Float        None             Yes       maximum allowed value
minKEQ                Float        None             Yes       minimum allowed value
maxKEQ                Float        None             Yes       maximum allowed value
location              String       None             Yes       paths inside the simulation kA and kD
maxFactor             Float        1000             No        if max/min > maxFactor a log transform is used.
component             Integer      None             Yes*      Index of the component used (starts at 0)
bound                 Integer      None             Yes*      Index of the bound state (starts at 0)
index                 Integer      None             Yes*      Index into the location (starts at 0) Either component + bound or index must be specificed not both.
experiments           String       None             No        If provided this variable is only used for the named experiments
=================== =========== ================ ========== =========================================================================================================

.. code-block:: json

    {
        "transform": "auto_keq",
        "component": 0,
        "bound": 0,
        "location": [
            "/input/model/unit_001/adsorption/LIN_KA",
            "/input/model/unit_001/adsorption/LIN_KD"
        ],
        "minKA": 1e-8,
        "maxKA": 1e8,
        "minKEQ": 1e-4,
        "maxKEQ": 1e4
    }

Norm Add
^^^^^^^^

This transform allows another parameter to be read and a fixed or variable value added to it and assigned to a second variable.
For example if you are optimizing the charge nu for SMA with a few different charge variants you may not know all the charge variants but you know they are all close together and so you can estimate one and then use norm_add for the others to require they are close.

=================== =========== ================ ========== =========================================================================================================
 Key                  Values       Default        Required     Description
=================== =========== ================ ========== =========================================================================================================
min                   Float        None             Yes       minimum allowed value
max                   Float        None             Yes       maximum allowed value
locationFrom          String       None             Yes       paths inside the simulation kA and kD
componentFrom         Integer      None             Yes*      Index of the component used (starts at 0)
boundFrom             Integer      None             Yes*      Index of the bound state (starts at 0)
indexFrom             Integer      None             Yes*      Index into the location (starts at 0) Either component + bound or index must be specificed not both.
locationTo            String       None             Yes       paths inside the simulation kA and kD
componentTo           Integer      None             Yes*      Index of the component used (starts at 0)
boundTo               Integer      None             Yes*      Index of the bound state (starts at 0)
indexTo               Integer      None             Yes*      Index into the location (starts at 0) Either component + bound or index must be specificed not both.
experiments           String       None             No        If provided this variable is only used for the named experiments
=================== =========== ================ ========== =========================================================================================================

.. code-block:: json

    {
        "transform": "norm_add",
        "locationFrom": "/input/model/unit_001/COL_POROSITY",
        "componentFrom": -1,
        "boundFrom": -1,
        "locationTo": "/input/model/unit_001/PAR_POROSITY",
        "componentTo": -1,
        "boundTo": -1,
        "min": -0.1,
        "max": 0.1
    }

Norm Mult
^^^^^^^^^

This transform allows another parameter to be read and a fixed or variable value multiplied to it and assigned to a second variable.
For instance if you are estimating the shielding factor sigma for a monomer and also need to estimate it for a dimer you can estimated sigma for the monomer normally and then specify that the dimer is approximately twice as large.

=================== =========== ================ ========== =========================================================================================================
 Key                  Values       Default        Required     Description
=================== =========== ================ ========== =========================================================================================================
min                   Float        None             Yes       minimum allowed value
max                   Float        None             Yes       maximum allowed value
locationFrom          String       None             Yes       paths inside the simulation kA and kD
componentFrom         Integer      None             Yes*      Index of the component used (starts at 0)
boundFrom             Integer      None             Yes*      Index of the bound state (starts at 0)
indexFrom             Integer      None             Yes*      Index into the location (starts at 0) Either component + bound or index must be specificed not both.
locationTo            String       None             Yes       paths inside the simulation kA and kD
componentTo           Integer      None             Yes*      Index of the component used (starts at 0)
boundTo               Integer      None             Yes*      Index of the bound state (starts at 0)
indexTo               Integer      None             Yes*      Index into the location (starts at 0) Either component + bound or index must be specificed not both.
experiments           String       None             No        If provided this variable is only used for the named experiments
=================== =========== ================ ========== =========================================================================================================

.. code-block:: json

	{
		"transform": "norm_add",
		"locationFrom": "/input/model/unit_001/COL_POROSITY",
		"componentFrom": -1,
		"boundFrom": -1,
		"locationTo": "/input/model/unit_001/PAR_POROSITY",
		"componentTo": -1,
		"boundTo": -1,
		"min": 0.8,
		"max": 1.5			
	}

Set Value
^^^^^^^^^

This transform copies a value from another estimated value.
One of the common usage cases is when estimating the axial dispersion of the tubing.
It can be a good assumption that the axial dispersion is the same in the tubing leading to the column and the tubing leaving it so with this one of them is estimated and the value copied to the other one so fewer values need to be estimated.

=================== =========== ================ ========== =========================================================================================================
 Key                  Values       Default        Required     Description
=================== =========== ================ ========== =========================================================================================================
locationFrom          String       None             Yes       paths inside the simulation kA and kD
componentFrom         Integer      None             Yes*      Index of the component used (starts at 0)
boundFrom             Integer      None             Yes*      Index of the bound state (starts at 0)
indexFrom             Integer      None             Yes*      Index into the location (starts at 0) Either component + bound or index must be specificed not both.
locationTo            String       None             Yes       paths inside the simulation kA and kD
componentTo           Integer      None             Yes*      Index of the component used (starts at 0)
boundTo               Integer      None             Yes*      Index of the bound state (starts at 0)
indexTo               Integer      None             Yes*      Index into the location (starts at 0) Either component + bound or index must be specificed not both.
experiments           String       None             No        If provided this variable is only used for the named experiments
=================== =========== ================ ========== =========================================================================================================

.. code-block:: json

	{
		"transform": "set_value",
		"locationFrom": "/input/model/unit_000/sec_000/CONST_COEFF",
		"componentFrom": 0,
		"boundFrom": 0,
		"locationTo": "/input/model/unit_000/sec_000/CONST_COEFF",
		"componentTo": 1,
		"boundTo": 0
	}

Sum
^^^

This transform reads two values and assigns it to a 3rd value.
This was created for a situation where the volume of two CSTRs where estimated and a 3rd CSTR needed to have a volume equal to the sum of the first two.

=================== =========== ================ ========== =========================================================================================================
 Key                  Values       Default        Required     Description
=================== =========== ================ ========== =========================================================================================================
location1            String       None             Yes       paths inside the simulation kA and kD
component1           Integer      None             Yes*      Index of the component used (starts at 0)
bound1               Integer      None             Yes*      Index of the bound state (starts at 0)
index1               Integer      None             Yes*      Index into the location (starts at 0) Either component + bound or index must be specificed not both.
location2            String       None             Yes       paths inside the simulation kA and kD
component2           Integer      None             Yes*      Index of the component used (starts at 0)
bound2               Integer      None             Yes*      Index of the bound state (starts at 0)
index2               Integer      None             Yes*      Index into the location (starts at 0) Either component + bound or index must be specificed not both.
locationSum          String       None             Yes       paths inside the simulation kA and kD
componentSum         Integer      None             Yes*      Index of the component used (starts at 0)
boundSum             Integer      None             Yes*      Index of the bound state (starts at 0)
indexSum             Integer      None             Yes*      Index into the location (starts at 0) Either component + bound or index must be specificed not both.
experiments          String       None             No        If provided this variable is only used for the named experiments
=================== =========== ================ ========== =========================================================================================================

.. code-block:: json

	{
		"transform": "sum",
		"location1": "/input/model/unit_001/INIT_VOLUME",
		"component1": -1,
		"bound1": -1,
		"location2": "/input/model/unit_003/INIT_VOLUME",
		"component2": -1,
		"bound2": -1,
		"locationSum": "/input/model/unit_004/INIT_VOLUME",
		"componentSum": -1,
		"boundSum": -1		
	}

Norm Diameter
^^^^^^^^^^^^^

CADET uses the cross sectional area of the column and tubing and measuring this precisely can be difficult.
It is often much simpler to measure the diameter and provide a small search range and then allow this transform to convert that to the area.
This assumes circular tubing and uses Area = pi*d^2/4.

=================== =========== ================ ========== =========================================================================================================
 Key                  Values       Default        Required     Description
=================== =========== ================ ========== =========================================================================================================
min                   Float        None             Yes       minimum allowed value
max                   Float        None             Yes       maximum allowed value
location              String       None             Yes       paths inside the simulation kA and kD
component             Integer      None             Yes*      Index of the component used (starts at 0)
bound                 Integer      None             Yes*      Index of the bound state (starts at 0)
index                 Integer      None             Yes*      Index into the location (starts at 0) Either component + bound or index must be specificed not both.
experiments           String       None             No        If provided this variable is only used for the named experiments
=================== =========== ================ ========== =========================================================================================================

.. code-block:: json

	{
		"transform": "norm_diameter",
		"location": "/input/model/unit_001/CROSS_SECTION_AREA",
		"min": 0.001,
		"max": 0.1,
		"component": -1,
		"bound": -1		
	}

Norm Volume Length
^^^^^^^^^^^^^^^^^^

When estimating the size of a Disperive Plug Flow Reactor needed to model a piece of tubing it is normal to estimate the dispersion, area and length.
This works but can be problematic to estimate and get a realistic estimate due to the degrees of freedom.
Finding the volume of the tube and the length of the tube is much easier to do accurately and this makes it a much better transform to work with.

=================== =========== ================ ========== =========================================================================================================
 Key                  Values       Default        Required     Description
=================== =========== ================ ========== =========================================================================================================
area_location         String       None             Yes       path to cross_section_area
length_location       String       None             Yes       path to col_length
component             Integer      None             Yes*      Index of the component used (starts at 0)
bound                 Integer      None             Yes*      Index of the bound state (starts at 0)
index                 Integer      None             Yes*      Index into the location (starts at 0) Either component + bound or index must be specificed not both.
minVolume             Float        None             Yes       minimum allowed value
maxVolume             Float        None             Yes       minimum allowed value
minLength             Float        None             Yes       minimum allowed value
maxLength             Float        None             Yes       minimum allowed value
experiments           String       None             No        If provided this variable is only used for the named experiments
=================== =========== ================ ========== =========================================================================================================

.. code-block:: json

	{
		"transform": "norm_volume_length",
		"area_location": "/input/model/unit_001/CROSS_SECTION_AREA",
		"length_location": "/input/model/unit_001/COL_LENGTH",
		"minVolume": 1e-06,
		"maxVolume": 0.0001,
		"minLength": 0.1,
		"maxLength": 0.3,
		"component": -1,
		"bound": -1		
	}

Norm Volume Area
^^^^^^^^^^^^^^^^

This transform works like the volume length transform except it uses volume an area and should only be used if it is easier to estimate the cross sectional area of the tubing than its length.

=================== =========== ================ ========== =========================================================================================================
 Key                  Values       Default        Required     Description
=================== =========== ================ ========== =========================================================================================================
area_location         String       None             Yes       path to cross_section_area
length_location       String       None             Yes       path to col_length
component             Integer      None             Yes*      Index of the component used (starts at 0)
bound                 Integer      None             Yes*      Index of the bound state (starts at 0)
index                 Integer      None             Yes*      Index into the location (starts at 0) Either component + bound or index must be specificed not both.
minVolume             Float        None             Yes       minimum allowed value
maxVolume             Float        None             Yes       minimum allowed value
minArea               Float        None             Yes       minimum allowed value
maxArea               Float        None             Yes       minimum allowed value
experiments           String       None             No        If provided this variable is only used for the named experiments
=================== =========== ================ ========== =========================================================================================================

.. code-block:: json

	{
		"transform": "norm_volume_area",
		"area_location": "/input/model/unit_001/CROSS_SECTION_AREA",
		"length_location": "/input/model/unit_001/COL_LENGTH",
		"minVolume": 1e-06,
		"maxVolume": 0.0001,
		"minArea": 1e-05,
		"maxArea": 0.001,
		"component": -1,
		"bound": -1		
	}

