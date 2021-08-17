.. _configuration:

Configuration
-------------

CADET-Match uses a JSON based configuration file format.
CADET-Match uses `jstyleson <https://github.com/linjackson78/jstyleson>`_ to support JSON with comments.
You can programatically create the JSON in Python using nested dictionaries or `addict <https://github.com/mewwts/addict>`_ and then convert to JSON.

Ususally, the configuration has to include:

* Basic setup of CADET-Match
* Connection with experimental data
* Configuration of features and scores
* Creation of reference model
* Seting up parameters (including transforms)
* Configuration of search algorithm

In this section, the general structure of the CADET-Match configuration is described.
For more information, see also the examples.


.. toctree::

    basic_config
    experiments
    scores
    transform
    search
    error
    graphing
    misc

