#################
Plato
#################

These pages provide the documentation of the `plato` python package. 
`plato` was developed to provide a simple, flexible, and efficient way to perform geodynamic analyses of global plate reconstructions in GPlates.
Here you'll find the relevant documentation as well as a set of notebooks with examples of such analyses.

Plato is tested on Python 3.10.12. In theory, it should work well on any system that has
access to Conda. 

Quickstart
----------

To download the repo, create a Conda environment, and install all dependencies, run the
following: 

.. code-block:: bash    

    > $ git clone https://github.com/thomas-schouten/plato.git
    > $ cd plato
    > $ conda env create -f environment.yml
    > $ conda activate plato
    > $ pip install -e .

The resulting Conda environment should be able to run all notebooks found in
hmclab/notebooks. See the installation page for more detailed instructions.


.. toctree::
    :maxdepth: 1
    :caption: Contents:
    :hidden:

    self
    plato
    setup
    notebooks
    api/index
    genindex

.. centered:: Thomas Schouten, 2024

