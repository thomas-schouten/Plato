# Plato
## Thomas Schouten

<!-- [![codecov](https://codecov.io/gh/larsgeb/hmclab/branch/master/graph/badge.svg?token=6svV9YDRhd)](https://codecov.io/gh/larsgeb/hmclab) [![license](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![GitHub releases](https://img.shields.io/badge/download-latest%20release-green.svg)](https://github.com/larsgeb/hmclab/releases/latest) -->

**Plato** is an object-oriented Python package that provide an efficient and standardised workflow to analyse the motions of tectonic plates using the torque balance expected from their reconstructed geometries.

This package leverages [GPlately](https://gplates.github.io/gplately/v1.3.0/) to interrogate quantitative plate reconstructions built using [GPlates](https://www.gplates.org). GPlates is open-source application software offering a novel combination of interactive plate-tectonic reconstructions, geographic information system functionality and raster data visualisation.

The package is based on the algorithm initially published by [Clennett et al. (2023)](https://www.nature.com/articles/s41598-023-37117-w).

<!-- - **Website:** https://hmclab.science -->
<!-- - **Python documentation:** https://python.hmclab.science -->
- **Source code:** https://github.com/thomas-schouten/plato
<!-- - **Docker image:** https://hub.docker.com/repository/docker/larsgebraad/hmclab -->
- **Bug reports:** https://github.com/thomas-schouten/plato/issues

<!-- It provides all the ingredients to set up probabilistic (and deterministic) inverse
problems, appraise them, and analyse them. This includes a plethora of prior
distributions, different physical modelling modules and various MCMC (and
other) algorithms. 

In particular it provides prior distributions, physics and appraisal algorithms.

**Prior distributions:**
- Normal
- Laplace
- Uniform
- Arbitrary composites of other priors
- Bayes rule
- User supplied distributions -->

# Online tutorial notebooks

All tutorial notebooks can also be accessed online in a non-interactive fashion. Simply 
use https://python.hmclab.science or use the following links:

- [Getting_started.ipynb](notebooks/tutorials/0%20-%20Getting%20started.ipynb)
- [Example_workflow_PlateTorques.ipynb](notebooks/tutorials/1%20-%20Tuning%20Hamiltonian%20Monte%20Carlo.ipynb)
- [Example_workflow_Slabs.ipynb](notebooks/tutorials/2%20-%20Separate%20priors%20per%20dimension.ipynb)

# The long way around: installing the package on your system

For full installation instructions, including creating a proper Python environment, [see the installation instructions](https://python.hmclab.science/setup.html). 

Directly to your environment:

```
pip install -e git+git@github.com:thomas-schouten/plato.git@master#egg=hmclab
```

From the project root directory:

```
pip install -e .
```

### Development dependencies

If you want to develop within this repo, we recommend a few extra packages. They can also be installed using pip.

In Bash:

```
pip install -e git+git@github.com:thomas-schouten/plato.git@master#egg=plato[dev] # from github repo
pip install -e .[dev] # from local clone
```

... or Zsh (which requires escapes for brackets):

```
pip install -e git+git@github.com:thomas-schouten/plato.git@master#egg=plato\[dev\] # from github repo
pip install -e .\[dev\] # from local clone
```
