# Optimization on multifractal loss landscapes explains a diverse range of geometrical and dynamical properties of deep learning

## Overview
This repository, `GD-multifractal-landscape`, provides all Python code to reproduce the results and analysis of the paper *Optimization on multifractal loss landscapes explains a diverse range of geometrical and dynamical properties of deep learning*, which is currently submitted and under revision..

## System requirements:

### Hardware requirements:
`GD-multifractal-landscape` requires only a standard computer with enough RAM to support the in-memory operations.

### Software requirements:
This software has been tested on `python==3.12.7` with the following dependencies:

```setup
numpy 1.26.4
scipy 1.13.0
matplotlib 3.9.2
fbm 0.3.0
torch 2.2.2
torchvision 0.17.2
plotly 5.24.1
nbformat 5.10.4
tqdm 4.66.5
jax 0.5.0
jaxlib 0.5.0
pymittagleffler 0.1.3
opentsne 1.0.2
```

## Installation guide

To install from github (~1-2 minutes):

```setup
git clone https://github.com/anly2178/GD-multifractal-landscape
cd GD-multifractal-landscape
```

Then, to install dependencies, using conda:

```setup
conda env create -f environment.yaml
```

or using pip:

```setup
pip install -r requirements.txt
```

## Demo and instructions

The code for simulations and plots that reproduce the results of the paper are included in the Jupyter notebook `demo.ipynb`. The code uses source data that is available at https://doi.org/10.5281/zenodo.14997499. The expected run time is approximately 45 minutes, excluding the training of neural networks and any subsequent visualization of their loss landscapes. Each run of network training and visualization using `LL_visualisation.py` may take up to 24 hours with a standard computer. Each run of network training using `train_supplementary.py` may take approximately 1 hour with a standard computer. The examples presented in the demo are instructive of how to use the codes in the repository, and can be easily adapted for further exploration.
