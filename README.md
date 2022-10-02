# Interdependency in stock traces: a Generative Adversarial Network approach

<p align="center">
    <a href="https://github.com/grok-ai/nn-template"><img alt="NN Template" src="https://shields.io/badge/nn--template-0.0.2-emerald?style=flat&labelColor=gray"></a>
    <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.9-blue.svg"></a>
    <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

This repository contains the codebase of my thesis work in the MSc in Computer Science at Sapienza University of Rome.

The main contributions of this work are i) the introduction of a novel deep learning architecture to solve the problem of generating multiple interdependent stock traces simultaneously, based on Wasserstein GANs with a revised critic score; ii) the development of a new metric, the *Cross-Correlation Distance*, to keep track of the model's ability to capture the interdependence aspects in the generated timeseries.


## Installation

```bash
pip install git+ssh://git@github.com/mikcnt/thesis-gan.git
```


## Quickstart
The repository is organized as follows:
* Directory `conf` contains the yaml configuration files used through the project to load the data, train and evaluate the models;
* Directory `src` contains the definitions of the datasets and models classes, along with the training pipelines (developed in PyTorch Lightning).
* The datasets for this project should be stored in the `data` directory.

## Development installation

Setup the development environment:

```bash
git clone git+ssh://git@github/mikcnt/thesis-gan.git
conda env create -f env.yaml
conda activate thesis-gan
pre-commit install
```


### Update the dependencies

Re-install the project in edit mode:

```bash
pip install -e .[dev]
```
