# Thesis Gan

<p align="center">
    <a href="https://github.com/mikcnt/thesis-gan/actions/workflows/test_suite.yml"><img alt="CI" src=https://img.shields.io/github/workflow/status/mikcnt/thesis-gan/Test%20Suite/main?label=main%20checks></a>
    <a href="https://mikcnt.github.io/thesis-gan"><img alt="Docs" src=https://img.shields.io/github/deployments/mikcnt/thesis-gan/github-pages?label=docs></a>
    <a href="https://github.com/grok-ai/nn-template"><img alt="NN Template" src="https://shields.io/badge/nn--template-0.0.2-emerald?style=flat&labelColor=gray"></a>
    <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.9-blue.svg"></a>
    <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

A new awesome project.


## Installation

```bash
pip install git+ssh://git@github.com/mikcnt/thesis-gan.git
```


## Quickstart

[comment]: <> (> Fill me!)


## Development installation

Setup the development environment:

```bash
git clone git+ssh://git@grok-ai/mikcnt/thesis-gan.git
conda env create -f env.yaml
conda activate thesis-gan
pre-commit install
```

Run the tests:

```bash
pre-commit run --all-files
pytest -v
```


### Update the dependencies

Re-install the project in edit mode:

```bash
pip install -e .[dev]
```
