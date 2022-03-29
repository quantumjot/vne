[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://github.com/quantumjot/vne/actions/workflows/test.yml/badge.svg)](quantumjot/vne/actions)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

# vne

von Neumman's Elephant

"With four parameters I can fit an elephant, and with five I can make him wiggle his trunk."

### Installation

```sh
conda create -n vne python=3.8
conda activate vne
git clone https://github.com/quantumjot/vne.git
cd vne
```

#### Inference only
```sh
pip install -e .
```

#### Training/Development
```sh
pip install -e ".[dev]"
```
