# s2let-denoising-demo

This work aims to recreate figure 5 in https://arxiv.org/abs/1211.1680.

[![Tests](https://github.com/paddyroddy/s2let-denoising-demo/actions/workflows/deploy.yml/badge.svg)](https://github.com/paddyroddy/s2let-denoising-demo/actions/workflows/deploy.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

## Installation

Clone the repo and install dependencies
```
git clone https://github.com/paddyroddy/s2let-denoising-demo.git
pip install -r requirements.txt
```

Then to install the package
```
pip install .
```

To check installation and see possible inputs run
```
demo -h
```

By default, running `demo` is like running
```
 demo -L 128 -j 0 -n 10 -s 3 -t real
```

## pre-commit

A series of hooks can be performed before committing with this simple step. This step is individual, however, and hence it is up to the user whether they want the strict checking. Execute `pip install -r requirements.txt; pre-commit install` to install git hooks in your `.git/` directory. Running `pre-commit run --all-files` will run over the whole repository.
