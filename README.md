# Generalized Concomitant Lasso

This repository hosts the implementation of solvers for multi-task Generalized Concomitant Lasso

The algorithms are in ```./shcl/*_fast.pyx```.
Currently implemented are:
* multi-task Generalized Concomitant Lasso
* multi-task Block Homoscedastic Concomitant Lasso

For optimal time performance, the algorithms are written in Cython, using calls to BLAS/LAPACK when possible.

# Installation
Clone the repository:

```
$git clone git@github.com:mathurinm/SHCL.git
$cd SHCL/
$conda env create --file environment.yml
$source activate shcl-env
$pip install --no-deps -e .
```

# Dependencies
All dependencies are in  ```./environment.yml```

# Cite
If you use this code, please cite [this paper](https://arxiv.org/abs/1705.09778):

Mathurin Massias, Olivier Fercoq, Alexandre Gramfort and Joseph Salmon

Generalized Concomitant Multi-Task Lasso for sparse multimodal regression

Arxiv preprint, 2017
