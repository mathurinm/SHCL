# Generalized Concomitant Lasso

This repository hosts the implementation of solvers for multi-task Generalized Concomitant Lasso

The algorithms are in ```./shcl/*_fast.pyx```.
Currently implemented are:
* multi-task Generalized Concomitant Lasso (```multitask_generalized_solver()```)
* multi-task Block Homoscedastic Concomitant Lasso (```multitask_blockhomo_solver()```)
* Block Homoscedastic Concomitant Lasso (one task)

For optimal time performance, the algorithms are written in Cython, using calls to BLAS/LAPACK when possible.

# Installation
Clone the repository:

```
$git clone https://github.com/mathurinm/SHCL.git
$cd SHCL/
$conda env create --file environment.yml
$source activate shcl-env
$pip install --no-deps -e .  # do not forget the . at the end
```

# Examples
Once you have created the conda environment with the previous command, you can run the examples with:
```
$source activate shcl-env
$ipython -i examples/blockhomo_example.py
```

# Dependencies
All dependencies are in  ```./environment.yml```

# Cite
If you use this code, please cite [this paper](https://arxiv.org/abs/1705.09778):

Mathurin Massias, Olivier Fercoq, Alexandre Gramfort and Joseph Salmon

Generalized Concomitant Multi-Task Lasso for sparse multimodal regression

to appear in AISTATS 2018
