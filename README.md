# Canonical RG prescription in tensor space using the GILT-HOTRG
This repository keeps the latex files of the arXiv e-print, [*Scaling dimensions from linearized tensor renormalization group transformations*](https://arxiv.org/abs/2102.08136), and the Python3 codes of the numerical calculations in the e-print, including the implementation of the [graph independent local truncation](https://arxiv.org/abs/1709.07460) (GILT) and the [higer-order tensor renormalization group](https://arxiv.org/abs/1201.1144) (HOTRG).

Apart from standard computational libraries like NumPy and SciPy, the implementation relies heavily on three libraries [tn-tools](https://github.com/mhauru/tntools), [ncon](https://github.com/mhauru/ncon) and [abeliantensors](https://github.com/mhauru/abeliantensors) written by Markus Hauru.

## Requirements
* [Anaconda packages](https://www.anaconda.com/download/), specifically NumPy and SciPy
* [google/jax](https://github.com/google/jax) library which is an extension of numpy to support automatic differentiation
* [tn-tools](https://github.com/mhauru/tntools) and [ncon](https://github.com/mhauru/ncon)
* Installation of [abeliantensors](https://github.com/mhauru/abeliantensors) is not necessary, since I've made small adjustments to this package and copied it here.

## File descriptions
All the Python3 codes of the numerical calulations for the 2D Ising model reported in the e-print are located in the `analysisCodes` directory.
The calculations for the 1D Ising model are located in `misc` directory.
All the remaining things are latex files of the e-print.
We give a detailed explanation of the Python3 codes in the `analysisCodes` directory.

## Inside `analysisCodes` directory
### `abeliantensors` directory
It is the package copied directly from the [abeliantensors](https://github.com/mhauru/abeliantensors) repository, with small adjustments to the `abeliantensors/abeliantensor.py` file to make the tensor RG flow more stable. The adjustments involve the `matrix_eig` function starting from line 1885. We add a boolean argument `evenTrunc` for this function. If `evenTrunc = True`, the bond dimension will be distributed evenly among two different sectors of a Z2 symmetric tensor.

### Functions
* `jncon.py`— It is the jax version of `ncon` function for tensor contractions. Basically, we replace NumPy matrix multiplications with the corresponding jax version. It will be useful when we linearize the RG equation of GILT-HOTRG to generate Eq. (55) in the e-print.
* `Isings.py`— Contain a function, `Ising2dT`, to generate the initial tensor for the 2D Ising model, 

### Scripts for analysis
