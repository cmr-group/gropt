<p align="left">
  <a href="https://github.com/mloecher/gropt/">
    <img src="docs/gropt_logo.png" height="110">
  </a>
</p>

A toolbox for MRI Gradient Optimization (GrOpt)

## 1. About
* Fast numerical optimizations for MR gradient waveform design (typically 1-100ms).
* Core libraries built completely in C/C++ for native integration in pulse sequences.
* Python and Matlab wrappers for easy prototyping.
* Flexible constraint system to enable a range of applications.
* Constraints applied in a modular fashion, so adding additional ones is relatively straightforward.

## 2. Contents
- [1. About](#1-about)
- [2. Contents](#2-contents)
- [3. Updates](#3-updates)
- [4. Installation](#4-installation)
  - [4.1. Python](#41-python)
  - [4.2. Matlab](#42-matlab)
- [5. Demos](#5-demos)
  - [5.1. Binder](#51-binder)
  - [5.2. Python](#52-python)
  - [5.3. Matlab](#53-matlab)
- [6. Literature](#6-literature)
- [7. Documentation](#7-documentation)


## 3. Updates

 * OpenMP version of TE finder is better implemented. See demo in the `test_TE_finder` function of src/optimize_kernel.c, which will call multiple GrOpt evals simultaneously to find the shortest feasible T.
 * AR-SDMM solver is in its own branch (arsdmm), currently merging 
 * Added minTE_finder in src/optimize_kernel.c (minTE_diff function) to more efficiently fine the minimum TE
 * Added simultaneous axis optimization, controlled with `Naxis` argument to optimize calls


## 4. Installation

The optimization is written in C and can be found in the src/ directory.

A very basic idea of how to compile for C is included in src/make.txt, however modifications would need to be made for input and output as it only runs a test case when run in C.  For easier usage use one of the wrappers:

### 4.1. Python

The Python module has been tested primarily with [Anaconda](https://www.anaconda.com/) and Python 3.7, though it should work with any type of Python environment.

The setup.py file will build the python module.  To build seperately you can run 
```bash
python setup.py build_ext --inplace
```
from within the python/ directory.  

This will use Cython to generate the source files for a Python module, and then compile it within the GrOpt folder.  For MacOS this procedure requires Xcode.  For Windows you may need a Visual Studio compiler (the free 2019 community version works just fine).  Some common binaries are included in the repository, which should work without any compilation for most.

### 4.2. Matlab

Assuming you have mex setup correctly (check with `mex -setup`), the three main functions can be compiled by running the `make.m` script. 

## 5. Demos

Example usage cases are provided for the Python and Matlab wrappers.  Examples for C applications are shown at the bottom of `src/optimize_kernel.c`

### 5.1. Binder

To run the examples shown in our NMR Biomedicine manuscript entitled "Optimization methods for magnetic resonance imaging gradient waveform design." Click the "Launch Binder" button below and navigate to the Python folder.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cmr-group/gropt/nmrb_optimization)

### 5.2. Python

Demos for Python are all in the form of Jupyter notebooks (.ipynb files) in the ./python/ folder.  Running `jupyter notebook` in the folder will get you started.  Examples show diffusion and non-diffusion gradient design, and most combinations of constraints.

### 5.3. Matlab

Demo Matlab scripts start with demo_*, are in the ./matlab/ folder, and can be run as is to see some example usage cases.


## 6. Manuscripts

There is a paper in MRM discussing the computational aspects of GrOpt:

Middione, MJ, Loecher, M, Ennis, DB. "A Gradient Optimization (GrOpt) Toolbox for General Purpose Time-Optimal MRI Gradient Waveform Design." *Magnetic Resonance in Medicine.* 2020 (Accepted, not Published)


There is also a review paper that discusses optimization in general:

Middione, MJ, Loecher, M, Moulin, K, Ennis, DB. "Optimization methods for magnetic resonance imaging gradient waveform design." *NMR in Biomedicine.* 2020.  https://doi.org/10.1002/nbm.4308

The specific demos and codebased used in this paper can be found in the `nmrb_optimization` branch. [Link](https://github.com/cmr-group/gropt/tree/nmrb_optimization).

## 7. Documentation

Further documentation, including descriptions of all constraints and their arguments and units, can be found at http://gropt.readthedocs.io
