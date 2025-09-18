# CUDA N-Body Simulation

This project implements a simple **N-Body simulation** in CUDA.  

It models the gravitational interactions of `N` bodies in 2D space.

The code demonstrates:
- Writing and launching a **CUDA kernel**
- Implementing a basic **semi-implicit Euler integrator**
- Profiling and optimizing GPU workloads

## Features
- CUDA kernel for all-pairs force computation  
- Shared memory tiling for improved performance  
- Simple host code to initialize bodies and run the simulation  

## Requirements
- CUDA Toolkit (tested with CUDA 13)
- `make` and `g++`/`nvcc`
