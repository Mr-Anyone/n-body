# CUDA N-Body Simulation

This project implements a simple **N-Body simulation** in CUDA.  

It models the gravitational interactions of `N` bodies in 3D space.

The code demonstrates:
- Writing and launching a **CUDA kernel**
- Using **shared memory tiling** to reduce global memory traffic
- Implementing a basic **semi-implicit Euler integrator**
- Profiling and optimizing GPU workloads

---
## Features
- CUDA kernel for all-pairs force computation  
- Shared memory tiling for improved performance  
- Adjustable parameters: time step, number of steps, softening factor  
- Simple host code to initialize bodies and run the simulation  
- `Makefile` included for easy compilation  


## Requirements
- CUDA-capable NVIDIA GPU (Compute Capability ≥ 6.0 recommended)
- CUDA Toolkit (tested with CUDA 11+)
- `make` and `g++`/`nvcc`
