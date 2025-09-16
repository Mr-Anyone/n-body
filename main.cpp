#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <iostream>
extern "C" {
    #include "nbody_cuda.cu"
}

int main() {
    const int N = 1<<14;
    const float dt = 0.01f;
    const float softening = 1e-7f;
    const int steps = 10;

    std::vector<Body> h_bodies(N);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < N; ++i) {
        h_bodies[i].pos = make_float3(dist(rng)*100.0f, dist(rng)*100.0f, dist(rng)*100.0f);
        h_bodies[i].vel = make_float3(0.f,0.f,0.f);
        h_bodies[i].mass = 1.0f + dist(rng)*0.5f;
    }

    Body *d_bodies[2];
    size_t bytes = N * sizeof(Body);
    cudaMalloc(&d_bodies[0], bytes);
    cudaMalloc(&d_bodies[1], bytes);

    cudaMemcpy(d_bodies[0], h_bodies.data(), bytes, cudaMemcpyHostToDevice);

    int blockSize = 128;
    int gridSize = (N + blockSize - 1) / blockSize;
    size_t sharedMemBytes = blockSize * sizeof(Body);

    for (int s = 0; s < steps; ++s) {
        nbody_step<<<gridSize, blockSize, sharedMemBytes>>>(d_bodies[0], d_bodies[1], N, dt, softening);
        cudaDeviceSynchronize();
        // swap buffers
        std::swap(d_bodies[0], d_bodies[1]);
    }

    cudaMemcpy(h_bodies.data(), d_bodies[0], bytes, cudaMemcpyDeviceToHost);

    // print first 5 bodies
    cudaFree(d_bodies[0]);
    cudaFree(d_bodies[1]);
    return 0;
}

