#include "integrate.h"

__global__ 
void integrate_impl(Body* next_state, Body* current_state, int size,
        float time_delta){
    int current_index = blockDim.x*blockIdx.x+ threadIdx.x;
    const Body& this_body = current_state[current_index];

    // the x component and y component of a vector
    Vec2 net_force (0.0, 0.0);
    for(int i = 0; i<size; ++i){
        if(i == current_index)
            continue;

        const Body& other_body = current_state[i];
        float distance = (other_body.pos - this_body.pos).length();
            // G * m1 * m2 / (distance)^3
        float constant_in_front =
            GRAVITATIONAL_CONSTANT *(other_body.mass * this_body.mass) 
                / (distance * distance * distance);
        Vec2 current_force = (other_body.pos - this_body.pos) * constant_in_front;
        net_force = net_force + current_force;
    }

    /// update acceleration 
    next_state[current_index].acc = net_force*(1/this_body.mass);
    next_state[current_index].vel = current_state[current_index].vel + current_state[current_index].acc * time_delta;
    next_state[current_index].pos = current_state[current_index].pos + current_state[current_index].vel * time_delta;
}

// FIXME: next_state, and current_state are from host, 
// we need to perform a memcpy 
void integrate(Body* next_state, Body* current_state, int count, 
        float time_delta){
    Body* next_state_cuda;
    Body* current_state_cuda;
    cudaMalloc(&next_state_cuda, count * sizeof(Body));
    cudaMemcpy(next_state_cuda, next_state,
            sizeof(Body)*count, cudaMemcpyHostToDevice);

    cudaMalloc(&current_state_cuda, count * sizeof(Body));
    cudaMemcpy(current_state_cuda, current_state,
            sizeof(Body)*count, cudaMemcpyHostToDevice);

    // FIXME: we are taking the floor, so there are some objects 
    // that aren't being executed into the buffer
    int block_number =  count / 512;
    // one block and 512 threads, change this later to respect size
    integrate_impl<<<block_number, 512>>>(next_state, current_state, count, time_delta);
    cudaDeviceSynchronize();

    cudaMemcpy(next_state, next_state_cuda,
            sizeof(Body)*count, cudaMemcpyDeviceToHost);
    // FIXME: I don't think we need to copy current state into host device
    cudaMemcpy(current_state, current_state_cuda,
            sizeof(Body)*count, cudaMemcpyDeviceToHost);
    cudaFree(current_state_cuda);
    cudaFree(next_state_cuda);
}

