#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda.h>

struct Body {
    float3 pos;
    float3 vel;
    float mass;
};

__device__ inline float3 make_float3_from(const float &x, const float &y, const float &z){
    return make_float3(x,y,z);
}

__global__ void nbody_step(Body *bodies, Body *bodies_out, int N, float dt, float softening) {
    extern __shared__ Body sdata[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    
    float3 myPos = make_float3(0.f,0.f,0.f);
    float3 myVel = make_float3(0.f,0.f,0.f);
    float myMass = 0.f;
    if (gid < N) {
        myPos = bodies[gid].pos;
        myVel = bodies[gid].vel;
        myMass = bodies[gid].mass;
    }

    float3 acc = make_float3(0.f, 0.f, 0.f);

    
    int numTiles = (N + blockDim.x - 1) / blockDim.x;
    for (int tile = 0; tile < numTiles; ++tile) {
        int idx = tile * blockDim.x + tid;
        if (idx < N) {
            sdata[tid] = bodies[idx];
        } else {
            
            sdata[tid].pos = make_float3(0.f,0.f,0.f);
            sdata[tid].vel = make_float3(0.f,0.f,0.f);
            sdata[tid].mass = 0.f;
        }
        __syncthreads();

        
        #pragma unroll 4
        for (int j = 0; j < blockDim.x; ++j) {
            int global_j = tile * blockDim.x + j;
            if (global_j >= N) break;

            float3 r;
            r.x = sdata[j].pos.x - myPos.x;
            r.y = sdata[j].pos.y - myPos.y;
            r.z = sdata[j].pos.z - myPos.z;

            float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + softening;
            float invDist = rsqrtf(distSqr); 
            float invDist3 = invDist * invDist * invDist;

            
            float s = sdata[j].mass * invDist3;

            acc.x += r.x * s;
            acc.y += r.y * s;
            acc.z += r.z * s;
        }
        __syncthreads();
    }

    
    if (gid < N) {
        if (myMass > 0.f) {
            acc.x /= myMass;
            acc.y /= myMass;
            acc.z /= myMass;
        }

        myVel.x += acc.x * dt;
        myVel.y += acc.y * dt;
        myVel.z += acc.z * dt;

        myPos.x += myVel.x * dt;
        myPos.y += myVel.y * dt;
        myPos.z += myVel.z * dt;

        bodies_out[gid].pos = myPos;
        bodies_out[gid].vel = myVel;
        bodies_out[gid].mass = myMass;
    }
}

