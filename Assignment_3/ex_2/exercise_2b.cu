#include <curand_kernel.h>
#include <curand.h>
#include <stdio.h>

#define NUM_PARTICLES 10000
#define NUM_ITERATIONS 10000

#define TPB 256
#define N (NUM_PARTICLES/TPB + 1)

struct Particle {
    float position_x;
    float position_y;
    float position_z;

    float velocity_x;
    float velocity_y;
    float velocity_z;
};

__global__ void simulate(Particle* particles) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    Particle* p = &particles[id];

    curandState state;
    curand_init(id, id, 0, &state);
    p->velocity_x += curand_uniform(&state);
    p->velocity_y += curand_uniform(&state);
    p->velocity_z += curand_uniform(&state);

    p->position_x += p->velocity_x;
    p->position_y += p->velocity_y;
    p->position_z += p->velocity_z;
}

int main()
{
    Particle *particles;

    cudaError_t code = cudaMallocManaged(&particles, sizeof(Particle) * NUM_PARTICLES);
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        simulate<<<N, TPB>>>(particles);
        cudaDeviceSynchronize();
    }
    for (int i = 0; i < 10; i++) {
        printf("%f %f %f\n", particles[i].position_x, particles[i].position_y, particles[i].position_z);
    }
}