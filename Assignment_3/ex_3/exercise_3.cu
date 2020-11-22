#include <curand_kernel.h>
#include <curand.h>
#include <stdio.h>

#define NUM_PARTICLES 256
#define NUM_ITERATIONS 10000
#define NUM_STREAMS 2

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

__global__ void simulate(Particle* particles, int offset) {
    int id = offset + threadIdx.x + blockIdx.x * blockDim.x;
    Particle* p = &particles[id];

    curandState state;
    curand_init(id, id, 0, &state);

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        p->velocity_x += curand_uniform(&state);
        p->velocity_y += curand_uniform(&state);
        p->velocity_z += curand_uniform(&state);

        p->position_x += p->velocity_x;
        p->position_y += p->velocity_y;
        p->position_z += p->velocity_z;
    }
}

int main()
{
    Particle *particles = new Particle[NUM_PARTICLES];
    Particle *d_particles = new Particle[NUM_PARTICLES];
    cudaMallocHost(&d_particles, sizeof(Particle) * NUM_PARTICLES);

    int streamSize = NUM_PARTICLES / NUM_STREAMS;
    int streamBytes = streamSize * sizeof(Particle);
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = i * streamSize;
        cudaMemcpyAsync(&d_particles[offset], &particles[offset], streamBytes, cudaMemcpyHostToDevice, streams[i]);
        simulate<<<N, TPB>>>(d_particles, offset);
        cudaMemcpyAsync(&particles[offset], &d_particles[offset], streamBytes, cudaMemcpyDeviceToHost, streams[i]);
    }

    cudaDeviceSynchronize();

    for (int i = 0; i < 10; i++) {
        printf("%f %f %f\n", particles[i].position_x, particles[i].position_y, particles[i].position_z);
    }
}