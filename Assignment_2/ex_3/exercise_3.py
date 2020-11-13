import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy
import time
import pycuda.driver as driver

NUM_PARTICLES = 10000
BLOCK_SIZE = 256
GRID_SIZE = (NUM_PARTICLES // BLOCK_SIZE if NUM_PARTICLES % BLOCK_SIZE == 0 else NUM_PARTICLES // BLOCK_SIZE + 1)
NUM_ITERATIONS = 10000

mod = SourceModule("""
  #include <curand_kernel.h>
  #include <curand.h>

  struct Particle {
    float position_x;
    float position_y;
    float position_z;

    float velocity_x;
    float velocity_y;
    float velocity_z;
  };

  extern "C" {
    __global__ void simulate(Particle* particles, int iterations) {
      int id = threadIdx.x + blockIdx.x * blockDim.x;
      Particle* p = &particles[id];

      curandState state;
      curand_init(id, id, 0, &state);

      for (int i = 0; i < iterations; i++) {
        p->velocity_x += curand_uniform(&state);
        p->velocity_y += curand_uniform(&state);
        p->velocity_z += curand_uniform(&state);

        p->position_x += p->velocity_x;
        p->position_y += p->velocity_y;
        p->position_z += p->velocity_z;
      }
    }
  }
  """, no_extern_c=True)


class ParticleStruct:
    mem_size = 6 * numpy.float32(0).nbytes
    def __init__(self, position, velocity, pointer):
      self.pointer = pointer
      cuda.memcpy_htod(pointer, position)
      cuda.memcpy_htod(pointer + numpy.float32(0).nbytes * (3), velocity)

    def __str__(self):
      return ("p: " + str(cuda.from_device(self.pointer, (3), numpy.float32)) + ", " +
                "v: " + str(cuda.from_device(self.pointer + numpy.float32(0).nbytes * (3), (3), numpy.float32)))


def simulate_gpu():
  driver.start_profiler()
  particle_pointer = cuda.mem_alloc(NUM_PARTICLES * ParticleStruct.mem_size)

  particles = []
  for i in range(NUM_PARTICLES):
    p = ParticleStruct(numpy.random.randn(3).astype(numpy.float32),
                        numpy.random.randn(3).astype(numpy.float32),
                        int(particle_pointer) + i * ParticleStruct.mem_size)
    particles.append(p)

  print("GPU initial:", list(map(lambda p: str(p), particles)))
  func = mod.get_function("simulate")
  func(particle_pointer, numpy.int32(NUM_ITERATIONS), grid=(GRID_SIZE,1), block=(BLOCK_SIZE,1,1))
  print("GPU result:", list(map(lambda p: str(p), particles)))
  driver.stop_profiler()


def simulate_cpu():
  velocity = numpy.random.randn(NUM_PARTICLES, 3)
  position = numpy.random.randn(NUM_PARTICLES, 3)

  print("CPU initial v:", velocity)
  print("CPU initial p:", position)

  for i in range(NUM_ITERATIONS):
    rand = numpy.random.randn(NUM_PARTICLES, 3)
    velocity += rand
    position += velocity

  print("CPU result v:", velocity)
  print("CPU result p:", position)


gpu_time_pre = time.time()
simulate_gpu()
gpu_execution_time = time.time() - gpu_time_pre

cpu_time_pre = time.time()
simulate_cpu()
cpu_execution_time = time.time() - gpu_time_pre

print("GPU execution time", gpu_execution_time)
print("CPU execution time", cpu_execution_time)