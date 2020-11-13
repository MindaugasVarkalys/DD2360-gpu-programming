import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy
import time
import matplotlib.pyplot as plt

ARRAY_SIZES = [1000,10000,100000,1000000,10000000,100000000]
BLOCK_SIZE = 512

execution_times = []
for ARRAY_SIZE in ARRAY_SIZES:

  GRID = (ARRAY_SIZE // BLOCK_SIZE if ARRAY_SIZE % BLOCK_SIZE == 0 else ARRAY_SIZE // BLOCK_SIZE + 1)
  
  a = numpy.float32(numpy.random.rand())
  x = numpy.random.randn(ARRAY_SIZE).astype(numpy.float32)
  y = numpy.random.randn(ARRAY_SIZE).astype(numpy.float32)
  result_GPU = numpy.random.randn(ARRAY_SIZE).astype(numpy.float32)

  mod = SourceModule("""
    __global__ void saxpy(float *x, float *y, float a, float *result)
    {
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      result[idx] = a * x[idx] + y[idx];
    }
    """)

  # Run on GPU with timing
  func = mod.get_function("saxpy")
  time_GPU_pre = time.time()
  func(cuda.In(x), cuda.In(y), a, cuda.Out(result_GPU), block=(BLOCK_SIZE, 1, 1), grid=(GRID, 1))
  time_GPU_post = time.time()
  delta_time_GPU = time_GPU_post - time_GPU_pre

  time_CPU_pre = time.time()
  result_CPU = a * x + y
  time_CPU_post = time.time()
  delta_time_CPU = time_CPU_post - time_CPU_pre

  execution_times.append([delta_time_GPU,delta_time_CPU])

  print(f"GRID: {GRID}")
  if (numpy.allclose(result_CPU, result_GPU, 0.1)):
    print("Results are equal!")
  else:
    print("Results are NOT equal!")

for exec_time in execution_times:
  print(f"GPU time / CPU time: {exec_time[0] / exec_time[1]}")


gpu_plot_data = []
cpu_plot_data = []
for i in range(len(ARRAY_SIZES)):
  gpu_plot_data.append(execution_times[i][0])
  cpu_plot_data.append(execution_times[i][1])

fig, ax = plt.subplots()
ax.plot(gpu_plot_data, label="GPU")
ax.plot(cpu_plot_data, label="CPU")
ax.set_xticklabels(ARRAY_SIZES)

ax.legend()

ax.set(xlabel='Array size', ylabel='Execution time (s)',
       title='Execution times')
ax.grid()

plt.show()