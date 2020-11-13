import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from wurlitzer import pipes

mod = SourceModule("""
    #include <stdio.h>

    __global__ void print_thread_id()
    {
      printf("Hello World! My threadId is %d\\n", threadIdx.x);
    }
""")

func = mod.get_function("print_thread_id")

with pipes() as (out, err):
  func(block=(256, 1, 1))
  cuda.Context.synchronize()

print(out.read())