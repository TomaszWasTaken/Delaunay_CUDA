#include "setup.h"

__global__ void setup(int* corr_triangle,
                      int* was_inserted) {

    corr_triangle[0] = -1;
    corr_triangle[1] = -1;
    corr_triangle[2] = -1;

    was_inserted[0] = 1;
    was_inserted[1] = 1;
    was_inserted[2] = 1;
}

__global__ void set_flag_to_0(int* flag) {
    *flag = 0;
}
__global__ void loop_control(const int* arr, const int n, int* flag) {
    unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if (tid < n) {
        int val = arr[tid]; 
        atomicAdd(flag, val);
    }
}
