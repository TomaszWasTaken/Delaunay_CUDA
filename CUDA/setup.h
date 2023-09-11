#ifndef SETUP_H_
#define SETUP_H_


__global__ void setup(int* corr_triangle,
                      int* was_inserted);

__global__ void set_flag_to_0(int* flag);

__global__ void loop_control(const int* arr, const int n, int* flag);
#endif /* SETUP_H_ */
