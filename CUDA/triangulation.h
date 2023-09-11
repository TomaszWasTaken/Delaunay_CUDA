#ifndef TRIANGULATION_H_
#define TRIANGULATION_H_

#include <iterator>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <iomanip>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <thrust/logical.h>
#include <thrust/functional.h>

#include "point_insertion.h"
#include "setup.h"
#include "flipping.h"

void triangulate(thrust::host_vector<float>& x_cpu, thrust::host_vector<float>& y_cpu);


#endif /* TRIANGULATION_H_ */
