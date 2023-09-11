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

#include <random>
#include <cstdio>
#include <iostream>

#include <cstdio>

#include <chrono>
#include <time.h>

#include <float.h>

#include "point_insertion.h"
#include "flipping.h"
#include "morton.h"
#include "setup.h"

#include "triangulation.h"

#include <string.h>

int main(int argc, char* argv[]) {
    /* Point data */
    int n;
    if (argc < 2) {
        n = 1000;
    }
    else {
        n = atoi(argv[1]);
    }

    thrust::host_vector<float> x_cpu(n);
    thrust::host_vector<float> y_cpu(n);

    x_cpu[0] = -2.0f;
    y_cpu[0] = 0.0f;
    x_cpu[1] = 3.0f;
    y_cpu[1] = 0.0f;
    x_cpu[2] = 0.5f;
    y_cpu[2] = 3.0f;

    std::mt19937 gen(12345L);
    std::uniform_real_distribution<> dist(0.0f, 1.0f);
    for (int i = 3; i < n; i++) {
        x_cpu[i] = dist(gen);
        y_cpu[i] = dist(gen);
    }

    thrust::host_vector<long long> morton_codes(n);
    for (int i = 0; i < n; i++) {
        morton_codes[i] = mortonIndex(x_cpu[i], y_cpu[i]);
    }

    thrust::sort_by_key(morton_codes.begin()+3, morton_codes.end(), thrust::make_zip_iterator(thrust::make_tuple(x_cpu.begin()+3, y_cpu.begin()+3)));

    triangulate(x_cpu, y_cpu);

    return 0;
}
