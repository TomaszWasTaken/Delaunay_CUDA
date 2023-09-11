#include "flipping.h"

//#include "gpu_predicates.h"

__device__ bool inCircle(const int a,
                         const int b,
                         const int c,
                         const int d,
                         const float* x,
                         const float* y) {
    float val = 0.0f;

    float e11 = x[a]-x[d];
    float e21 = x[b]-x[d];
    float e31 = x[c]-x[d];

    float e12 = y[a]-y[d];
    float e22 = y[b]-y[d];
    float e32 = y[c]-y[d];

    float e13 = e11*e11 + e12*e12;
    float e23 = e21*e21 + e22*e22;
    float e33 = e31*e31 + e32*e32;

    val = e11*e22*e33 + e12*e23*e31 + e13*e21*e32 -
          e12*e21*e33 - e11*e23*e32 - e13*e22*e31;
    
    return val > 0.0f;
}

__device__ bool vertex_in_triangle(int a, int b, int c, int d) {
    return (d == a) || (d == b) || (d == c);
}

__device__ bool edge_in_triangle(int a, int b, int c, int w, int v) {
    return vertex_in_triangle(a,b,c,w) && vertex_in_triangle(a,b,c,v);
}

__device__ int find_compl_point(int a, int b, int c, int w, int v) {
    if (((a == w) && (b == v)) || ((a == v) && (b == w))) {
        return c;
    }
    if (((b == w) && (c == v)) || ((b == v) && (c == w))) {
        return a;
    }
    if (((c == w) && (a == v)) || ((c == v) && (a == w))) {
        return b;
    }
    return -1;
}

__global__ void flip(const float* x,
                     const float* y,
                     const int n,
                     const int m,
                     int* v0,
                     int* v1,
                     int* v2,
                     int* n0,
                     int* n1,
                     int* n2,
                     int* active_triangles,
                     int* flip_vec
                     ) {
    //unsigned int tid = threadIdx.x;
    unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
    //if ((tid < m) && (active_triangles[tid])) {
    if ((tid < m)) {
        /* Vertices of current triangle */
        int a = v0[tid];
        int b = v1[tid];
        int c = v2[tid];

        /* Take care of a-b neighbour */
        int d = -1;
        int e = -1;
        int f = -1;

        if (n0[tid] != -1) {
            int a0 = v0[n0[tid]];
            int b0 = v1[n0[tid]];
            int c0 = v2[n0[tid]];

            d = find_compl_point(a0,b0,c0,a,b);
            if (d != -1) {
            //if (active_triangles[n0[tid]]) {
            //    if (tid < n0[tid]) {
            //        if (inCircle(a,b,c,d,x,y)) {
            //            //printf("tid: %d, n0[tid] (active): %d, a: %d, b: %d, c: %d, d: %d, not Delaunay.\n", tid, n0[tid], a, b, c, d);
            //            atomicMax(flip_vec+tid, n0[tid]);
            //        }
            //    }
            //}
            //else {
                if (inCircle(a,b,c,d,x,y)) {
                    //printf("tid: %d, n0[tid] (not active): %d, a: %d, b: %d, c: %d, d: %d, not Delaunay.\n", tid, n0[tid], a, b, c, d);
                    atomicMax(flip_vec+tid, n0[tid]);
                }
            }
            //}
        }

        if (n1[tid] != -1) {
            int a1 = v0[n1[tid]];
            int b1 = v1[n1[tid]];
            int c1 = v2[n1[tid]];

            e = find_compl_point(a1,b1,c1,b,c);
            if (e != -1) {
            //if (active_triangles[n1[tid]]) {
            //    if (tid < n1[tid]) {
            //        if (inCircle(a,b,c,e,x,y)) {
            //         //   printf("tid: %d, n1[tid] (active): %d, a: %d, b: %d, c: %d, e: %d, not Delaunay.\n", tid, n1[tid], a, b, c, e);
            //            atomicMax(flip_vec+tid, n1[tid]);
            //        }
            //    }
            //}
            //else {
                if (inCircle(a,b,c,e,x,y)) {
                   // printf("tid: %d, n1[tid] (not active): %d, a: %d, b: %d, c: %d, e: %d, Delaunay.\n", tid, n1[tid], a, b, c, e);
                    atomicMax(flip_vec+tid, n1[tid]);
                }
            }
            //}
        }

        if (n2[tid] != -1) {
            int a2 = v0[n2[tid]];
            int b2 = v1[n2[tid]];
            int c2 = v2[n2[tid]];

            f = find_compl_point(a2,b2,c2,c,a);

            //if (active_triangles[n2[tid]]) {
            //    if (tid < n2[tid]) {
            //        if (inCircle(a,b,c,f,x,y)) {
            //     //       printf("tid: %d, n2[tid] (active): %d, a: %d, b: %d, c: %d, f: %d, Delaunay.\n", tid, n2[tid], a, b, c, f);
            //            atomicMax(flip_vec+tid, n2[tid]);
            //        }
            //    }
            //}
            //else {
            if (f != -1) {
                if (inCircle(a,b,c,f,x,y)) {
                 //   printf("tid: %d, n2[tid] (not active): %d, a: %d, b: %d, c: %d, f: %d, Delaunay.\n", tid, n2[tid], a, b, c, f);
                    atomicMax(flip_vec+tid, n2[tid]);
                }
            }
            //}
        }
    }
}

__forceinline__ __device__ int find_neigh(int a, int b, int c, int e0, int e1) {
    if ((a == e0) && (b == e1)) {
        return 0;
    }
    if ((b == e0) && (c == e1)) {
        return 1;
    }
    if ((c == e0) && (a == e1)) {
        return 2;
    }
    return -1;
}


__forceinline__ __device__ float sign(float x1, float y1, float x2, float y2, float x3, float y3) {
    float res = (x1-x3)*(y2-y3);
    res -= (x2-x3)*(y1-y3);
    return res;
}

__forceinline__ __device__ bool point_in_tri(float x, float y, float v0x, float v0y, float v1x, float v1y, float v2x, float v2y) {
    float d1, d2, d3;
    bool has_neg, has_pos;
     
    d1 = sign(x,y,v0x,v0y,v1x,v1y);
    d2 = sign(x,y,v1x,v1y,v2x,v2y);
    d3 = sign(x,y,v2x,v2y,v0x,v0y);

    has_neg = (d1 < 0.0f) || (d2 < 0.0f) || (d3 < 0.0f);
    has_pos = (d1 > 0.0f) || (d2 > 0.0f) || (d3 > 0.0f);

    return !(has_neg && has_pos);
}


__global__ void update_corr_triangles_bis_(const float* x,
                                          const float* y,
                                          const int n,
                                          const int m,
                                          const int* v0,
                                          const int* v1,
                                          const int* v2,
                                          const int* was_inserted,
                                          const int* has_insertion,
                                          int* corr_triangle) {
    unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if (tid < m) {
            float ax = x[v0[tid]];
            float ay = y[v0[tid]];
        
            float bx = x[v1[tid]];
            float by = y[v1[tid]];

            float cx = x[v2[tid]];
            float cy = y[v2[tid]];

            for (int i = 0; i < n; i++) {
                if (was_inserted[i] == 0) {
                    float px = x[i];
                    float py = y[i];

                    if (point_in_tri(px, py, ax, ay, bx, by, cx, cy)) {
                        corr_triangle[i] = tid;
                    }
                }
            }
        }
}

__device__ int which_neighbour(int a0, int b0, int c0, int a1, int b1, int c1) {
    int result = -1;
    if ((a0 == b1) && (b0 == a1)) {
        result = 0;
    }
    if ((a0 == c1) && (b0 == b1)) {
        result = 0;
    }
    if ((a0 == a1) && (b0 == c1)) {
        result = 0;
    }
    // n1 from t0 
    if ((b0 == b1) && (c0 == a1)) {
        result = 1;
    }
    if ((b0 == c1) && (c0 == b1)) {
        result = 1;
    }
    if ((b0 == a1) && (c0 == c1)) {
        result = 1;
    }
    // n2 from t0 
    if ((c0 == b1) && (a0 == a1)) {
        result = 2;
    }
    if ((c0 == c1) && (a0 == b1)) {
        result = 2;
    }
    if ((c0 == a1) && (a0 == c1)) {
        result = 2;
    }

    return result;
}

__global__ void performFlips(int* v0,
                             int* v1,
                             int* v2,
                             int* n0,
                             int* n1,
                             int* n2,
                             int m,
                             int* flip_vec,
                             int* was_changed) {
    //unsigned int tid = threadIdx.x;
    unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if (tid < m) {
        if (flip_vec[tid] != -1) {
            int t0 = tid;  // 1st triangle from flip
            int t1 = flip_vec[tid]; // 2nd triangle from flip

            int a0 = v0[t0];
            int b0 = v1[t0];
            int c0 = v2[t0];

            int a1 = v0[t1];
            int b1 = v1[t1];
            int c1 = v2[t1];

            // n0 from t0 
            if ((a0 == b1) && (b0 == a1)) {
                v1[t0] = c1;
                v1[t1] = c0;
            }
            if ((a0 == c1) && (b0 == b1)) {
                v1[t0] = a1;
                v2[t1] = c0;
            }
            if ((a0 == a1) && (b0 == c1)) {
                v1[t0] = b1;
                v0[t1] = c0;
            }
            // n1 from t0 
            if ((b0 == b1) && (c0 == a1)) {
                v2[t0] = c1; 
                v1[t1] = a0;
            }
            if ((b0 == c1) && (c0 == b1)) {
                v2[t0] = a1;
                v2[t1] = a0;
            }
            if ((b0 == a1) && (c0 == c1)) {
                v2[t0] = b1; 
                v0[t1] = a0;
            }
            // n2 from t0 
            if ((a0 == a1) && (c0 == b1)) {
                v2[t0] = c1;
                v0[t1] = b0;
            }
            if ((c0 == c1) && (a0 == b1)) {
                v2[t0] = a1;
                v1[t1] = b0;
            }
            if ((c0 == a1) && (a0 == c1)) {
                v2[t0] = b1;
                v2[t1] = b0;
            }

            was_changed[t0] = 1;
            was_changed[t1] = 1;
        }
    }
}

__global__ void redouble_flips(int* flip_vec, int m) {
    //unsigned int tid = threadIdx.x;
    unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if (tid < m) {
        if (flip_vec[tid] != -1) {
            flip_vec[flip_vec[tid]] = tid;
        }
    }
}

__device__ void set_neigh(int tid, int* n0, int* n1, int* n2, int val, int index) {
    if (index == 0) {
        n0[tid] = val;
    }
    if (index == 1) {
        n1[tid] = val;
    }
    if (index == 2) {
        n2[tid] = val;
    }
}
__global__ void updateNeighsAfterFlips(int* v0,
                                       int* v1,
                                       int* v2,
                                       int* n0,
                                       int* n1,
                                       int* n2,
                                       int* n0_old,
                                       int* n1_old,
                                       int* n2_old,
                                       int m,
                                       int* flip_vec,
                                       int* was_changed) {
    //unsigned int tid = threadIdx.x;
    unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if (tid < m) {
        int neighs[12] = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};

        if (n0_old[tid] != -1) {
            neighs[0] = n0_old[tid];
            if ((was_changed[neighs[0]]) && (flip_vec[neighs[0]] != tid)) {
                neighs[6] = flip_vec[neighs[0]];
            }
        }
        if (n1_old[tid] != -1) {
            neighs[1] = n1_old[tid];
            if ((was_changed[neighs[1]]) && (flip_vec[neighs[1]] != tid)) {
                neighs[7] = flip_vec[neighs[1]];
            }
        }
        if (n2_old[tid] != -1) {
            neighs[2] = n2_old[tid];
            if ((was_changed[neighs[2]]) && (flip_vec[neighs[2]] != tid)) {
                neighs[8] = flip_vec[neighs[2]];
            }
        }
        int t1 = flip_vec[tid];
        if (t1 != -1) {
            if (n0_old[t1] != -1) {
                neighs[3] = n0_old[t1];
                if ((was_changed[neighs[3]]) && (flip_vec[neighs[3]] != t1)) {
                    neighs[9] = flip_vec[neighs[3]];
                }
            }
            if (n1_old[t1] != -1) {
                neighs[4] = n1_old[t1];
                if ((was_changed[neighs[4]]) && (flip_vec[neighs[4]] != t1)) {
                    neighs[10] = flip_vec[neighs[4]];
                }
            }
            if (n2_old[t1] != -1) {
                neighs[5] = n2_old[t1];
                if ((was_changed[neighs[5]]) && (flip_vec[neighs[5]] != t1)) {
                    neighs[11] = flip_vec[neighs[5]];
                }
            }


        }

        for (int i = 0; i < 12; i++) {
            int index = neighs[i];
            if (index != -1) {
                int neigh_index = which_neighbour(v0[tid], v1[tid], v2[tid], v0[index], v1[index], v2[index]);
                if (neigh_index != -1) {
                    set_neigh(tid, n0, n1, n2, index, neigh_index); 
                }
            }
        }
    }
}

__global__ void remove_duplicates(int* arr, int m) {
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if (tid < m) {
        int curr = arr[tid];
        for (int i = tid+1; i < m; i++) {
            if ((i != tid) && (arr[i] == curr)) {
                arr[i] = -1;
            }
        }
    }
}

__global__ void remove_duplicates_bis(int* arr, int m) {
    unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if (tid < m) {
       for (int i = 0; i < m; i++) {
           if (arr[i] != -1) {
                for (int j = 0; j < m; j++) {
                    if ((arr[j] == i) && (i != j)) {
                        arr[j] = -1;
                    }
                }
            }
       }
    }
}

__global__ void flip_active_tri(const int* flip_vec,
                                int* active_triangles,
                                const int m) {
    unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if (tid < m) {
        int index = flip_vec[tid];
        if (index != -1) {
            active_triangles[tid] = 1;
            active_triangles[index] = 1;
        }
    }
}
