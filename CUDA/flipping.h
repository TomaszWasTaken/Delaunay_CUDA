#ifndef FLIPPING_H_
#define FLIPPING_H_

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
                     );

__global__ void update_corr_triangles_bis_(const float* x,
                                          const float* y,
                                          const int n,
                                          const int m,
                                          const int* v0,
                                          const int* v1,
                                          const int* v2,
                                          const int* was_inserted,
                                          const int* has_insertion,
                                          int* corr_triangle);

__global__ void performFlips(int* v0,
                             int* v1,
                             int* v2,
                             int* n0,
                             int* n1,
                             int* n2,
                             int m,
                             int* flip_vec,
                             int* was_changed);

__global__ void redouble_flips(int* flip_vec, int m);

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
                                       int* was_changed);

__global__ void remove_duplicates(int* arr, int m);

__global__ void remove_duplicates_bis(int* arr, int m);

__global__ void flip_active_tri(const int* flip_vec,
                                int* active_triangles,
                                const int m);
#endif /* FLIPPING_H_ */
