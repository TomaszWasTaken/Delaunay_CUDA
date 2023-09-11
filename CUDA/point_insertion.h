#ifndef POINT_INSERTION_H_
#define POINT_INSERTION_H_

__global__ void compute_centers(const float* x,
                                const float* y,
                                const int n,
                                const int* v0,
                                const int* v1,
                                const int* v2,
                                const int m,
                                float* x_c,
                                float* y_c);

__global__ void pick_winner_points_bis_a(const float* x,
                                         const float* y,
                                         const int* corr_triangle,
                                         const int n,
                                         const float* x_c,
                                         const float* y_c,
                                         const int m,
                                         float* arr);

__global__ void pick_winner_points_bis_b(const float* x,
                                         const float* y,
                                         const int* corr_triangle,
                                         const int n,
                                         const float* x_c,
                                         const float* y_c,
                                         const int m,
                                         const float* arr,
                                         int* winner_points);

__global__ void insert_points_bis(const int* winner_points,
                                  int* v0,
                                  int* v1,
                                  int* v2,
                                  int* n0,
                                  int* n1,
                                  int* n2,
                                  int* corr_triangle,
                                  int* has_insertion,
                                  const int m,
                                  int* active_triangles,
                                  const int* offsets);

__global__ void update_corr_triangles_bis_bis(const float* x,
                                              const float* y,
                                              const float* x_loc,
                                              const float* y_loc,
                                              const int n_insert,
                                              const int m,
                                              const int* v0,
                                              const int* v1,
                                              const int* v2,
                                              const int* loc_to_glob_index,
                                              const int* has_insertion,
                                              int* corr_triangle);
__global__ void update_corr_triangles_bis_bis_(const float* x,
                                              const float* y,
                                              const float* x_loc,
                                              const float* y_loc,
                                              const int n_insert,
                                              const int m_insert,
                                              const int* v0,
                                              const int* v1,
                                              const int* v2,
                                              const int* loc_to_glob_index,
                                              const int* tri_loc_to_glob_index,
                                              int* corr_triangle);

__global__ void copy_triangles_with_ins(const int* has_insertion,
                                        const int* offsets,
                                        const int* v0,
                                        const int* v1,
                                        const int* v2,
                                        const int m,
                                        int* v0_loc,
                                        int* v1_loc,
                                        int* v2_loc,
                                        int* tri_loc_to_glob_index);

__global__ void copy_uninserted_points(const int* was_inserted,
                                       const int* offsets,
                                       const float* x,
                                       const float* y,
                                       const int n,
                                       float* x_loc,
                                       float* y_loc,
                                       int* loc_to_glob_index);
__global__ void update_was_inserted(int* was_inserted,
                                    int* winner_points,
                                    int* corr_triangle,
                                    int* has_insertion,
                                    int n,
                                    int m);
#endif /* POINT_INSERTION_H_ */
