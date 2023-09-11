#include "point_insertion.h"

#define det(a,b,c,d,e,f,g,h,i) ((a)*(e)*(i) + (b)*(f)*(g) + (c)*(d)*(h) - (c)*(e)*(g) - (b)*(d)*(i) - (a)*(f)*(h))

__global__ void compute_centers(const float* x,
                                const float* y,
                                const int n,
                                const int* v0,
                                const int* v1,
                                const int* v2,
                                const int m,
                                float* x_c,
                                float* y_c) {
    unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
    /* Iterate over the triangles */
    if (tid < m) {
        /* Load the vertex indices into registers */
        const int v0_r = v0[tid];
        const int v1_r = v1[tid];
        const int v2_r = v2[tid];

        /* Load the points */
        const float Ax = x[v0_r];        
        const float Ay = y[v0_r];
        const float Bx = x[v1_r];
        const float By = y[v1_r];
        const float Cx = x[v2_r];
        const float Cy = y[v2_r];

        const float A2 = Ax*Ax + Ay*Ay;
        const float B2 = Bx*Bx + By*By;
        const float C2 = Cx*Cx + Cy*Cy;

        float t1 = By-Cy;
        float t2 = Cy-Ay;
        float t3 = Ay-By;

        float Sx = A2*t1;
        Sx += B2*t2;
        Sx += C2*t3;
        Sx *= 0.5f;
        //((a)*(e)*(i) + (b)*(f)*(g) + (c)*(d)*(h) - (c)*(e)*(g) - (b)*(d)*(i) - (a)*(f)*(h))
        float Sy = 0.5f*det(Ax,A2,1.0f,Bx,B2,1.0f,Cx,C2,1.0f);

        float a = det(Ax,Ay,1.0f,Bx,By,1.0f,Cx,Cy,1.0f);
        
        x_c[tid] = Sx/a;
        y_c[tid] = Sy/a;
    }
}

__device__ __forceinline__ float atomicMinFloat (float * addr, float value) {
        float old;
        old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) :
             __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));

        return old;
}

#define dist_sq(ax,ay,bx,by) ((((ax)-(bx))*((ax)-(bx)))+(((ay)-(by))*((ay)-(by))))

__global__ void pick_winner_points_bis_a(const float* x,
                             const float* y,
                             const int* corr_triangle,
                             const int n,
                             const float* x_c,
                             const float* y_c,
                             const int m,
                             float* arr) {
    unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if (tid < n) {
        // read corresponding triangle
        int tri_id = corr_triangle[tid];
        if (tri_id != -1) {
            //has_insertion[tri_id] = 1;
            // load data to registers 
            float xc = x_c[tri_id];
            float yc = y_c[tri_id];
            float px = x[tid];
            float py = y[tid];
            // every point compute the distance to its circumcenter
            float d = dist_sq(px, py, xc, yc);
//            printf("%d: %d, %f\n", tid, tri_id, d);
            // write the distance to the array
            atomicMinFloat(arr+tri_id, d);
        }
    }
}

__global__ void pick_winner_points_bis_b(const float* x,
                             const float* y,
                             const int* corr_triangle,
                             const int n,
                             const float* x_c,
                             const float* y_c,
                             const int m,
                             const float* arr,
                             int* winner_points) {
    unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if (tid < n) {
        int tri_id = corr_triangle[tid];
        if (tri_id != -1) {
            // load data to registers 
            float xc = x_c[tri_id];
            float yc = y_c[tri_id];
            float px = x[tid];
            float py = y[tid];
            // every point compute the distance to its circumcenter
            float d = dist_sq(px, py, xc, yc);
            // write the distance to the array
            if (d == arr[tri_id]) {
                atomicMin(winner_points+tri_id, tid);
            }
        }
    }
}

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
                              const int* offsets) {
    unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if ((tid < m) && (has_insertion[tid])) {
        int a = v0[tid];
        int b = v1[tid];
        int c = v2[tid];
        int p = winner_points[tid];

        int offset = m;
        //int offset_bis = m;

        int n0_index = -1;
        int n1_index = -1;
        int n2_index = -1;

        //int n_insert = 0;
        int n_insert = offsets[m-1];

        int flag_n0 = 0;
        int flag_n1 = 0;
        int flag_n2 = 0;

        int n0_offset = m; 
        int n1_offset = m;
        int n2_offset = m;

        //int n0_offset_bis = m; 
        //int n1_offset_bis = m;
        //int n2_offset_bis = m;

        if (n0[tid] != -1) {
            flag_n0 = 1;
            //n0_offset = m - n0[tid];
        }

        if (n1[tid] != -1) {
            flag_n1 = 1;
            //n1_offset = m - n1[tid];
        }

        if (n2[tid] != -1) {
            flag_n2 = 1;
            //n2_offset = m - n2[tid];
        }

        if (tid > 0) {
            offset = m + offsets[tid-1] - tid;
        }

        //for (int i = 0; i < m; i++) {
        //    if (has_insertion[i]) {
        //        //n_insert++;
        //    }
        //    else {
        //        if (i < tid) {
        //            //offset--;
        //        }
        //    }
        //}
        if (flag_n2) {
            int a2 = v0[n2[tid]];
            int b2 = v1[n2[tid]];
            int c2 = v2[n2[tid]];

            int index = -1;
            if (((a2 == c) && (b2 == a)) || ((a2 == a) && (b2 == c))) {
                index = 0;
            }
            else if (((b2 == c) && (c2 == a)) || ((b2 == a) && (c2 == c))) {
                index = 1;
            }
            else if (((c2 == c) && (a2 == a)) || ((c2 == a) && (a2 == c))) {
                index = 2;
            }

            n2_index = index;
                
            //for (int i = 0; i < n2[tid]; i++) {
            //    if (has_insertion[i] == 0) {
            //        n2_offset-=1;
            //    }
            //}

            if ((n2[tid] > 0)) {
                //offset = m + offsets[tid-1] - tid;
                //n2_offset_bis = m + offsets[n2[tid]] - n2[tid];
                //n2_offset = m + offsets[n2[tid]] - n2[tid] - 1;
                n2_offset = m - n2[tid] + offsets[n2[tid]-1];
                //printf("n1_offset: %d, offsets[]: %d\n", n1_offset, m + offsets[n1[tid]] - n1[tid]);
            }
        }

        // Check if a-b neighbour
        if (flag_n0) {
            int a0 = v0[n0[tid]];
            int b0 = v1[n0[tid]];
            int c0 = v2[n0[tid]];

            int index = -1;
            if (((a0 == a) && (b0 == b)) || ((a0 == b) && (b0 == a))) {
                index = 0;
            }
            else if (((b0 == a) && (c0 == b)) || ((b0 == b) && (c0 == a))) {
                index = 1;
            }
            else if (((c0 == a) && (a0 == b)) || ((c0 == b) && (a0 == a))) {
                index = 2;
            }

            n0_index = index;

            //for (int i = 0; i < n0[tid]; i++) {
            //    if (has_insertion[i] == 0) {
            //        n0_offset-=1;
            //    }
            //}
            if ((n0[tid] > 0)) {
                //offset = m + offsets[tid-1] - tid;
                //n0_offset_bis = m + offsets[n0[tid]-1] - n0[tid];
                //n0_offset = m + offsets[n0[tid]] - n0[tid] - 1;
                n0_offset = m - n0[tid] + offsets[n0[tid]-1];
                //printf("n1_offset: %d, offsets[]: %d\n", n1_offset, m + offsets[n1[tid]] - n1[tid]);
            }
        } 

        //if (n1[tid] != -1) {
        if (flag_n1) {
            int a1 = v0[n1[tid]];
            int b1 = v1[n1[tid]];
            int c1 = v2[n1[tid]];

            int index = -1;
            if (((a1 == b) && (b1 == c)) || ((a1 == c) && (b1 == b))) {
                index = 0;
            }
            else if (((b1 == b) && (c1 == c)) || ((b1 == c) && (c1 == b))) {
                index = 1;
            }
            else if (((c1 == b) && (a1 == c)) || ((c1 == c) && (a1 == b))) {
                index = 2;
            }

            n1_index = index;

            //for (int i = 0; i < n1[tid]; i++) {
            //    if (has_insertion[i] == 0) {
            //        n1_offset-=1;
            //    }
            //}
            if ((n1[tid] > 0)) {
                //offset = m + offsets[tid-1] - tid;
                //n1_offset_bis = m + offsets[n1[tid]-1] - n1[tid];
                //n1_offset = m + offsets[n1[tid]] - n1[tid] - 1;
                n1_offset = m - n1[tid] + offsets[n1[tid]-1];
                //printf("n1_offset: %d, offsets[]: %d\n", n1_offset, m + offsets[n1[tid]] - n1[tid]);
            }
        }

        // Check if c-a neighbour
        //if (n2[tid] != -1) {
//        if (flag_n2) {
//            int a2 = v0[n2[tid]];
//            int b2 = v1[n2[tid]];
//            int c2 = v2[n2[tid]];
//
//            int index = -1;
//            if (((a2 == c) && (b2 == a)) || ((a2 == a) && (b2 == c))) {
//                index = 0;
//            }
//            else if (((b2 == c) && (c2 == a)) || ((b2 == a) && (c2 == c))) {
//                index = 1;
//            }
//            else if (((c2 == c) && (a2 == a)) || ((c2 == a) && (a2 == c))) {
//                index = 2;
//            }
//
//            n2_index = index;
//                
//            //for (int i = 0; i < n2[tid]; i++) {
//            //    if (has_insertion[i] == 0) {
//            //        n2_offset-=1;
//            //    }
//            //}
//
//            if ((n2[tid] > 0)) {
//                //offset = m + offsets[tid-1] - tid;
//                //n2_offset_bis = m + offsets[n2[tid]] - n2[tid];
//                //n2_offset = m + offsets[n2[tid]] - n2[tid] - 1;
//                n2_offset = m - n2[tid] + offsets[n2[tid]-1];
//                //printf("n1_offset: %d, offsets[]: %d\n", n1_offset, m + offsets[n1[tid]] - n1[tid]);
//            }
//        }

        //printf("tid: %d, n_insert: %d, n_insert_bis, %d, %d\n", tid, n_insert, n_insert_bis, n_insert == n_insert_bis);
        //printf("tid: %d, offset: %d, offset_bis %d, %d\n", tid, offset, offset_bis, offset == offset_bis);
        //printf("tid: %d, n0_offset: %d, n0_offset_bis %d, %d\n", tid, n0_offset, n0_offset_bis, n0_offset == n0_offset_bis);
        //printf("tid: %d, n1_offset: %d, n1_offset_bis %d, %d\n", tid, n1_offset, n1_offset_bis, n1_offset == n1_offset_bis);
        //printf("tid: %d, n2_offset: %d, n2_offset_bis %d, %d\n", tid, n2_offset, n2_offset_bis, n2_offset == n2_offset_bis);
        //if (n0_offset != n0_offset_bis) {
        //    printf("tid: %d, n0: %d, n0_bis: %d\n", tid, n0_offset, n0_offset_bis);
        //}
        //if (n1_offset != n1_offset_bis) {
        //    printf("tid: %d, n1: %d, n1_bis: %d\n", tid, n1_offset, n1_offset_bis);
        //}
        //if (n2_offset != n2_offset_bis) {
        //    printf("tid: %d, n2: %d, n2_bis: %d\n", tid, n2_offset, n2_offset_bis);
        //}

        //if (n_insert != n_insert_bis) {
        //    printf("tid: %d, n_insert: %d, n_insert_bis: %d\n", tid, n_insert, n_insert_bis);
        //}

        //if (offset == offset_bis) {
        //    printf("tid: %d, offset: %d, offset_bis: %d\n", tid, offset, offset_bis);
        //}

        //n0_offset = n0_offset_bis;
        //n1_offset = n1_offset_bis;
        //n2_offset = n2_offset_bis;

        //n_insert = n_insert_bis;
        //offset = offset_bis;

        int n0_old = n0[tid];
        int n1_old = n1[tid];
        int n2_old = n2[tid];

        //printf("tid: %d, n0: %d, n1: %d, n2: %d, n0_offset: %d, n1_offset: %d, n2_offset: %d\n", tid, n0_old, n1_old, n2_old, n0_offset, n1_offset, n2_offset);
        /* Change triangles */
        v2[tid] = p; // race condition

        v0[tid+offset] = b;
        v1[tid+offset] = c;
        v2[tid+offset] = p;

        v0[tid+offset+n_insert] = c;
        v1[tid+offset+n_insert] = a;
        v2[tid+offset+n_insert] = p;

        //printf("tid: %d, tid+offset: %d, tid+offset+n_insert: %d, m: %d\n", tid, tid+offset, tid+offset+n_insert, m);

        active_triangles[tid] = 1;
        active_triangles[tid+offset] = 1;
        active_triangles[tid+offset+n_insert] = 1;

        has_insertion[tid+offset] = 1;
        has_insertion[tid+offset+n_insert] = 1;

        //printf("tid: %d, offset: %d\n", tid, offset);

        /* Internal */
        n1[tid] = tid+offset;  // race condition
        n2[tid] = tid+offset+n_insert; // race condition

        n1[tid+offset] = tid+offset+n_insert;
        n2[tid+offset] = tid;

        n1[tid+offset+n_insert] = tid;
        n2[tid+offset+n_insert] = tid+offset;


        /* External */
        if (n0_index != -1) {  // We have a a-b neighbour
            if (has_insertion[n0_old]) {  // a-b neighbour is split
                if (n0_index == 0) {
                    n0[tid] = n0_old;     // if a-b is the first edge (impossible)
                }
                if (n0_index == 1) {      // if a-b is the 2nd edge
                    n0[tid] = n0_old+n0_offset; 
                }
                if (n0_index == 2) {      // if a-b is the 3rd edge
                    n0[tid] = n0_old+n0_offset+n_insert; 
                }
            }
            else {
                n0[tid] = n0_old;
                if (n0_index == 0) {
                    n0[n0_old] = tid;   // race condition
                }
                if (n0_index == 1) {
                    n1[n0_old] = tid;   // race condition
                }
                if (n0_index == 2) {
                    n2[n0_old] = tid;   // race condition
                }
            }
        }        
        if (n1_index != -1) {
            if (has_insertion[n1_old]) {
                if (n1_index == 0) {
                    n0[tid+offset] = n1_old;
                }
                if (n1_index == 1) {
                    n0[tid+offset] = n1_old+n1_offset;
                }
                if (n1_index == 2) {
                    n0[tid+offset] = n1_old+n1_offset+n_insert;
                }
            }
            else {
                n0[tid+offset] = n1_old;
                if (n1_index == 0) {
                    n0[n1_old] = tid+offset;  // race condition
                }
                if (n1_index == 1) {
                    n1[n1_old] = tid+offset;  // race condition
                }
                if (n1_index == 2) {
                    n2[n1_old] = tid+offset;  // race condition
                }
            }
        }        
        if (n2_index != -1) {
            if (has_insertion[n2_old]) {
                if (n2_index == 0) {
                    n0[tid+offset+n_insert] = n2_old;
                }
                if (n2_index == 1) {
                    n0[tid+offset+n_insert] = n2_old+n2_offset;
                }
                if (n2_index == 2) {
                    n0[tid+offset+n_insert] = n2_old+n2_offset+n_insert;
                }
            }
            else {
                n0[tid+offset+n_insert] = n2_old;
                if (n2_index == 0) {
                    n0[n2_old] = tid+offset+n_insert;  // race condition
                }
                if (n2_index == 1) {
                    n1[n2_old] = tid+offset+n_insert;  // race condition
                }
                if (n2_index == 2) {
                    n2[n2_old] = tid+offset+n_insert;  // race condition
                }
            }
        }
    }
}

__forceinline__ __device__ float sign(float x1, float y1, float x2, float y2, float x3, float y3) {
    float res = (x1-x3)*(y2-y3);
    res -= (x2-x3)*(y1-y3);
    return res;
    //return (x1-x3)*(y2-y3) - (x2-x3)*(y1-y3);
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
                                              int* corr_triangle) {
    unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if ((tid < m) && (has_insertion[tid])) {
        const int v0_r = v0[tid];
        const int v1_r = v1[tid];
        const int v2_r = v2[tid];

        const float ax = x[v0_r];
        const float ay = y[v0_r];
        
        const float bx = x[v1_r];
        const float by = y[v1_r];

        const float cx = x[v2_r];
        const float cy = y[v2_r];

        float px, py;
        for (int i = 0; i < n_insert; i++) {
            px = x_loc[i];
            py = y_loc[i];

            if (point_in_tri(px, py, ax, ay, bx, by, cx, cy)) {
                corr_triangle[loc_to_glob_index[i]] = tid;
            }
        }
    }
}

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
                                              int* corr_triangle) {
    unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if (tid < m_insert)  {
        const int v0_r = v0[tid];
        const int v1_r = v1[tid];
        const int v2_r = v2[tid];

        const float ax = x[v0_r];
        const float ay = y[v0_r];
        
        const float bx = x[v1_r];
        const float by = y[v1_r];

        const float cx = x[v2_r];
        const float cy = y[v2_r];

        const int tri_index_r = tri_loc_to_glob_index[tid];

        float px, py;
        for (int i = 0; i < n_insert; i++) {
            px = x_loc[i];
            py = y_loc[i];

            if (point_in_tri(px, py, ax, ay, bx, by, cx, cy)) {
                corr_triangle[loc_to_glob_index[i]] = tri_index_r;
            }
        }
    }
}

__global__ void copy_triangles_with_ins(const int* has_insertion,
                                        const int* offsets,
                                        const int* v0,
                                        const int* v1,
                                        const int* v2,
                                        const int m,
                                        int* v0_loc,
                                        int* v1_loc,
                                        int* v2_loc,
                                        int* tri_loc_to_glob_index) {
    unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if (tid < m) {
        if (has_insertion[tid]) {
            const int v0_r = v0[tid];
            const int v1_r = v1[tid];
            const int v2_r = v2[tid];
            int index = offsets[tid]-1;
            v0_loc[index] = v0_r;
            v1_loc[index] = v1_r;
            v2_loc[index] = v2_r;
            tri_loc_to_glob_index[index] = tid;
        }
    }
}

__global__ void copy_uninserted_points(const int* was_inserted,
                                       const int* offsets,
                                       const float* x,
                                       const float* y,
                                       const int n,
                                       float* x_loc,
                                       float* y_loc,
                                       int* loc_to_glob_index) {
    unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if (tid < n) {
        int was_inserted_reg = was_inserted[tid]; 
        float x_reg = x[tid];
        float y_reg = y[tid];
        if (was_inserted_reg == 0) {
            int index = tid-offsets[tid];
            //printf("tid: %d, index: %d\n", tid, index);
            x_loc[index] = x_reg;
            y_loc[index] = y_reg;
            loc_to_glob_index[index] = tid;
        }
    }
}

__global__ void update_was_inserted(int* was_inserted,
                                    int* winner_points,
                                    int* corr_triangle,
                                    int* has_insertion,
                                    int n,
                                    int m) {
    unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if (tid < m) {
        int val = winner_points[tid];
        if ((val != -1) && (val < n)) {
            was_inserted[val] = 1;
            corr_triangle[val] = -1;
            has_insertion[tid] = 1;
        }
        else {
            has_insertion[tid] = 0;
        }
    }
}
