#include "triangulation.h"


#define FLIPPING 0
#define SAVE_TO_FILE 0

struct not_equals_minus_one : public thrust::unary_function<int,bool>
{
  __host__ __device__
  bool operator()(int x) { return x != -1; }
};

void triangulate(thrust::host_vector<float>& x_cpu, thrust::host_vector<float>& y_cpu) {

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    const int n = x_cpu.size();

    thrust::device_vector<float> x(n);
    thrust::device_vector<float> y(n);
    thrust::device_vector<int> corr_triangle(n);
    thrust::device_vector<int> was_inserted(n);

    x = x_cpu;
    y = y_cpu;
    /* Correspondig triangles, at first, all points assocated with super. */
    thrust::fill(corr_triangle.begin(), corr_triangle.end(), 0);
    thrust::fill(was_inserted.begin(), was_inserted.end(), 0);

    setup<<<1,1>>>(thrust::raw_pointer_cast(corr_triangle.data()), thrust::raw_pointer_cast(was_inserted.data()));

    const int M = 10*n;
    int m = 1;

    thrust::device_vector<int> v0(M);
    thrust::device_vector<int> v1(M);
    thrust::device_vector<int> v2(M);
    /* First triangle */
    v0[0] = 0;
    v1[0] = 1;
    v2[0] = 2;

    /* Neighbours data */
    thrust::device_vector<int> n0(M);
    thrust::device_vector<int> n1(M);
    thrust::device_vector<int> n2(M);

    thrust::fill(n0.begin(), n0.end(), -1);
    thrust::fill(n1.begin(), n1.end(), -1);
    thrust::fill(n2.begin(), n2.end(), -1);
    
    /* Helper data */
    thrust::device_vector<float> xc(M);
    thrust::device_vector<float> yc(M);

    thrust::device_vector<int> winner_points(M);
    thrust::device_vector<int> has_insertion(M);

    thrust::fill(winner_points.begin(), winner_points.end(), M);
    thrust::fill(has_insertion.begin(), has_insertion.end(), 0);

    thrust::device_vector<int> active_triangles(M);
    thrust::fill(active_triangles.begin(), active_triangles.end(), 0);

    thrust::device_vector<int> flip_vec(M);

    thrust::fill(flip_vec.begin(), flip_vec.end(), -1);

    thrust::device_vector<int> was_changed(M);
    thrust::fill(was_changed.begin(), was_changed.end(), 0);

    thrust::device_vector<int> n0_old(M);
    thrust::device_vector<int> n1_old(M);
    thrust::device_vector<int> n2_old(M);

    int blocksize = 256;
    int iter_count = 0;
    int gridsize_n = n/blocksize + ( !( n % blocksize) ? 0 : 1);

    thrust::device_vector<float> arr(M);
    thrust::fill(arr.begin(), arr.end(), FLT_MAX);

    int flag_cpu = 0;

    int* flag_gpu;
    cudaMalloc((void**) &flag_gpu, sizeof(int));

    cudaMemcpy(flag_gpu, &flag_cpu, sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Local indices
    thrust::device_vector<int> v0_loc(M);
    thrust::device_vector<int> v1_loc(M);
    thrust::device_vector<int> v2_loc(M);

    thrust::device_vector<int> tri_loc_to_glob_index(M);


    thrust::device_vector<float> x_loc(n);
    thrust::device_vector<float> y_loc(n);
    thrust::device_vector<int> pts_offsets(n);
    thrust::device_vector<int> loc_to_glob_index(n);

    thrust::device_vector<int> offsets(M);

    thrust::device_vector<int> m_range(M);

    while (true) {
        thrust::fill(has_insertion.begin(), has_insertion.end(), 0);
        thrust::fill(winner_points.begin(), winner_points.end(), M);
        thrust::fill(arr.begin(), arr.end(), FLT_MAX);

        int gridsize = m/blocksize + ( !( m % blocksize) ? 0 : 1);

        // Determine circumcircles
        compute_centers<<<gridsize,blocksize>>>(thrust::raw_pointer_cast(x.data()),
                                                thrust::raw_pointer_cast(y.data()),
                                                n,
                                                thrust::raw_pointer_cast(v0.data()),
                                                thrust::raw_pointer_cast(v1.data()),
                                                thrust::raw_pointer_cast(v2.data()),
                                                m,
                                                thrust::raw_pointer_cast(xc.data()),   // <- this modified
                                                thrust::raw_pointer_cast(yc.data()));  // <- this modified

        pick_winner_points_bis_a<<<gridsize_n, blocksize>>>(thrust::raw_pointer_cast(x.data()),
                                                            thrust::raw_pointer_cast(y.data()),
                                                            thrust::raw_pointer_cast(corr_triangle.data()),
                                                            n,
                                                            thrust::raw_pointer_cast(xc.data()),
                                                            thrust::raw_pointer_cast(yc.data()),
                                                            m,
                                                            thrust::raw_pointer_cast(arr.data()));  // <- this modified
                                                            
        pick_winner_points_bis_b<<<gridsize_n, blocksize>>>(thrust::raw_pointer_cast(x.data()),
                                                            thrust::raw_pointer_cast(y.data()),
                                                            thrust::raw_pointer_cast(corr_triangle.data()),
                                                            n,
                                                            thrust::raw_pointer_cast(xc.data()),
                                                            thrust::raw_pointer_cast(yc.data()),
                                                            m,
                                                            thrust::raw_pointer_cast(arr.data()),
                                                            thrust::raw_pointer_cast(winner_points.data()));  // <- this modified

        update_was_inserted<<<gridsize, blocksize>>>(thrust::raw_pointer_cast(was_inserted.data()),
                                                     thrust::raw_pointer_cast(winner_points.data()),
                                                     thrust::raw_pointer_cast(corr_triangle.data()),
                                                     thrust::raw_pointer_cast(has_insertion.data()),
                                                     n,
                                                     m);

        thrust::inclusive_scan(was_inserted.begin(), was_inserted.end(), pts_offsets.begin());

        thrust::inclusive_scan(has_insertion.begin(), has_insertion.end(), offsets.begin());

        int val = thrust::reduce(has_insertion.begin(), has_insertion.begin()+m, 0, thrust::plus<int>());

        int n_insert = thrust::reduce(was_inserted.begin(), was_inserted.end(), 0, thrust::plus<int>());

        n_insert = n - n_insert;

        copy_uninserted_points<<<gridsize_n, blocksize>>>(thrust::raw_pointer_cast(was_inserted.data()),
                                                          thrust::raw_pointer_cast(pts_offsets.data()),
                                                          thrust::raw_pointer_cast(x.data()),
                                                          thrust::raw_pointer_cast(y.data()),
                                                          n,
                                                          thrust::raw_pointer_cast(x_loc.data()),
                                                          thrust::raw_pointer_cast(y_loc.data()),
                                                          thrust::raw_pointer_cast(loc_to_glob_index.data()));

        thrust::fill(active_triangles.begin(), active_triangles.end(), 0);

        cudaEventRecord(start);
                              
        insert_points_bis<<<gridsize, blocksize>>>(thrust::raw_pointer_cast(winner_points.data()),
                               thrust::raw_pointer_cast(v0.data()),
                               thrust::raw_pointer_cast(v1.data()),
                               thrust::raw_pointer_cast(v2.data()),
                               thrust::raw_pointer_cast(n0.data()),
                               thrust::raw_pointer_cast(n1.data()),
                               thrust::raw_pointer_cast(n2.data()),
                               thrust::raw_pointer_cast(corr_triangle.data()),
                               thrust::raw_pointer_cast(has_insertion.data()),
                               m,
                               thrust::raw_pointer_cast(active_triangles.data()),
                               thrust::raw_pointer_cast(offsets.data()));
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, start, stop);

        thrust::inclusive_scan(has_insertion.begin(), has_insertion.end(), offsets.begin());

        //std::cout << "iter: " << iter_count << ", t: " << milliseconds << '\n';

        m += 2*val;

        gridsize = m/blocksize + ( !( m % blocksize) ? 0 : 1);

        copy_triangles_with_ins<<<gridsize, blocksize>>>(thrust::raw_pointer_cast(has_insertion.data()),
                                                         thrust::raw_pointer_cast(offsets.data()),
                                                         thrust::raw_pointer_cast(v0.data()),
                                                         thrust::raw_pointer_cast(v1.data()),
                                                         thrust::raw_pointer_cast(v2.data()),
                                                         m,
                                                         thrust::raw_pointer_cast(v0_loc.data()),
                                                         thrust::raw_pointer_cast(v1_loc.data()),
                                                         thrust::raw_pointer_cast(v2_loc.data()),
                                                         thrust::raw_pointer_cast(tri_loc_to_glob_index.data()));


        cudaEventRecord(start);

        int m_insert = 3*val;
        int blocksize_m = 64;
        int gridsize_m = (m_insert/blocksize_m) + (!(m_insert % blocksize_m) ? 0 : 1);

        //std::cout << "m_insert: " << m_insert << ", n_insert: " << n_insert << '\n';

        update_corr_triangles_bis_bis_<<<gridsize_m, blocksize_m>>>(thrust::raw_pointer_cast(x.data()),
                                                                  thrust::raw_pointer_cast(y.data()),
                                                                  thrust::raw_pointer_cast(x_loc.data()),
                                                                  thrust::raw_pointer_cast(y_loc.data()),
                                                                  n_insert,
                                                                  m_insert,
                                                                  thrust::raw_pointer_cast(v0_loc.data()),
                                                                  thrust::raw_pointer_cast(v1_loc.data()),
                                                                  thrust::raw_pointer_cast(v2_loc.data()),
                                                                  thrust::raw_pointer_cast(loc_to_glob_index.data()),
                                                                  thrust::raw_pointer_cast(tri_loc_to_glob_index.data()),
                                                                  thrust::raw_pointer_cast(corr_triangle.data()));
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, start, stop);


#if FLIPPING
        const int iter_max = 1000;
        int flip_loop_counter = 0;

        while (flip_loop_counter < iter_max) {
            thrust::fill(flip_vec.begin(), flip_vec.end(), -1);
            flip<<<gridsize, blocksize>>>(thrust::raw_pointer_cast(x.data()),
                          thrust::raw_pointer_cast(y.data()),
                          n,
                          m,
                          thrust::raw_pointer_cast(v0.data()),
                          thrust::raw_pointer_cast(v1.data()),
                          thrust::raw_pointer_cast(v2.data()),
                          thrust::raw_pointer_cast(n0.data()),
                          thrust::raw_pointer_cast(n1.data()),
                          thrust::raw_pointer_cast(n2.data()),
                          thrust::raw_pointer_cast(active_triangles.data()),
                          thrust::raw_pointer_cast(flip_vec.data()));

            thrust::fill(active_triangles.begin(), active_triangles.end(), 0);

            flip_active_tri<<<gridsize, blocksize>>>(thrust::raw_pointer_cast(flip_vec.data()),
                            thrust::raw_pointer_cast(active_triangles.data()),
                            m);



            remove_duplicates_bis<<<1,1>>>(thrust::raw_pointer_cast(flip_vec.data()), m);
            remove_duplicates<<<gridsize, blocksize>>>(thrust::raw_pointer_cast(flip_vec.data()), m);

            n0_old = n0;
            n1_old = n1;
            n2_old = n2;

            performFlips<<<gridsize,blocksize>>>(thrust::raw_pointer_cast(v0.data()),
                                  thrust::raw_pointer_cast(v1.data()),
                                  thrust::raw_pointer_cast(v2.data()),
                                  thrust::raw_pointer_cast(n0.data()),
                                  thrust::raw_pointer_cast(n1.data()),
                                  thrust::raw_pointer_cast(n2.data()),
                                  m,
                                  thrust::raw_pointer_cast(flip_vec.data()),
                                  thrust::raw_pointer_cast(was_changed.data()));

            redouble_flips<<<gridsize, blocksize>>>(thrust::raw_pointer_cast(flip_vec.data()), m);

            updateNeighsAfterFlips<<<gridsize, blocksize>>>(thrust::raw_pointer_cast(v0.data()),
                                            thrust::raw_pointer_cast(v1.data()),
                                            thrust::raw_pointer_cast(v2.data()),
                                            thrust::raw_pointer_cast(n0.data()),
                                            thrust::raw_pointer_cast(n1.data()),
                                            thrust::raw_pointer_cast(n2.data()),
                                            thrust::raw_pointer_cast(n0_old.data()),
                                            thrust::raw_pointer_cast(n1_old.data()),
                                            thrust::raw_pointer_cast(n2_old.data()),
                                            m,
                                            thrust::raw_pointer_cast(flip_vec.data()),
                                            thrust::raw_pointer_cast(was_changed.data()));

            flip_loop_counter++;

            bool flag = thrust::any_of(flip_vec.begin(), flip_vec.end(), not_equals_minus_one());

            if (flag == false) {
                break;
            }
        }

        update_corr_triangles_bis_<<<gridsize, blocksize>>>(thrust::raw_pointer_cast(x.data()),
                                       thrust::raw_pointer_cast(y.data()),
                                       n,
                                       m,
                                       thrust::raw_pointer_cast(v0.data()),
                                       thrust::raw_pointer_cast(v1.data()),
                                       thrust::raw_pointer_cast(v2.data()),
                                       thrust::raw_pointer_cast(was_inserted.data()),
                                       thrust::raw_pointer_cast(has_insertion.data()),
                                       thrust::raw_pointer_cast(corr_triangle.data()));
#endif
        iter_count++;
        set_flag_to_0<<<1,1>>>(flag_gpu);
        loop_control<<<gridsize_n, blocksize>>>(thrust::raw_pointer_cast(was_inserted.data()), n, flag_gpu);
        cudaMemcpy(&flag_cpu, flag_gpu, sizeof(int), cudaMemcpyDeviceToHost);

        if (flag_cpu == n) {
            break;
        }
    }

    cudaFree(flag_gpu);
    //while (thrust::reduce(corr_triangle.begin(), corr_triangle.end(), 0, thrust::plus<int>()) != -n);
    //while (thrust::reduce(was_inserted.begin(), was_inserted.end(), 0, thrust::plus<int>()) != n);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << std::setprecision(8) << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()/1000.0f << "[s]" << std::endl;


#if SAVE_TO_FILE
    FILE* fp = fopen("points.dat", "w");
    for (int i = 3; i < n; i++) {
        fprintf(fp, "%f %f\n", x_cpu[i], y_cpu[i]);
    }
    fclose(fp);

    fp = fopen("delaunay.dat", "w");
    for (int i = 0; i < m; i++) {
        int v_0 = v0[i];
        int v_1 = v1[i];
        int v_2 = v2[i];
        if ((v_0 != 0) && (v_0 != 1) && (v_0 != 2)) {
            if ((v_1 != 0) && (v_1 != 1) && (v_1 != 2)) {
                if ((v_2 != 0) && (v_2 != 1) && (v_2 != 2)) {
                    fprintf(fp, "%d %d %d\n", v_0, v_1, v_2);
                }
            }
        }
    }

    fclose(fp);

#endif
}
