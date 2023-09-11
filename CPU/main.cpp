/**
 * @file main.cpp
 * @brief Main file for the project.
 * @author Tomasz Kwasniewicz
 * @date 2023-06-05
 */

#include <algorithm>
#include <bits/chrono.h>
#include <iostream>
#include <vector>
#include <cstdio>
#include <chrono>

#include "points.h"
#include "triangulation.h"


int main(int argc, char* argv[]) {
    int n = 1000;
    if (argc == 2) {
        n = atoi(argv[1]);
    }

    std::vector<Point> pts(n);
    fill_rand(pts);

    const bool save_to_file = true;
    const bool perform_checks = false;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    triangulate(pts, save_to_file, perform_checks);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    std::cout << "Took: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count()/1000.f << "[s]" << std::endl;
}


