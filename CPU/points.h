#ifndef POINTS_H_
#define POINTS_H_

#include <cstddef>
#include <iostream>
#include <iomanip>
#include <ostream>
#include <random>
#include <vector>

extern "C" {
    #include "predicates.h"
};

/**
 * @class Point
 * @brief A struct to represent 2D coordinates.
 */
struct Point{
    float x {0.0f}; /**< x-coordinate */
    float y {0.0f}; /**< y-coordinate */

};

std::ostream& operator<<(std::ostream&, const Point&);

bool operator<(const Point& a, const Point& b);
bool operator==(const Point& a, const Point& b);
bool operator>(const Point& a, const Point& b);

float orient2d(const Point&, const Point&, const Point&);
float orient2d_exact(const Point&, const Point&, const Point&);

void fill_rand(std::vector<Point>&);
void fill_rand_norm(std::vector<Point>&);

void add_super_triangle(std::vector<Point>&);
void add_super_triangle(std::vector<Point>&, float, float, float, float, float, float);

#endif /* POINTS_H_ */
