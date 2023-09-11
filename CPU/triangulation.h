#ifndef TRIANGULATION_H_
#define TRIANGULATION_H_

#include <iostream>
#include <ostream>
#include <string>
#include <vector>

#include <cassert>

#include "points.h"

/**
 * @struct Node
 * @brief Represent a triangle node in the triangulation DAG.
 */
struct Node {
    bool is_leaf() const {return children.empty();};
    bool in_triangle(const int p, const std::vector<Point>&) const;

    int a;  /**< 1st vertex */
    int b;  /**< 2nd vertex */
    int c;  /**< 3rd vertex */

    int n0; /**< Neighbour, a-b */
    int n1; /**< Neighbour, b-c */
    int n2; /**< Neighbour, c-a */

    std::vector<int> children  {std::vector<int>()};  /**< Children of the node */
};

/**
 * @struct TriangulationDAG
 * @brief Datastructure to store the triangulation.
 */
struct TriangulationDAG {
    void insert_triangle(int v0, int v1, int v2);

    void insert(int p, const std::vector<Point>&);
    void legalize_edge(int t, int neigh, const std::vector<Point>&);

    int find_neigh(int a, int b, const Node& v) const;

    std::vector<Node> triangles {std::vector<Node>()};
};

std::ostream& operator<<(std::ostream&, const Node&);
std::ostream& operator<<(std::ostream&, const TriangulationDAG&);
bool inCircle(const Point& a, const Point& b, const Point& c, const Point& d);
bool inCircle_exact(const Point& a, const Point& b, const Point& c, const Point& d);

void triangulate(std::vector<Point>& pts, const bool save_to_file, const bool perform_checks);

void check_triangulation(const std::vector<Point>& pts, const TriangulationDAG& T);
#endif /* TRIANGULATION_H_ */
