#include "points.h"

#include <algorithm>
#include <iterator>
#include <ostream>
#include <vector>



std::ostream& operator<<(std::ostream& os, const Point& pt) {
    os << std::fixed;
    os << std::setprecision(3);
    os << pt.x << ", " << pt.y << '\n';
    return os;
}

bool operator==(const Point &a, const Point& b) {
    return (a.x == b.x) && (a.y == b.y);
}

bool operator<(const Point& a, const Point& b) {
    if (a.x == b.x) {
        return (a.y < b.y);
    }
    else {
        return (a.x < b.x);
    }
}

bool operator>(const Point& a, const Point& b) {
    return !(a < b) && !(a == b);
}


void fill_rand(std::vector<Point>& pts) {
    std::mt19937 gen;
    gen.seed(12345L);
    std::uniform_real_distribution<> dist(0.0f, 1.0f);
    for (auto& pt: pts) {
        pt.x = dist(gen);
        pt.y = dist(gen);
    }
}

void fill_rand_norm(std::vector<Point>& pts) {
    std::mt19937 gen;
    gen.seed(12345L);
    std::normal_distribution<float> dist(0.5f,0.1f);
    for (auto& pt: pts) {
        pt.x = dist(gen);
        pt.y = dist(gen);
    }
}

//void add_super_triangle(std::vector<Point>& pts) {
//    pts.push_back(Point{-15.0f, 0.0f});
//    pts.push_back(Point{15.0f, 0.0f});
//    pts.push_back(Point{-15.0f, 15.0f});
//}

void add_super_triangle(std::vector<Point>& pts,
                             float ax,
                             float ay,
                             float bx,
                             float by,
                             float cx,
                             float cy) {
    pts.push_back(Point{ax, ay});
    pts.push_back(Point{bx, by});
    pts.push_back(Point{cx, cy});
}

void add_super_triangle(std::vector<Point>& pts) {
    auto leftmost = std::min_element(pts.begin(),
                                     pts.end(),
                                     [](const Point& a, const Point& b) {
                                        if (a.x == b.x) {
                                            return a.y < b.y;
                                        }
                                        else {
                                            return a.x < b.x;
                                        }
                                     });
    auto rightmost = std::max_element(pts.begin(),
                                     pts.end(),
                                     [](const Point& a, const Point& b) {
                                        if (a.x == b.x) {
                                            return a.y < b.y;
                                        }
                                        else {
                                            return a.x < b.x;
                                        }
                                     });
    auto origin = *rightmost;

    auto temp = pts[0];
    pts[0] = *rightmost;
    *rightmost = temp;
    auto min_angle = std::min_element(pts.begin()+1,
                                      pts.end(),
                                      [&origin](const Point& a, const Point& b) {
                                            return (a.y-origin.y)/(a.x-origin.x) < (b.y-origin.y)/(b.x-origin.x);
                                      });

    auto max_angle = std::max_element(pts.begin()+1,
                                      pts.end(),
                                      [&origin](const Point& a, const Point& b) {
                                            return (a.y-origin.y)/(a.x-origin.x) < (b.y-origin.y)/(b.x-origin.x);
                                      });

    Point gamma = Point{origin.x, origin.y};


    auto find_intersection = [](const Point& a, const Point& b, const Point& c) {
        const float m = (a.y-b.y)/(a.x-b.x);
        const float p = 0.5f*((a.y+b.y)-m*(a.x+b.x));

        const float x = c.x;
        const float y = m*(x) + p;

        return Point{x, y};
    };

    Point alpha = find_intersection(*min_angle, gamma, *leftmost);
    Point beta = find_intersection(*max_angle, gamma, *leftmost);

    pts.push_back(gamma);
    pts.push_back(alpha);
    pts.push_back(beta);
}


float orient2d(const Point& a, const Point& b, const Point& c) {
    return (a.x-c.x)*(b.y-c.y) - (a.y-c.y)*(b.x-c.x);
}

float orient2d_exact(const Point& a, const Point& b, const Point& c) {
    exactinit();
    float pa[2] = {a.x, a.y};
    float pb[2] = {b.x, b.y};
    float pc[2] = {c.x, c.y};

    const float res = orient2d(pa, pb, pc);
    return res;
}
