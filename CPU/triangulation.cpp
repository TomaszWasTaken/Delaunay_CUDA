#include "triangulation.h"
#include <ostream>
#include <algorithm>
#include <vector>

extern "C" {
    #include "predicates.h"
};

#define USE_EXACT_PREDICATES 1

std::ostream& operator<<(std::ostream& os, const Node& node) {
    os << "Triangle: " 
       << node.a << ' ' 
       << node.b << ' '
       << node.c << "\n\t"
       << "Neighbours: "
       << node.n0 << ' ' 
       << node.n1 << ' '
       << node.n2 << "\n\t"
       << "Children: ";
   for (const auto& child: node.children) {
        os << child << ' ';
   }
   os << '\n';
   return os;
}

std::ostream& operator<<(std::ostream& os, const TriangulationDAG& T) {
    for (const auto& t: T.triangles) {
        os << t;
    }
    return os;
}

void TriangulationDAG::insert_triangle(int v0, int v1, int v2) {
    triangles.push_back({v0,v1,v2,-1,-1,-1,{}});
}

bool Node::in_triangle(int p, const std::vector<Point>& pts) const {
#if USE_EXACT_PREDICATES
    return (orient2d_exact(pts[a], pts[b], pts[p]) > 0.0f) &&
           (orient2d_exact(pts[b], pts[c], pts[p]) > 0.0f) &&
           (orient2d_exact(pts[c], pts[a], pts[p]) > 0.0f);
#else
    return (orient2d_exact(pts[a], pts[b], pts[p]) > 0.0f) &&
           (orient2d_exact(pts[b], pts[c], pts[p]) > 0.0f) &&
           (orient2d_exact(pts[c], pts[a], pts[p]) > 0.0f);
#endif
}

bool inCircle(const Point& a, const Point& b, const Point& c, const Point& d) {
    float val = 0.0f;

    float e11 = a.x-d.x;
    float e21 = b.x-d.x;
    float e31 = c.x-d.x;

    float e12 = a.y-d.y;
    float e22 = b.y-d.y;
    float e32 = c.y-d.y;

    float e13 = e11*e11 + e12*e12;
    float e23 = e21*e21 + e22*e22;
    float e33 = e31*e31 + e32*e32;

    val = e11*e22*e33 + e12*e23*e31 + e13*e21*e32 -
          e12*e21*e33 - e11*e23*e32 - e13*e22*e31;
    return val > 0.0f;
}

bool inCircle_exact(const Point& a, const Point& b, const Point& c, const Point& d) {
    exactinit();

    float pa[2] = {a.x, a.y};
    float pb[2] = {b.x, b.y};
    float pc[2] = {c.x, c.y};
    float pd[2] = {d.x, d.y};

    const float val = incircle(pa,pb,pc,pd);

    return val > 0.0f;
}

bool vertex_in_triangle(int a, const Node& v) {
    return (v.a == a || v.b == a || v.c == a);
}

bool edge_in_triangle(int a, int b, const Node& v) {
    return vertex_in_triangle(a, v) && vertex_in_triangle(b, v);
}

int TriangulationDAG::find_neigh(int a, int b, const Node& v) const {
    int res = -1;

    if (v.n0 != -1) {
        if (edge_in_triangle(a,b,triangles[v.n0])) {
            res = v.n0;
        }
    }
    if (v.n1 != -1) {
        if (edge_in_triangle(a,b,triangles[v.n1])) {
            res = v.n1;
        }
    }
    if (v.n2 != -1) {
        if (edge_in_triangle(a,b,triangles[v.n2])) {
            res = v.n2;
        }
    }
    return res;
}

int find_neigh_index(int a, int b, const Node& node) {
    int index = -1;
    if ((node.a == a && node.b == b) || (node.a == b && node.b == a)) {
        index = 0;
    }
    else if ((node.b == a && node.c == b) || (node.b == b && node.c == a))  {
        index = 1;
    }
    else if ((node.c == a && node.a == b) || (node.c == b && node.a == a))  {
        index = 2;
    }
    return index;
}

void TriangulationDAG::legalize_edge(int t, int neigh, const std::vector<Point>& pts) {
    if (neigh != -1) {
        int a = triangles[t].a;
        int b = triangles[t].b;
        int r = triangles[t].c;
        int d = -1;

        if (triangles[neigh].a != a && triangles[neigh].a != b && triangles[neigh].a != r) {
            d = triangles[neigh].a;
        }
        else if (triangles[neigh].b != a && triangles[neigh].b != b && triangles[neigh].b != r) {
            d = triangles[neigh].b;
        }
        else if (triangles[neigh].c != a && triangles[neigh].c != b && triangles[neigh].c != r) {
            d = triangles[neigh].c;
        }

#if USE_EXACT_PREDICATES
        if (inCircle_exact(pts[a],pts[b],pts[r],pts[d]) || inCircle_exact(pts[b],pts[a],pts[d],pts[r])) {
#else
        if (inCircle(pts[a],pts[b],pts[r],pts[d]) || inCircle(pts[b],pts[a],pts[d],pts[r])) {
#endif
            int n = triangles.size();

            triangles[t].children.push_back(n);
            triangles[t].children.push_back(n+1);
            triangles[neigh].children.push_back(n);
            triangles[neigh].children.push_back(n+1);

            int n1 = find_neigh(a,d,triangles[neigh]);
            int n2 = find_neigh(r,a,triangles[t]);
            triangles.push_back({a,d,r,n1,n+1,n2,{}});

            if (n1 != -1) {
                int e1 = find_neigh_index(a,d,triangles[n1]);
                if (e1 == 0) {
                    triangles[n1].n0 = n;
                }
                else if (e1 == 1) {
                    triangles[n1].n1 = n;
                }
                else if (e1 == 2) {
                    triangles[n1].n2 = n;
                }
            }

            if (n2 != -1) {
                int e2 = find_neigh_index(r,a,triangles[n2]);
                if (e2 == 0) {
                    triangles[n2].n0 = n;
                }
                else if (e2 == 1) {
                    triangles[n2].n1 = n;
                }
                else if (e2 == 2) {
                    triangles[n2].n2 = n;
                }
            }


            int n3 = find_neigh(d,b,triangles[neigh]);
            int n4 = find_neigh(b,r,triangles[t]);
            triangles.push_back({d,b,r,n3,n4,n,{}});

            if (n3 != -1) {
                int e3 = find_neigh_index(d,b,triangles[n3]);
                if (e3 == 0) {
                    triangles[n3].n0 = n+1;
                }
                else if (e3 == 1) {
                    triangles[n3].n1 = n+1;
                }
                else if (e3 == 2) {
                    triangles[n3].n2 = n+1;
                }
            }

            if (n4 != -1) {
                int e4 = find_neigh_index(b,r,triangles[n4]);
                if (e4 == 0) {
                    triangles[n4].n0 = n+1;
                }
                else if (e4 == 1) {
                    triangles[n4].n1 = n+1;
                }
                else if (e4 == 2) {
                    triangles[n4].n2 = n+1;
                }
            }

            legalize_edge(n, n1, pts);
            legalize_edge(n, n2, pts);

            legalize_edge(n+1, n3, pts);
            legalize_edge(n+1, n4, pts);
        }
    }
}

#define FLIP 1

void TriangulationDAG::insert(int p, const std::vector<Point>& pts) {
    int curr = 0;

    while (!triangles[curr].is_leaf()) {
        for (const auto& child: triangles[curr].children) {
            if (triangles[child].in_triangle(p, pts)) {
                curr = child;
            }
        } 
    }

    int new_index = triangles.size();

    int a = triangles[curr].a;
    int b = triangles[curr].b;
    int c = triangles[curr].c;

    int n1 = triangles[curr].n0;
    int n2 = triangles[curr].n1;
    int n3 = triangles[curr].n2;

    triangles.push_back({a,b,p,n1,new_index+1,new_index+2,{}});
    triangles.push_back({b,c,p,n2,new_index+2,new_index,{}});
    triangles.push_back({c,a,p,n3,new_index, new_index+1,{}});

    triangles[curr].children.push_back(new_index);
    triangles[curr].children.push_back(new_index+1);
    triangles[curr].children.push_back(new_index+2);

    if (n1 != -1) {
        int e1 = find_neigh_index(a,b,triangles[n1]);
        if (e1 == 0) {
            triangles[n1].n0 = new_index;
        }
        else if (e1 == 1) {
            triangles[n1].n1 = new_index;
        }
        else if (e1 == 2) {
            triangles[n1].n2 = new_index;
        }
    }
    if (n2 != -1) {
        int e2 = find_neigh_index(b,c,triangles[n2]);
        if (e2 == 0) {
            triangles[n2].n0 = new_index+1;
        }
        else if (e2 == 1) {
            triangles[n2].n1 = new_index+1;
        }
        else if (e2 == 2) {
            triangles[n2].n2 = new_index+1;
        }
    }
    if (n3 != -1) {
        int e3 = find_neigh_index(c,a,triangles[n3]);
        if (e3 == 0) {
            triangles[n3].n0 = new_index+2;
        }
        else if (e3 == 1) {
            triangles[n3].n1 = new_index+2;
        }
        else if (e3 == 2) {
            triangles[n3].n2 = new_index+2;
        }
    }

#if FLIP
    legalize_edge(new_index, triangles[new_index].n0, pts);
    legalize_edge(new_index+1, triangles[new_index+1].n0, pts);
    legalize_edge(new_index+2, triangles[new_index+2].n0, pts);
#endif

}

void triangulate(std::vector<Point>& pts, 
                 const bool save_to_file,
                 const bool perform_checks) {

    const int n = pts.size();

    add_super_triangle(pts, 15.0f, 0.0f,
                            -10.0f, -10.0f,
                            -10.0f, 15.0f);

    TriangulationDAG T;
    T.triangles.reserve(10*n);
    T.insert_triangle(n, n+1, n+2);

    for (int i = 0; i < n; ++i) {
        T.insert(i, pts);
    }

    FILE* fp = NULL;

    if (save_to_file) {
        fp = fopen("points.dat", "w");
        for (int i = 0; i < n; i++) {
            float x = pts[i].x;
            float y = pts[i].y;
            fprintf(fp, "%f, %f\n", x, y);
        }
        fclose(fp);

        fp = fopen("triangulation.dat", "w");
        for (const auto& t: T.triangles) {
            if (t.is_leaf()) {
                if ((t.a != n) && (t.a != n+1) && (t.a != n+2)) {
                    if ((t.b != n) && (t.b != n+1) && (t.b != n+2)) {
                        if ((t.c != n) && (t.c != n+1) && (t.c != n+2)) {
                          fprintf(fp, "%d, %d, %d\n", t.a, t.b, t.c);
                        }
                    }
                }
            }
        }
        fclose(fp);
    }

    if (perform_checks) {
        check_triangulation(pts, T);
    }
}

void check_triangulation(const std::vector<Point>& pts, const TriangulationDAG& T) {
    bool flag = false;
    for (const auto& t: T.triangles) {
        if (t.is_leaf()) {
            for (int i = 0; i < (int) pts.size(); ++i) {
                if (i != t.a && i != t.b && i != t.c) {
                    if (inCircle_exact(pts[t.a], pts[t.b], pts[t.c], pts[i])) {
                        flag = true;
                        break;
                    }
                }
            }
        }
    }
    std::cout << "Check Triangulation: " << flag << '\n';
}
