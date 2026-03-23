#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <random>
#include <algorithm>

namespace aniso {

template<int Dim>
using Vec = Eigen::Matrix<double, Dim, 1>;

template<int Dim>
using Mat = Eigen::Matrix<double, Dim, Dim>;

// ============================================================
//  Fast analytical 2x2 symmetric matrix operations
//  Avoids Eigen's iterative SelfAdjointEigenSolver entirely.
// ============================================================
namespace fast2 {

struct Eig2 {
    double l1, l2;          // eigenvalues (l1 <= l2)
    double v1x, v1y;        // eigenvector for l1
    double v2x, v2y;        // eigenvector for l2
};

inline Eig2 eig(double a, double b, double d) {
    double avg  = 0.5 * (a + d);
    double diff = 0.5 * (a - d);
    double disc = std::sqrt(diff * diff + b * b);
    Eig2 r;
    r.l1 = avg - disc;
    r.l2 = avg + disc;
    if (std::abs(b) < 1e-15) {
        if (a <= d) { r.v1x=1; r.v1y=0; r.v2x=0; r.v2y=1; }
        else        { r.v1x=0; r.v1y=1; r.v2x=1; r.v2y=0; }
    } else {
        r.v1x = b;          r.v1y = r.l1 - a;
        double n = std::sqrt(r.v1x*r.v1x + r.v1y*r.v1y);
        r.v1x /= n;         r.v1y /= n;
        r.v2x = -r.v1y;     r.v2y = r.v1x;
    }
    return r;
}

inline Eig2 eig(const Mat<2>& M) {
    return eig(M(0,0), M(0,1), M(1,1));
}

inline Mat<2> inverse(const Mat<2>& M) {
    double det = M(0,0)*M(1,1) - M(0,1)*M(1,0);
    double inv_det = 1.0 / std::max(det, 1e-12);
    Mat<2> r;
    r(0,0) =  M(1,1) * inv_det;
    r(0,1) = -M(0,1) * inv_det;
    r(1,0) = -M(1,0) * inv_det;
    r(1,1) =  M(0,0) * inv_det;
    return r;
}

inline Mat<2> reconstruct(const Eig2& e) {
    Mat<2> r;
    r(0,0) = e.l1*e.v1x*e.v1x + e.l2*e.v2x*e.v2x;
    r(0,1) = e.l1*e.v1x*e.v1y + e.l2*e.v2x*e.v2y;
    r(1,0) = r(0,1);
    r(1,1) = e.l1*e.v1y*e.v1y + e.l2*e.v2y*e.v2y;
    return r;
}

} // namespace fast2

// Symmetric positive-definite tensor field with helpers
template<int Dim>
class TensorField {
public:
    Mat<Dim> G;

    TensorField() : G(Mat<Dim>::Identity()) {}
    explicit TensorField(const Mat<Dim>& m) : G(m) {}

    double trace() const { return G.trace(); }

    Mat<Dim> traceless() const {
        return G - (trace() / Dim) * Mat<Dim>::Identity();
    }

    void symmetrize() {
        G = 0.5 * (G + G.transpose());
    }

    void clamp_eigenvalues(double lo, double hi) {
        if constexpr (Dim == 2) {
            auto e = fast2::eig(G);
            e.l1 = std::clamp(e.l1, lo, hi);
            e.l2 = std::clamp(e.l2, lo, hi);
            G = fast2::reconstruct(e);
        } else {
            Eigen::SelfAdjointEigenSolver<Mat<Dim>> solver(G);
            auto vals = solver.eigenvalues();
            auto vecs = solver.eigenvectors();
            for (int i = 0; i < Dim; ++i)
                vals(i) = std::clamp(vals(i), lo, hi);
            G = vecs * vals.asDiagonal() * vecs.transpose();
        }
    }

    auto eigenvalues() const {
        if constexpr (Dim == 2) {
            auto e = fast2::eig(G);
            Vec<2> v; v(0) = e.l1; v(1) = e.l2;
            return v;
        } else {
            Eigen::SelfAdjointEigenSolver<Mat<Dim>> solver(G);
            return solver.eigenvalues();
        }
    }

    double traceless_norm_sq() const {
        Mat<Dim> Q = traceless();
        return (Q.transpose() * Q).trace();
    }
};

template<int Dim>
struct SimState {
    Vec<Dim> x;
    TensorField<Dim> G;
    double t = 0.0;
};

// Information-limited observation: what the controller actually sees
template<int Dim>
struct Observation {
    Vec<Dim> y;              // noisy state estimate
    TensorField<Dim> G_hat;  // noisy G estimate (controller never sees true G)
    Mat<Dim> F;              // Fisher information: observation quality per direction
};

using RNG = std::mt19937_64;

} // namespace aniso
