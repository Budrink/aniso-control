#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <random>

namespace aniso {

template<int Dim>
using Vec = Eigen::Matrix<double, Dim, 1>;

template<int Dim>
using Mat = Eigen::Matrix<double, Dim, Dim>;

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
        Eigen::SelfAdjointEigenSolver<Mat<Dim>> solver(G);
        auto vals = solver.eigenvalues();
        auto vecs = solver.eigenvectors();
        for (int i = 0; i < Dim; ++i)
            vals(i) = std::clamp(vals(i), lo, hi);
        G = vecs * vals.asDiagonal() * vecs.transpose();
    }

    auto eigenvalues() const {
        Eigen::SelfAdjointEigenSolver<Mat<Dim>> solver(G);
        return solver.eigenvalues();
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

using RNG = std::mt19937_64;

} // namespace aniso
