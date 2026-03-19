#pragma once

#include "types.hpp"
#include <cmath>
#include <algorithm>
#include <string>

namespace aniso {

template<int Dim>
struct IResolution {
    virtual ~IResolution() = default;

    // L(G): resolution tensor — noise amplitude matrix
    virtual Mat<Dim> resolution_tensor(const TensorField<Dim>& G) const = 0;

    // F(G) = L^{-2}: Fisher information — observation quality per direction
    virtual Mat<Dim> fisher_info(const TensorField<Dim>& G) const = 0;

    virtual void set_param(const std::string& name, double val) = 0;
    virtual std::string type_name() const = 0;
};

// ---------------------------------------------------------------------------
//  IdentityResolution — no resolution effects (backward compatibility)
//  L = sigma0 * I, F = I / sigma0^2
// ---------------------------------------------------------------------------
template<int Dim>
class IdentityResolution : public IResolution<Dim> {
    double sigma0_;
public:
    explicit IdentityResolution(double sigma0 = 0.04) : sigma0_(std::max(sigma0, 0.0)) {}

    void set_param(const std::string& name, double val) override {
        if (name == "sigma0" || name == "l0") sigma0_ = std::max(val, 0.0);
    }

    std::string type_name() const override { return "identity"; }

    Mat<Dim> resolution_tensor(const TensorField<Dim>&) const override {
        return sigma0_ * Mat<Dim>::Identity();
    }

    Mat<Dim> fisher_info(const TensorField<Dim>&) const override {
        double s = std::max(sigma0_, 1e-12);
        return (1.0 / (s * s)) * Mat<Dim>::Identity();
    }
};

// ---------------------------------------------------------------------------
//  MetricResolution — resolution scale deforms with G
//  L = l0 * G^{alpha/2}    (GR analogy: ds ~ sqrt(g) dx)
//  F = (1/l0^2) * G^{-alpha}
//
//  alpha = 1: standard metric geometry (l ~ sqrt(G))
//  alpha > 1: resolution degrades faster than metric stretches
//  alpha = 0: degenerates to IdentityResolution
// ---------------------------------------------------------------------------
template<int Dim>
class MetricResolution : public IResolution<Dim> {
    double l0_;
    double alpha_;
public:
    MetricResolution(double l0, double alpha)
        : l0_(std::max(l0, 0.0)), alpha_(alpha) {}

    void set_param(const std::string& name, double val) override {
        if (name == "l0")    l0_    = std::max(val, 0.0);
        if (name == "alpha") alpha_ = val;
    }

    std::string type_name() const override { return "metric"; }

    Mat<Dim> resolution_tensor(const TensorField<Dim>& G) const override {
        Eigen::SelfAdjointEigenSolver<Mat<Dim>> solver(G.G);
        auto vals = solver.eigenvalues();
        auto vecs = solver.eigenvectors();
        Vec<Dim> l_eig;
        for (int i = 0; i < Dim; ++i)
            l_eig(i) = l0_ * std::pow(std::max(vals(i), 0.01), alpha_ * 0.5);
        return vecs * l_eig.asDiagonal() * vecs.transpose();
    }

    Mat<Dim> fisher_info(const TensorField<Dim>& G) const override {
        Eigen::SelfAdjointEigenSolver<Mat<Dim>> solver(G.G);
        auto vals = solver.eigenvalues();
        auto vecs = solver.eigenvectors();
        Vec<Dim> f_eig;
        double s = std::max(l0_, 1e-12);
        double inv_l0_sq = 1.0 / (s * s);
        for (int i = 0; i < Dim; ++i)
            f_eig(i) = inv_l0_sq * std::pow(std::max(vals(i), 0.01), -alpha_);
        return vecs * f_eig.asDiagonal() * vecs.transpose();
    }
};

} // namespace aniso
