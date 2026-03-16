#pragma once

#include "types.hpp"
#include <memory>
#include <cmath>

namespace aniso {

template<int Dim>
struct ICoupling {
    virtual ~ICoupling() = default;
    virtual Mat<Dim> drive(const Vec<Dim>& u) const = 0;
    virtual void set_alpha(double) {}
};

// alpha * |u|^(gamma-2) * u (x) u
template<int Dim>
class Rank1Coupling : public ICoupling<Dim> {
    double alpha_, gamma_;
public:
    Rank1Coupling(double alpha, double gamma) : alpha_(alpha), gamma_(gamma) {}
    void set_alpha(double a) override { alpha_ = a; }

    Mat<Dim> drive(const Vec<Dim>& u) const override {
        double u_norm = u.norm();
        if (u_norm < 1e-12) return Mat<Dim>::Zero();
        double scale = alpha_ * std::pow(u_norm, gamma_ - 2.0);
        return scale * (u * u.transpose());
    }
};

// alpha * |u|^gamma * I  (scalar-model compatible)
template<int Dim>
class IsotropicCoupling : public ICoupling<Dim> {
    double alpha_, gamma_;
public:
    IsotropicCoupling(double alpha, double gamma) : alpha_(alpha), gamma_(gamma) {}
    void set_alpha(double a) override { alpha_ = a; }

    Mat<Dim> drive(const Vec<Dim>& u) const override {
        double u_norm = u.norm();
        if (u_norm < 1e-12) return Mat<Dim>::Zero();
        return alpha_ * std::pow(u_norm, gamma_) * Mat<Dim>::Identity();
    }
};

} // namespace aniso
