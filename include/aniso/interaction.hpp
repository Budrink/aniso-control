#pragma once

#include "types.hpp"
#include <memory>

namespace aniso {

template<int Dim>
struct IInteraction {
    virtual ~IInteraction() = default;
    virtual Mat<Dim> compute(const TensorField<Dim>& G) const = 0;
    virtual void set_mu(double) {}
    virtual void set_r_max(double) {}
};

template<int Dim>
class NoInteraction : public IInteraction<Dim> {
public:
    Mat<Dim> compute(const TensorField<Dim>&) const override {
        return Mat<Dim>::Zero();
    }
};

// Re-entrant Landau: anisotropy grows for g_c < s < g_c+r_max,
// but is destroyed by "overheating" when s > g_c+r_max.
// S = mu_eff(r) * Q  -  nu * |Q|^2 * Q
// where r = tr(G)/Dim - g_c
//   mu_eff = mu * r * (1 - r/r_max)  when r_max > 0 and r > 0
//   mu_eff = mu * r                   otherwise (original Landau)
template<int Dim>
class LandauInteraction : public IInteraction<Dim> {
    double mu_, g_c_, nu_, r_max_;
public:
    LandauInteraction(double mu, double g_c, double nu, double r_max = 0)
        : mu_(mu), g_c_(g_c), nu_(nu), r_max_(r_max) {}
    void set_mu(double m) override { mu_ = m; }
    void set_r_max(double rm) override { r_max_ = rm; }

    Mat<Dim> compute(const TensorField<Dim>& G) const override {
        Mat<Dim> Q = G.traceless();
        double r = G.trace() / Dim - g_c_;
        double eff_mu;
        if (r_max_ > 1e-8 && r > 0.0) {
            eff_mu = mu_ * r * (1.0 - r / r_max_);
        } else {
            eff_mu = mu_ * r;
        }
        double Q_sq = (Q.transpose() * Q).trace();
        return eff_mu * Q - nu_ * Q_sq * Q;
    }
};

} // namespace aniso
