#pragma once

#include "types.hpp"
#include "resolution.hpp"
#include <memory>
#include <cmath>
#include <algorithm>

namespace aniso {

template<int Dim>
struct IObserver {
    virtual ~IObserver() = default;
    virtual Observation<Dim> observe(const Vec<Dim>& x,
                                     const TensorField<Dim>& G,
                                     RNG& rng) const = 0;
    virtual void set_sigma_G(double) {}
    virtual IResolution<Dim>& resolution() = 0;
    virtual const IResolution<Dim>& resolution() const = 0;
};

// ---------------------------------------------------------------------------
//  ResolutionObserver — observation noise shaped by resolution tensor L(G)
//
//  State:  y  = x + L(G) · ξ_x           ξ_x ~ N(0, I)
//  Tensor: G_hat eigenvalues perturbed:   λ̂_i = λ_i + σ_G · l_i · ξ_i
//          where l_i = l₀ · λ_i^{α/2}    (from resolution tensor)
//  Fisher: F = resolution.fisher_info(G)
// ---------------------------------------------------------------------------
template<int Dim>
class ResolutionObserver : public IObserver<Dim> {
    std::shared_ptr<IResolution<Dim>> resolution_;
    double sigma_G_;   // G estimation noise scale (0 = perfect G knowledge)

public:
    ResolutionObserver(std::shared_ptr<IResolution<Dim>> res, double sigma_G = 0.3)
        : resolution_(std::move(res)), sigma_G_(sigma_G) {}

    void set_sigma_G(double s) override { sigma_G_ = std::max(s, 0.0); }
    IResolution<Dim>& resolution() override { return *resolution_; }
    const IResolution<Dim>& resolution() const override { return *resolution_; }

    Observation<Dim> observe(const Vec<Dim>& x,
                             const TensorField<Dim>& G,
                             RNG& rng) const override
    {
        Observation<Dim> obs;
        std::normal_distribution<double> nd(0.0, 1.0);

        // --- State observation: y = x + L · ξ ---
        Mat<Dim> L = resolution_->resolution_tensor(G);
        Vec<Dim> xi_x;
        for (int i = 0; i < Dim; ++i) xi_x(i) = nd(rng);
        obs.y = x + L * xi_x;

        // --- G observation: noisy eigenvalue estimation ---
        Eigen::SelfAdjointEigenSolver<Mat<Dim>> solver(G.G);
        auto vals = solver.eigenvalues();
        auto vecs = solver.eigenvectors();

        Vec<Dim> vals_hat;
        for (int i = 0; i < Dim; ++i) {
            double li = std::abs(L.col(i).dot(vecs.col(i)));
            li = std::max(li, std::abs(L(i, i)));
            double noise = (sigma_G_ > 1e-12 && li > 1e-12)
                         ? sigma_G_ * li * nd(rng) : 0.0;
            vals_hat(i) = std::max(vals(i) + noise, 0.01);
        }
        obs.G_hat.G = vecs * vals_hat.asDiagonal() * vecs.transpose();
        obs.G_hat.symmetrize();

        // --- Fisher information (from true G — observer knows its own optics) ---
        obs.F = resolution_->fisher_info(G);

        return obs;
    }
};

} // namespace aniso
