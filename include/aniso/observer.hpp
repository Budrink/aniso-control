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
                                     RNG& rng,
                                     double E = 0.0) const = 0;
    virtual void set_sigma_G(double) {}
    virtual void set_E_noise_beta(double) {}
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
    double sigma_G_;          // G estimation noise scale (0 = perfect G knowledge)
    double E_noise_beta_;     // energy-dependent noise amplification (0 = no E effect)

public:
    ResolutionObserver(std::shared_ptr<IResolution<Dim>> res,
                       double sigma_G = 0.3, double E_noise_beta = 0.0)
        : resolution_(std::move(res)), sigma_G_(sigma_G),
          E_noise_beta_(E_noise_beta) {}

    void set_sigma_G(double s) override { sigma_G_ = std::max(s, 0.0); }
    void set_E_noise_beta(double b) override { E_noise_beta_ = std::max(b, 0.0); }
    IResolution<Dim>& resolution() override { return *resolution_; }
    const IResolution<Dim>& resolution() const override { return *resolution_; }

    Observation<Dim> observe(const Vec<Dim>& x,
                             const TensorField<Dim>& G,
                             RNG& rng,
                             double E = 0.0) const override
    {
        Observation<Dim> obs;
        std::normal_distribution<double> nd(0.0, 1.0);

        // Energy-dependent noise amplification:
        // hot medium → more radiation background → worse diagnostics
        double E_scale = std::sqrt(1.0 + E_noise_beta_ * std::max(E, 0.0));

        // --- State observation: y = x + L · ξ · E_scale ---
        Mat<Dim> L = resolution_->resolution_tensor(G);
        Vec<Dim> xi_x;
        for (int i = 0; i < Dim; ++i) xi_x(i) = nd(rng);
        obs.y = x + L * xi_x * E_scale;

        // --- G observation: noisy eigenvalue estimation ---
        if constexpr (Dim == 2) {
            auto e = fast2::eig(G.G);
            Mat<2> vecs;
            vecs(0,0)=e.v1x; vecs(1,0)=e.v1y;
            vecs(0,1)=e.v2x; vecs(1,1)=e.v2y;

            double li0 = std::max(std::abs(L.col(0).dot(vecs.col(0))),
                                  std::abs(L(0,0)));
            double li1 = std::max(std::abs(L.col(1).dot(vecs.col(1))),
                                  std::abs(L(1,1)));
            double n0 = (sigma_G_ > 1e-12 && li0 > 1e-12)
                       ? sigma_G_ * li0 * E_scale * nd(rng) : 0.0;
            double n1 = (sigma_G_ > 1e-12 && li1 > 1e-12)
                       ? sigma_G_ * li1 * E_scale * nd(rng) : 0.0;
            fast2::Eig2 hat{std::max(e.l1+n0,0.01), std::max(e.l2+n1,0.01),
                            e.v1x, e.v1y, e.v2x, e.v2y};
            obs.G_hat.G = fast2::reconstruct(hat);
            obs.G_hat.symmetrize();
        } else {
            Eigen::SelfAdjointEigenSolver<Mat<Dim>> solver(G.G);
            auto vals = solver.eigenvalues();
            auto vecs = solver.eigenvectors();

            Vec<Dim> vals_hat;
            for (int i = 0; i < Dim; ++i) {
                double li = std::abs(L.col(i).dot(vecs.col(i)));
                li = std::max(li, std::abs(L(i, i)));
                double noise = (sigma_G_ > 1e-12 && li > 1e-12)
                             ? sigma_G_ * li * E_scale * nd(rng) : 0.0;
                vals_hat(i) = std::max(vals(i) + noise, 0.01);
            }
            obs.G_hat.G = vecs * vals_hat.asDiagonal() * vecs.transpose();
            obs.G_hat.symmetrize();
        }

        // --- Fisher information (from true G — observer knows its own optics) ---
        obs.F = resolution_->fisher_info(G);

        return obs;
    }
};

} // namespace aniso
