#pragma once

#include "types.hpp"
#include "coupling.hpp"
#include "interaction.hpp"
#include "feedback.hpp"
#include "observer.hpp"
#include "controller.hpp"
#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>
#include <random>

namespace aniso {

template<int Dim>
struct GridParams {
    int Nx = 48, Ny = 48;

    // Plant
    Mat<Dim> A  = Mat<Dim>::Zero();
    Mat<Dim> B  = Mat<Dim>::Identity();
    Vec<Dim> w  = Vec<Dim>::Zero();
    Vec<Dim> x0 = Vec<Dim>::Zero();

    // Time
    double dt = 0.01;

    // Energy dynamics
    double D_E        = 0.5;   // energy diffusion base coefficient
    double gamma_diss = 1.0;   // energy dissipation rate

    // G dynamics
    double tau_0      = 1.0;   // base relaxation time
    double kappa_tau  = 20.0;  // anisotropy slows relaxation: tau = tau_0*(1+kappa_tau*aniso^2)
    double noise_amp  = 0.5;   // noise base amplitude (scaled by sqrt(E))

    // State diffusion through G^{-1}
    double D_x = 0.1;

    // Spatial drive profile (Gaussian heating zone)
    double drive_cx = 0.5, drive_cy = 0.5;
    double drive_rx = 0.18, drive_ry = 0.18;
    double drive_peak = 5.0;

    // Eigenvalue clamp
    double eig_lo = 0.3;
    double eig_hi = 5.0;

    // Initial G perturbation
    double g_noise_init = 0.05;

    uint64_t seed = 42;
};

template<int Dim>
class GridEngine {
    int Nx_, Ny_, total_;
    std::vector<Vec<Dim>> x_, x_buf_;
    std::vector<TensorField<Dim>> G_, G_buf_;
    std::vector<double> E_, E_buf_;              // energy field
    std::vector<Vec<Dim>> last_u_;
    std::vector<double> drive_profile_;

    std::unique_ptr<ICoupling<Dim>>    coupling_;
    std::unique_ptr<IInteraction<Dim>> interaction_;  // kept for compat, unused
    std::unique_ptr<IFeedback<Dim>>    feedback_;
    std::unique_ptr<IObserver<Dim>>    observer_;
    std::unique_ptr<IController<Dim>>  controller_;

    GridParams<Dim> params_;
    double t_ = 0;
    RNG rng_;
    bool initialized_ = false;

    int idx(int i, int j) const { return i * Ny_ + j; }

    // Compute G^{-1} via eigendecomposition (safe due to eigenvalue clamping)
    Mat<Dim> G_inv(int k) const {
        Eigen::SelfAdjointEigenSolver<Mat<Dim>> solver(G_[k].G);
        auto ev = solver.eigenvalues();
        auto evec = solver.eigenvectors();
        Vec<Dim> ev_inv;
        for (int d = 0; d < Dim; ++d)
            ev_inv(d) = 1.0 / std::max(ev(d), 0.01);
        return evec * ev_inv.asDiagonal() * evec.transpose();
    }

public:
    GridEngine(GridParams<Dim> p,
               std::unique_ptr<ICoupling<Dim>>    coupling,
               std::unique_ptr<IInteraction<Dim>> interaction,
               std::unique_ptr<IFeedback<Dim>>    feedback,
               std::unique_ptr<IObserver<Dim>>    observer,
               std::unique_ptr<IController<Dim>>  controller)
        : Nx_(p.Nx), Ny_(p.Ny), total_(p.Nx * p.Ny)
        , coupling_(std::move(coupling))
        , interaction_(std::move(interaction))
        , feedback_(std::move(feedback))
        , observer_(std::move(observer))
        , controller_(std::move(controller))
        , params_(std::move(p))
    {
        x_.resize(total_); x_buf_.resize(total_);
        G_.resize(total_); G_buf_.resize(total_);
        E_.resize(total_, 0.0); E_buf_.resize(total_, 0.0);
        last_u_.resize(total_);
        drive_profile_.resize(total_);
        rebuild_drive_profile();
    }

    void rebuild_drive_profile() {
        for (int i = 0; i < Nx_; ++i) {
            double rx = (Nx_ > 1) ? (double)i / (Nx_ - 1) : 0.5;
            for (int j = 0; j < Ny_; ++j) {
                double ry = (Ny_ > 1) ? (double)j / (Ny_ - 1) : 0.5;
                double dx = (rx - params_.drive_cx)
                          / std::max(params_.drive_rx, 0.01);
                double dy = (ry - params_.drive_cy)
                          / std::max(params_.drive_ry, 0.01);
                drive_profile_[idx(i, j)] =
                    params_.drive_peak * std::exp(-0.5 * (dx*dx + dy*dy));
            }
        }
    }

    void reset() {
        rng_.seed(params_.seed);
        std::normal_distribution<double> noise(0, 0.02);
        std::normal_distribution<double> g_noise(0, params_.g_noise_init);
        for (int k = 0; k < total_; ++k) {
            x_[k] = params_.x0;
            for (int d = 0; d < Dim; ++d) x_[k](d) += noise(rng_);
            G_[k] = TensorField<Dim>();
            E_[k] = 0.0;

            if (params_.g_noise_init > 1e-8) {
                Mat<Dim> pert = Mat<Dim>::Zero();
                for (int a = 0; a < Dim; ++a)
                    for (int b = a; b < Dim; ++b) {
                        double v = g_noise(rng_);
                        pert(a, b) = v;
                        pert(b, a) = v;
                    }
                double tr = pert.trace();
                for (int a = 0; a < Dim; ++a) pert(a, a) -= tr / Dim;
                G_[k].G += pert;
                G_[k].symmetrize();
                G_[k].clamp_eigenvalues(params_.eig_lo, params_.eig_hi);
            }

            last_u_[k] = Vec<Dim>::Zero();
        }
        t_ = 0;
        initialized_ = true;
    }

    bool step() {
        if (!initialized_) reset();

        const double dt = params_.dt;
        const auto I = Mat<Dim>::Identity();
        std::normal_distribution<double> ndist(0.0, 1.0);
        const double sqrt_dt = std::sqrt(dt);

        // ============================================================
        //  Phase 1: Controller + state dynamics
        // ============================================================
        for (int i = 0; i < Nx_; ++i) {
            for (int j = 0; j < Ny_; ++j) {
                int k = idx(i, j);

                Vec<Dim> y = observer_->observe(x_[k], G_[k], rng_);
                Vec<Dim> u = controller_->compute(t_, y, G_[k]);
                last_u_[k] = u;

                // State: plant + G-state coupling + control + disturbance
                Vec<Dim> fb = feedback_->coupling(G_[k], x_[k]);
                Vec<Dim> dxv = params_.A * x_[k] + fb
                             + params_.B * u + params_.w;

                // State diffusion through G^{-1} (barrier blocks state flow)
                Mat<Dim> gi = G_inv(k);
                double gxx = gi(0, 0), gyy = gi(1, 1);
                Vec<Dim> x_flow = Vec<Dim>::Zero();
                if (i > 0)     x_flow += gxx * (x_[idx(i-1,j)] - x_[k]);
                if (i < Nx_-1) x_flow += gxx * (x_[idx(i+1,j)] - x_[k]);
                if (j > 0)     x_flow += gyy * (x_[idx(i,j-1)] - x_[k]);
                if (j < Ny_-1) x_flow += gyy * (x_[idx(i,j+1)] - x_[k]);
                dxv += params_.D_x * x_flow;

                x_buf_[k] = x_[k] + dxv * dt;
            }
        }

        // ============================================================
        //  Phase 2: Energy injection + diffusion + dissipation
        // ============================================================
        for (int i = 0; i < Nx_; ++i) {
            for (int j = 0; j < Ny_; ++j) {
                int k = idx(i, j);

                // Energy injection from controller
                double u_norm = last_u_[k].norm();
                double inject = 0.0;
                if (u_norm > 1e-12)
                    inject = coupling_->drive(last_u_[k]).trace()
                           * drive_profile_[k];

                // Energy anisotropic diffusion through G^{-1}
                Mat<Dim> gi = G_inv(k);
                double gxx = gi(0, 0), gyy = gi(1, 1);
                double dE_flow = 0;
                if (i > 0)     dE_flow += gxx * (E_[idx(i-1,j)] - E_[k]);
                if (i < Nx_-1) dE_flow += gxx * (E_[idx(i+1,j)] - E_[k]);
                if (j > 0)     dE_flow += gyy * (E_[idx(i,j-1)] - E_[k]);
                if (j < Ny_-1) dE_flow += gyy * (E_[idx(i,j+1)] - E_[k]);

                // Energy dissipation
                double dissip = params_.gamma_diss * E_[k];

                E_buf_[k] = E_[k] + (inject + params_.D_E * dE_flow
                          - dissip) * dt;
                E_buf_[k] = std::max(E_buf_[k], 0.0);
            }
        }

        // ============================================================
        //  Phase 3: G dynamics — drive(u), relax with aniso tau, noise(E)
        // ============================================================
        for (int i = 0; i < Nx_; ++i) {
            for (int j = 0; j < Ny_; ++j) {
                int k = idx(i, j);

                // Rank-1 drive from controller, modulated by drive profile
                Mat<Dim> drive = coupling_->drive(last_u_[k])
                               * drive_profile_[k];

                // Relaxation with anisotropy-dependent tau
                Mat<Dim> Q = G_[k].traceless();
                double Q_norm = std::sqrt((Q.transpose() * Q).trace());
                double s = G_[k].trace() / Dim;
                double aniso = Q_norm / std::max(s, 0.1);
                double tau_eff = params_.tau_0
                    * (1.0 + params_.kappa_tau * aniso * aniso);
                Mat<Dim> relax = -(G_[k].G - I) / tau_eff;

                G_buf_[k].G = G_[k].G + (drive + relax) * dt;

                // Stochastic noise scaled by sqrt(local energy)
                double local_E = std::max(E_buf_[k], 0.0);
                if (params_.noise_amp > 1e-8 && local_E > 1e-8) {
                    Mat<Dim> xi = Mat<Dim>::Zero();
                    for (int a = 0; a < Dim; ++a)
                        for (int b = a; b < Dim; ++b) {
                            double v = ndist(rng_);
                            xi(a, b) = v;
                            xi(b, a) = v;
                        }
                    double tr = xi.trace();
                    for (int a = 0; a < Dim; ++a) xi(a, a) -= tr / Dim;
                    G_buf_[k].G += params_.noise_amp
                                 * std::sqrt(local_E) * sqrt_dt * xi;
                }
                G_buf_[k].symmetrize();
                G_buf_[k].clamp_eigenvalues(params_.eig_lo, params_.eig_hi);
            }
        }

        std::swap(x_, x_buf_);
        std::swap(E_, E_buf_);
        std::swap(G_, G_buf_);
        t_ += dt;
        return true;
    }

    // --- accessors ---
    int    Nx()    const { return Nx_; }
    int    Ny()    const { return Ny_; }
    int    total() const { return total_; }
    double t()     const { return t_; }
    bool   done()  const { return false; }

    const Vec<Dim>&         x(int i, int j)      const { return x_[idx(i,j)]; }
    const TensorField<Dim>& G(int i, int j)      const { return G_[idx(i,j)]; }
    const Vec<Dim>&         last_u(int i, int j)  const { return last_u_[idx(i,j)]; }
    double                  E(int i, int j)       const { return E_[idx(i,j)]; }
    double                  drive_prof(int i, int j) const { return drive_profile_[idx(i,j)]; }

    GridParams<Dim>&       params()       { return params_; }
    const GridParams<Dim>& params() const { return params_; }

    IController<Dim>&  ctrl()  { return *controller_; }
    ICoupling<Dim>&    coup()  { return *coupling_; }
    IObserver<Dim>&    obs()   { return *observer_; }

    void swap_controller(std::unique_ptr<IController<Dim>> c) {
        controller_ = std::move(c);
    }

    double health(int i, int j) const {
        double tr = G_[idx(i,j)].trace();
        double max_tr = params_.eig_hi * Dim;
        return std::clamp(1.0 - (tr - Dim) / (max_tr - Dim), 0.0, 1.0);
    }

    double anisotropy(int i, int j) const {
        auto ev = G_[idx(i,j)].eigenvalues();
        double lmax = ev.maxCoeff();
        double lmin = std::max(ev.minCoeff(), 1e-6);
        return lmax / lmin - 1.0;
    }
};

} // namespace aniso
