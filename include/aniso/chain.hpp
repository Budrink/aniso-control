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
struct ChainParams {
    int N = 64;
    double D_G = 0.05;
    double D_x = 0.1;
    double drive_center = 0.5;
    double drive_width  = 0.2;
    double drive_peak   = 2.5;

    Mat<Dim> A  = Mat<Dim>::Zero();
    Mat<Dim> B  = Mat<Dim>::Identity();
    Vec<Dim> w  = Vec<Dim>::Zero();
    Vec<Dim> x0 = Vec<Dim>::Zero();

    double tau    = 8.0;
    double dt     = 0.005;
    double t_end  = 300.0;
    double eig_lo = 0.3;
    double eig_hi = 5.0;
    uint64_t seed = 42;

    // Bistability: positive feedback above critical trace/Dim
    // When trace/Dim > s_crit, G-I is amplified at rate trap*excess
    // This competes with relaxation 1/tau, creating two stable phases
    double trap   = 10.0;
    double s_crit = 2.0;
};

template<int Dim>
class ChainEngine {
    int N_;
    std::vector<Vec<Dim>> x_, x_buf_;
    std::vector<TensorField<Dim>> G_, G_buf_;
    std::vector<Vec<Dim>> last_u_;
    std::vector<double> drive_profile_;

    std::unique_ptr<ICoupling<Dim>>    coupling_;
    std::unique_ptr<IInteraction<Dim>> interaction_;
    std::unique_ptr<IFeedback<Dim>>    feedback_;
    std::unique_ptr<IObserver<Dim>>    observer_;
    std::unique_ptr<IController<Dim>>  controller_;

    ChainParams<Dim> params_;
    double t_ = 0;
    RNG rng_;
    bool initialized_ = false;

public:
    ChainEngine(ChainParams<Dim> p,
                std::unique_ptr<ICoupling<Dim>>    coupling,
                std::unique_ptr<IInteraction<Dim>> interaction,
                std::unique_ptr<IFeedback<Dim>>    feedback,
                std::unique_ptr<IObserver<Dim>>    observer,
                std::unique_ptr<IController<Dim>>  controller)
        : N_(p.N)
        , coupling_(std::move(coupling))
        , interaction_(std::move(interaction))
        , feedback_(std::move(feedback))
        , observer_(std::move(observer))
        , controller_(std::move(controller))
        , params_(std::move(p))
    {
        x_.resize(N_); x_buf_.resize(N_);
        G_.resize(N_); G_buf_.resize(N_);
        last_u_.resize(N_);
        drive_profile_.resize(N_);
        rebuild_drive_profile();
    }

    void rebuild_drive_profile() {
        drive_profile_.resize(N_);
        for (int i = 0; i < N_; ++i) {
            double r = (N_ > 1) ? (double)i / (N_ - 1) : 0.5;
            double d = (r - params_.drive_center)
                     / std::max(params_.drive_width, 0.01);
            // Zero-base Gaussian: no drive at edges, full drive at center
            drive_profile_[i] = params_.drive_peak * std::exp(-0.5 * d * d);
        }
    }

    void reset() {
        rng_.seed(params_.seed);
        std::normal_distribution<double> noise(0, 0.02);
        for (int i = 0; i < N_; ++i) {
            x_[i] = params_.x0;
            for (int d = 0; d < Dim; ++d) x_[i](d) += noise(rng_);
            G_[i] = TensorField<Dim>();
            last_u_[i] = Vec<Dim>::Zero();
        }
        t_ = 0;
        initialized_ = true;
    }

    bool step() {
        if (!initialized_) reset();
        if (t_ > params_.t_end) return false;

        const double dt = params_.dt;
        const auto I = Mat<Dim>::Identity();

        for (int i = 0; i < N_; ++i) {
            Vec<Dim> y = observer_->observe(x_[i], G_[i], rng_);
            Vec<Dim> u = controller_->compute(t_, y, G_[i]);
            last_u_[i] = u;

            // --- state dynamics ---
            Vec<Dim> fb = feedback_->coupling(G_[i], x_[i]);
            Vec<Dim> dx = params_.A * x_[i] + fb
                        + params_.B * u + params_.w;

            // discrete lattice diffusion (Neumann BC)
            // D_x is direct neighbor coupling strength, no dr^2 scaling
            {
                Vec<Dim> lap = Vec<Dim>::Zero();
                if (i > 0)     lap += x_[i-1] - x_[i];
                if (i < N_-1)  lap += x_[i+1] - x_[i];
                dx += params_.D_x * lap;
            }
            x_buf_[i] = x_[i] + dx * dt;

            // --- tensor dynamics ---
            Mat<Dim> drive = coupling_->drive(u) * drive_profile_[i];
            Mat<Dim> inter = interaction_->compute(G_[i]);

            // Nonlinear relaxation: above s_crit, recovery slows drastically
            // Creates hysteresis without cascading through diffusion
            double s = G_[i].trace() / Dim;
            double excess = std::max(0.0, s - params_.s_crit);
            double tau_eff = params_.tau * (1.0 + params_.trap * excess * excess);
            Mat<Dim> relax = -(G_[i].G - I) / tau_eff;

            Mat<Dim> dG = drive + relax + inter;

            // discrete lattice diffusion (Neumann BC)
            {
                Mat<Dim> lap = Mat<Dim>::Zero();
                if (i > 0)     lap += G_[i-1].G - G_[i].G;
                if (i < N_-1)  lap += G_[i+1].G - G_[i].G;
                dG += params_.D_G * lap;
            }
            G_buf_[i].G = G_[i].G + dG * dt;
            G_buf_[i].symmetrize();
            G_buf_[i].clamp_eigenvalues(params_.eig_lo, params_.eig_hi);
        }

        std::swap(x_, x_buf_);
        std::swap(G_, G_buf_);
        t_ += dt;
        return true;
    }

    // --- accessors ---
    int    N()    const { return N_; }
    double t()    const { return t_; }
    bool   done() const { return t_ > params_.t_end; }

    const Vec<Dim>&         x(int i)      const { return x_[i]; }
    const TensorField<Dim>& G(int i)      const { return G_[i]; }
    const Vec<Dim>&         last_u(int i)  const { return last_u_[i]; }
    double                  drive(int i)   const { return drive_profile_[i]; }

    ChainParams<Dim>&       params()       { return params_; }
    const ChainParams<Dim>& params() const { return params_; }

    IController<Dim>&  ctrl()  { return *controller_; }
    ICoupling<Dim>&    coup()  { return *coupling_; }
    IObserver<Dim>&    obs()   { return *observer_; }
    IInteraction<Dim>& inter() { return *interaction_; }

    void swap_controller(std::unique_ptr<IController<Dim>> c) {
        controller_ = std::move(c);
    }

    double health(int i) const {
        double tr = G_[i].trace();
        double max_tr = params_.eig_hi * Dim;
        return std::clamp(1.0 - (tr - Dim) / (max_tr - Dim), 0.0, 1.0);
    }

    double anisotropy(int i) const {
        auto ev = G_[i].eigenvalues();
        double lmax = ev.maxCoeff();
        double lmin = std::max(ev.minCoeff(), 1e-6);
        return lmax / lmin - 1.0;
    }
};

} // namespace aniso
