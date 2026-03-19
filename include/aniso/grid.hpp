#pragma once

#include "types.hpp"
#include "coupling.hpp"
#include "interaction.hpp"
#include "feedback.hpp"
#include "observer.hpp"
#include "controller.hpp"
#include "g_response.hpp"
#include "heater.hpp"
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
    double kappa_tau  = 20.0;  // anisotropy slows relaxation
    double noise_amp  = 0.5;   // noise base amplitude (scaled by sqrt(E))
    double D_G        = 0.0;   // G tensor diffusion between cells (barrier coupling)

    // State diffusion through G^{-1}
    double D_x = 0.1;

    // Heating spatial profile (Gaussian centered on grid)
    double heat_cx = 0.5, heat_cy = 0.5;
    double heat_rx = 0.25, heat_ry = 0.25;
    double heat_peak = 1.0;

    // Controller energy cost: fraction of |u|^2 deposited as heat
    double eta_ctrl = 0.3;

    // Wall boundary: E = 0 outside radius (cylindrical tube cross-section)
    bool wall_absorb = true;
    double wall_radius = 0.45;

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
    std::vector<double> E_, E_buf_;
    std::vector<Vec<Dim>> last_u_;
    std::vector<double> heat_profile_;
    std::vector<bool>   is_wall_;

    std::unique_ptr<ICoupling<Dim>>    coupling_;
    std::unique_ptr<IInteraction<Dim>> interaction_;
    std::unique_ptr<IFeedback<Dim>>    feedback_;
    std::unique_ptr<IObserver<Dim>>    observer_;
    std::unique_ptr<IController<Dim>>  controller_;
    std::unique_ptr<IGResponse<Dim>>   g_response_;
    std::unique_ptr<IHeater<Dim>>      heater_;

    GridParams<Dim> params_;
    double t_ = 0;
    RNG rng_;
    bool initialized_ = false;

    // Disruption tracking (updated each step)
    double wall_flux_ = 0;       // energy flux absorbed by wall this step

    int idx(int i, int j) const { return i * Ny_ + j; }

    Mat<Dim> G_inv(int k) const {
        Eigen::SelfAdjointEigenSolver<Mat<Dim>> solver(G_[k].G);
        auto ev = solver.eigenvalues();
        auto evec = solver.eigenvectors();
        Vec<Dim> ev_inv;
        for (int d = 0; d < Dim; ++d)
            ev_inv(d) = 1.0 / std::max(ev(d), 0.01);
        return evec * ev_inv.asDiagonal() * evec.transpose();
    }

    // Safe energy access: out-of-bounds and wall cells return 0
    double E_safe(int i, int j) const {
        if (i < 0 || i >= Nx_ || j < 0 || j >= Ny_) return 0.0;
        int k = idx(i, j);
        return is_wall_[k] ? 0.0 : E_[k];
    }

    // Safe state access: out-of-bounds returns center value (Neumann BC)
    const Vec<Dim>& x_safe(int i, int j, int ci, int cj) const {
        if (i < 0 || i >= Nx_ || j < 0 || j >= Ny_) return x_[idx(ci, cj)];
        return x_[idx(i, j)];
    }

public:
    GridEngine(GridParams<Dim> p,
               std::unique_ptr<ICoupling<Dim>>    coupling,
               std::unique_ptr<IInteraction<Dim>> interaction,
               std::unique_ptr<IFeedback<Dim>>    feedback,
               std::unique_ptr<IObserver<Dim>>    observer,
               std::unique_ptr<IController<Dim>>  controller,
               std::unique_ptr<IGResponse<Dim>>   g_response = nullptr,
               std::unique_ptr<IHeater<Dim>>      heater     = nullptr)
        : Nx_(p.Nx), Ny_(p.Ny), total_(p.Nx * p.Ny)
        , coupling_(std::move(coupling))
        , interaction_(std::move(interaction))
        , feedback_(std::move(feedback))
        , observer_(std::move(observer))
        , controller_(std::move(controller))
        , g_response_(std::move(g_response))
        , heater_(std::move(heater))
        , params_(std::move(p))
    {
        if (!g_response_) {
            GResponseParams rp{params_.tau_0, params_.kappa_tau,
                               params_.noise_amp, params_.eig_lo, params_.eig_hi};
            g_response_ = std::make_unique<RelaxAnisoResponse<Dim>>(rp);
        }
        if (!heater_)
            heater_ = std::make_unique<ConstantHeater<Dim>>(1.0);

        x_.resize(total_); x_buf_.resize(total_);
        G_.resize(total_); G_buf_.resize(total_);
        E_.resize(total_, 0.0); E_buf_.resize(total_, 0.0);
        last_u_.resize(total_);
        heat_profile_.resize(total_);
        is_wall_.resize(total_, false);
        rebuild_profiles();
    }

    void rebuild_profiles() {
        for (int i = 0; i < Nx_; ++i) {
            double rx = (Nx_ > 1) ? (double)i / (Nx_ - 1) : 0.5;
            for (int j = 0; j < Ny_; ++j) {
                double ry = (Ny_ > 1) ? (double)j / (Ny_ - 1) : 0.5;
                int k = idx(i, j);

                double dx = (rx - params_.heat_cx)
                          / std::max(params_.heat_rx, 0.01);
                double dy = (ry - params_.heat_cy)
                          / std::max(params_.heat_ry, 0.01);
                heat_profile_[k] =
                    params_.heat_peak * std::exp(-0.5 * (dx*dx + dy*dy));

                double cx = rx - 0.5, cy = ry - 0.5;
                double r2 = cx*cx + cy*cy;
                double R = params_.wall_radius;
                is_wall_[k] = params_.wall_absorb && (r2 > R * R);
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
        wall_flux_ = 0;
        initialized_ = true;
    }

    bool step() {
        if (!initialized_) reset();

        const double dt = params_.dt;
        const auto I = Mat<Dim>::Identity();
        std::normal_distribution<double> ndist(0.0, 1.0);
        const double sqrt_dt = std::sqrt(dt);

        // ============================================================
        //  Phase 1: Controller + state dynamics (full tensor diffusion)
        // ============================================================
        for (int i = 0; i < Nx_; ++i) {
            for (int j = 0; j < Ny_; ++j) {
                int k = idx(i, j);

                auto obs = observer_->observe(x_[k], G_[k], rng_);
                Vec<Dim> u = controller_->compute(t_, obs);
                last_u_[k] = u;

                Vec<Dim> fb = feedback_->coupling(G_[k], x_[k]);
                Vec<Dim> dxv = params_.A * x_[k] + fb
                             + params_.B * u + params_.w;

                // Full tensor diffusion: ∇·(G⁻¹ ∇x)
                Mat<Dim> gi = G_inv(k);
                double gxx = gi(0, 0);
                double gyy = (Dim >= 2) ? gi(1, 1) : 0.0;
                double gxy = (Dim >= 2) ? gi(0, 1) : 0.0;

                Vec<Dim> x_c = x_[k];
                // Axial terms (5-point stencil)
                Vec<Dim> x_flow = gxx * (x_safe(i-1,j,i,j) + x_safe(i+1,j,i,j) - 2.0*x_c)
                                + gyy * (x_safe(i,j-1,i,j) + x_safe(i,j+1,i,j) - 2.0*x_c);
                // Cross terms (diagonal neighbors)
                if constexpr (Dim >= 2) {
                    x_flow += 0.5 * gxy * (x_safe(i+1,j+1,i,j) + x_safe(i-1,j-1,i,j)
                                          - x_safe(i+1,j-1,i,j) - x_safe(i-1,j+1,i,j));
                }
                dxv += params_.D_x * x_flow;

                x_buf_[k] = x_[k] + dxv * dt;
            }
        }

        // Update heater with global metrics (for global_event, adaptive_pulsed)
        heater_->update_global(barrier_anisotropy(), confinement_ratio(),
                               total_energy());

        // ============================================================
        //  Phase 2: Energy (full tensor diffusion + wall flux tracking)
        // ============================================================
        double step_wall_flux = 0;
        for (int i = 0; i < Nx_; ++i) {
            for (int j = 0; j < Ny_; ++j) {
                int k = idx(i, j);

                if (is_wall_[k]) {
                    E_buf_[k] = 0.0;
                    continue;
                }

                double Q_heat = heater_->compute(
                    t_, heat_profile_[k], x_[k], E_[k], G_[k]);

                double u2 = last_u_[k].squaredNorm();
                double Q_ctrl = params_.eta_ctrl * u2;

                // Full tensor energy diffusion: ∇·(G⁻¹ ∇E)
                Mat<Dim> gi = G_inv(k);
                double gxx = gi(0, 0);
                double gyy = (Dim >= 2) ? gi(1, 1) : 0.0;
                double gxy = (Dim >= 2) ? gi(0, 1) : 0.0;

                double E_c = E_[k];
                double dE_flow = gxx * (E_safe(i-1,j) + E_safe(i+1,j) - 2.0*E_c)
                               + gyy * (E_safe(i,j-1) + E_safe(i,j+1) - 2.0*E_c);
                if constexpr (Dim >= 2) {
                    dE_flow += 0.5 * gxy * (E_safe(i+1,j+1) + E_safe(i-1,j-1)
                                           - E_safe(i+1,j-1) - E_safe(i-1,j+1));
                }

                // Track wall flux: energy flowing toward wall neighbors
                auto count_wall_flux = [&](int ni, int nj, double D_coeff) {
                    if (ni < 0 || ni >= Nx_ || nj < 0 || nj >= Ny_) return;
                    if (is_wall_[idx(ni, nj)])
                        step_wall_flux += params_.D_E * D_coeff * E_c * dt;
                };
                count_wall_flux(i-1, j, gxx);
                count_wall_flux(i+1, j, gxx);
                count_wall_flux(i, j-1, gyy);
                count_wall_flux(i, j+1, gyy);

                double dissip = params_.gamma_diss * E_[k];

                E_buf_[k] = E_[k] + (Q_heat + Q_ctrl
                          + params_.D_E * dE_flow - dissip) * dt;
                E_buf_[k] = std::max(E_buf_[k], 0.0);
            }
        }
        wall_flux_ = step_wall_flux;

        // ============================================================
        //  Phase 3: G dynamics — g_response + G diffusion
        // ============================================================
        for (int i = 0; i < Nx_; ++i) {
            for (int j = 0; j < Ny_; ++j) {
                int k = idx(i, j);
                double local_E = std::max(E_buf_[k], 0.0);
                Mat<Dim> drive = coupling_->drive(x_[k]) * local_E;
                G_buf_[k] = g_response_->evolve(
                    G_[k], local_E, drive, dt, sqrt_dt, rng_);
            }
        }

        // G tensor diffusion (barrier coupling between neighbors)
        if (params_.D_G > 1e-12) {
            for (int i = 0; i < Nx_; ++i) {
                for (int j = 0; j < Ny_; ++j) {
                    int k = idx(i, j);
                    Mat<Dim> lap = Mat<Dim>::Zero();
                    if (i > 0)     lap += G_[idx(i-1,j)].G - G_[k].G;
                    if (i < Nx_-1) lap += G_[idx(i+1,j)].G - G_[k].G;
                    if (j > 0)     lap += G_[idx(i,j-1)].G - G_[k].G;
                    if (j < Ny_-1) lap += G_[idx(i,j+1)].G - G_[k].G;
                    G_buf_[k].G += params_.D_G * lap * dt;
                    G_buf_[k].symmetrize();
                    G_buf_[k].clamp_eigenvalues(params_.eig_lo, params_.eig_hi);
                }
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
    double                  heat_prof(int i, int j) const { return heat_profile_[idx(i,j)]; }
    bool                    is_wall(int i, int j)  const { return is_wall_[idx(i,j)]; }

    GridParams<Dim>&       params()       { return params_; }
    const GridParams<Dim>& params() const { return params_; }

    IController<Dim>&  ctrl()  { return *controller_; }
    ICoupling<Dim>&    coup()  { return *coupling_; }
    IObserver<Dim>&    obs()   { return *observer_; }
    IGResponse<Dim>&   g_resp() { return *g_response_; }
    IHeater<Dim>&      heat()   { return *heater_; }

    void swap_controller(std::unique_ptr<IController<Dim>> c) {
        controller_ = std::move(c);
    }
    void swap_g_response(std::unique_ptr<IGResponse<Dim>> r) {
        g_response_ = std::move(r);
    }
    void swap_heater(std::unique_ptr<IHeater<Dim>> h) {
        heater_ = std::move(h);
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

    // ---- Disruption observables ----

    // Energy flux absorbed by wall in last step
    double last_wall_flux() const { return wall_flux_; }

    // Total stored energy (interior cells only)
    double total_energy() const {
        double sum = 0;
        for (int k = 0; k < total_; ++k)
            if (!is_wall_[k]) sum += E_[k];
        return sum;
    }

    // Average energy in center region (r < r_frac of wall_radius)
    double center_energy(double r_frac = 0.3) const {
        double sum = 0; int n = 0;
        double R_cut = params_.wall_radius * r_frac;
        for (int i = 0; i < Nx_; ++i)
            for (int j = 0; j < Ny_; ++j) {
                double rx = (Nx_ > 1) ? (double)i / (Nx_ - 1) - 0.5 : 0.0;
                double ry = (Ny_ > 1) ? (double)j / (Ny_ - 1) - 0.5 : 0.0;
                if (rx*rx + ry*ry < R_cut*R_cut) {
                    sum += E_[idx(i,j)]; ++n;
                }
            }
        return n > 0 ? sum / n : 0.0;
    }

    // Average energy of non-wall cells adjacent to the wall
    double edge_energy() const {
        double sum = 0; int n = 0;
        for (int i = 0; i < Nx_; ++i)
            for (int j = 0; j < Ny_; ++j) {
                int k = idx(i, j);
                if (is_wall_[k]) continue;
                bool near_wall = false;
                if (i > 0     && is_wall_[idx(i-1,j)]) near_wall = true;
                if (i < Nx_-1 && is_wall_[idx(i+1,j)]) near_wall = true;
                if (j > 0     && is_wall_[idx(i,j-1)]) near_wall = true;
                if (j < Ny_-1 && is_wall_[idx(i,j+1)]) near_wall = true;
                if (near_wall) { sum += E_[k]; ++n; }
            }
        return n > 0 ? sum / n : 0.0;
    }

    // Confinement quality: center_E / edge_E (high = good confinement)
    double confinement_ratio() const {
        double ce = center_energy();
        double ee = edge_energy();
        return ce / std::max(ee, 1e-8);
    }

    // Energy confinement time: total_E / wall_flux (in time units)
    double confinement_time() const {
        return total_energy() / std::max(wall_flux_ / params_.dt, 1e-12);
    }

    // Average anisotropy in an annular ring (r_lo..r_hi as fraction of wall_radius)
    double barrier_anisotropy(double r_lo = 0.5, double r_hi = 0.85) const {
        double sum = 0; int n = 0;
        double R = params_.wall_radius;
        double R_lo = R * r_lo, R_hi = R * r_hi;
        for (int i = 0; i < Nx_; ++i)
            for (int j = 0; j < Ny_; ++j) {
                int k = idx(i, j);
                if (is_wall_[k]) continue;
                double rx = (Nx_ > 1) ? (double)i / (Nx_ - 1) - 0.5 : 0.0;
                double ry = (Ny_ > 1) ? (double)j / (Ny_ - 1) - 0.5 : 0.0;
                double r = std::sqrt(rx*rx + ry*ry);
                if (r >= R_lo && r <= R_hi) {
                    sum += anisotropy(i, j); ++n;
                }
            }
        return n > 0 ? sum / n : 0.0;
    }
};

} // namespace aniso
