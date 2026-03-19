#pragma once

#include "types.hpp"
#include <cmath>
#include <algorithm>
#include <string>

namespace aniso {

// Parameters shared across G-response models
struct GResponseParams {
    double tau_0     = 1.0;
    double kappa     = 20.0;   // model-specific sensitivity parameter
    double noise_amp = 0.5;
    double eig_lo    = 0.3;
    double eig_hi    = 5.0;
};

template<int Dim>
struct IGResponse {
    virtual ~IGResponse() = default;

    virtual TensorField<Dim> evolve(
        const TensorField<Dim>& G_cur,
        double E,
        const Mat<Dim>& drive,
        double dt, double sqrt_dt,
        RNG& rng) const = 0;

    virtual void set_param(const std::string& name, double val) = 0;
    virtual std::string type_name() const = 0;
};

// ---------------------------------------------------------------------------
//  1. RelaxAniso — current model
//     tau_eff = tau_0 * (1 + kappa * aniso^2)
//     Anisotropy slows its own relaxation → self-sustaining barriers
// ---------------------------------------------------------------------------
template<int Dim>
class RelaxAnisoResponse : public IGResponse<Dim> {
    mutable GResponseParams p_;
public:
    explicit RelaxAnisoResponse(GResponseParams p) : p_(std::move(p)) {}

    void set_param(const std::string& name, double val) override {
        if (name == "tau")   p_.tau_0     = val;
        if (name == "kappa") p_.kappa     = val;
        if (name == "noise") p_.noise_amp = val;
    }

    std::string type_name() const override { return "relax_aniso"; }

    TensorField<Dim> evolve(
        const TensorField<Dim>& G_cur, double E,
        const Mat<Dim>& drive, double dt, double sqrt_dt,
        RNG& rng) const override
    {
        const auto I = Mat<Dim>::Identity();
        Mat<Dim> Q = G_cur.traceless();
        double Q_norm = std::sqrt((Q.transpose() * Q).trace());
        double s = G_cur.trace() / Dim;
        double aniso = Q_norm / std::max(s, 0.1);
        double tau_eff = p_.tau_0 * (1.0 + p_.kappa * aniso * aniso);
        Mat<Dim> relax = -(G_cur.G - I) / tau_eff;

        TensorField<Dim> G_new;
        G_new.G = G_cur.G + (drive + relax) * dt;
        add_noise(G_new, E, sqrt_dt, rng);
        G_new.symmetrize();
        G_new.clamp_eigenvalues(p_.eig_lo, p_.eig_hi);
        return G_new;
    }

private:
    void add_noise(TensorField<Dim>& G, double E, double sqrt_dt, RNG& rng) const {
        if (p_.noise_amp < 1e-8 || E < 1e-8) return;
        std::normal_distribution<double> nd(0.0, 1.0);
        Mat<Dim> xi = Mat<Dim>::Zero();
        for (int a = 0; a < Dim; ++a)
            for (int b = a; b < Dim; ++b) {
                double v = nd(rng);
                xi(a, b) = v;
                xi(b, a) = v;
            }
        double tr = xi.trace();
        for (int a = 0; a < Dim; ++a) xi(a, a) -= tr / Dim;
        G.G += p_.noise_amp * std::sqrt(E) * sqrt_dt * xi;
    }
};

// ---------------------------------------------------------------------------
//  2. RelaxEnergy — hot zones relax slower
//     tau_eff = tau_0 * (1 + kappa * E)
//     High energy "freezes" deformed G, low energy allows recovery
// ---------------------------------------------------------------------------
template<int Dim>
class RelaxEnergyResponse : public IGResponse<Dim> {
    mutable GResponseParams p_;
public:
    explicit RelaxEnergyResponse(GResponseParams p) : p_(std::move(p)) {}

    void set_param(const std::string& name, double val) override {
        if (name == "tau")   p_.tau_0     = val;
        if (name == "kappa") p_.kappa     = val;
        if (name == "noise") p_.noise_amp = val;
    }

    std::string type_name() const override { return "relax_energy"; }

    TensorField<Dim> evolve(
        const TensorField<Dim>& G_cur, double E,
        const Mat<Dim>& drive, double dt, double sqrt_dt,
        RNG& rng) const override
    {
        const auto I = Mat<Dim>::Identity();
        double tau_eff = p_.tau_0 * (1.0 + p_.kappa * std::max(E, 0.0));
        Mat<Dim> relax = -(G_cur.G - I) / tau_eff;

        TensorField<Dim> G_new;
        G_new.G = G_cur.G + (drive + relax) * dt;
        add_noise(G_new, E, sqrt_dt, rng);
        G_new.symmetrize();
        G_new.clamp_eigenvalues(p_.eig_lo, p_.eig_hi);
        return G_new;
    }

private:
    void add_noise(TensorField<Dim>& G, double E, double sqrt_dt, RNG& rng) const {
        if (p_.noise_amp < 1e-8 || E < 1e-8) return;
        std::normal_distribution<double> nd(0.0, 1.0);
        Mat<Dim> xi = Mat<Dim>::Zero();
        for (int a = 0; a < Dim; ++a)
            for (int b = a; b < Dim; ++b) {
                double v = nd(rng);
                xi(a, b) = v;
                xi(b, a) = v;
            }
        double tr = xi.trace();
        for (int a = 0; a < Dim; ++a) xi(a, a) -= tr / Dim;
        G.G += p_.noise_amp * std::sqrt(E) * sqrt_dt * xi;
    }
};

// ---------------------------------------------------------------------------
//  3. Melt — energy directly pushes G toward isotropy
//     dG = drive + alpha_melt * E * (I - G) + relax
//     At high E the "melt" term dominates, erasing structure
//     kappa here plays the role of alpha_melt
// ---------------------------------------------------------------------------
template<int Dim>
class MeltResponse : public IGResponse<Dim> {
    mutable GResponseParams p_;
public:
    explicit MeltResponse(GResponseParams p) : p_(std::move(p)) {}

    void set_param(const std::string& name, double val) override {
        if (name == "tau")   p_.tau_0     = val;
        if (name == "kappa") p_.kappa     = val;
        if (name == "noise") p_.noise_amp = val;
    }

    std::string type_name() const override { return "melt"; }

    TensorField<Dim> evolve(
        const TensorField<Dim>& G_cur, double E,
        const Mat<Dim>& drive, double dt, double sqrt_dt,
        RNG& rng) const override
    {
        const auto I = Mat<Dim>::Identity();
        double Ec = std::max(E, 0.0);

        // Baseline relaxation (constant tau)
        Mat<Dim> relax = -(G_cur.G - I) / p_.tau_0;

        // Melt term: energy pushes G → I (destroys structure)
        Mat<Dim> melt = p_.kappa * Ec * (I - G_cur.G);

        TensorField<Dim> G_new;
        G_new.G = G_cur.G + (drive + relax + melt) * dt;
        add_noise(G_new, E, sqrt_dt, rng);
        G_new.symmetrize();
        G_new.clamp_eigenvalues(p_.eig_lo, p_.eig_hi);
        return G_new;
    }

private:
    void add_noise(TensorField<Dim>& G, double E, double sqrt_dt, RNG& rng) const {
        if (p_.noise_amp < 1e-8 || E < 1e-8) return;
        std::normal_distribution<double> nd(0.0, 1.0);
        Mat<Dim> xi = Mat<Dim>::Zero();
        for (int a = 0; a < Dim; ++a)
            for (int b = a; b < Dim; ++b) {
                double v = nd(rng);
                xi(a, b) = v;
                xi(b, a) = v;
            }
        double tr = xi.trace();
        for (int a = 0; a < Dim; ++a) xi(a, a) -= tr / Dim;
        G.G += p_.noise_amp * std::sqrt(E) * sqrt_dt * xi;
    }
};

// ---------------------------------------------------------------------------
//  4. LandauEnergy — phase transition driven by local energy
//     mu_eff = kappa * (E - E_c),  E_c = 1/kappa (so onset at E ~ 1/kappa)
//     Below E_c: anisotropy suppressed. Above E_c: anisotropy grows.
//     Saturates via cubic: dG_landau = mu_eff * Q - nu * |Q|^2 * Q
//     Combined with baseline relaxation.
//     tau_0 = base relaxation, kappa = Landau coupling strength
// ---------------------------------------------------------------------------
template<int Dim>
class LandauEnergyResponse : public IGResponse<Dim> {
    mutable GResponseParams p_;
    double nu_ = 0.5;
public:
    explicit LandauEnergyResponse(GResponseParams p, double nu = 0.5)
        : p_(std::move(p)), nu_(nu) {}

    void set_param(const std::string& name, double val) override {
        if (name == "tau")   p_.tau_0     = val;
        if (name == "kappa") p_.kappa     = val;
        if (name == "noise") p_.noise_amp = val;
        if (name == "nu")    nu_          = val;
    }

    std::string type_name() const override { return "landau_energy"; }

    TensorField<Dim> evolve(
        const TensorField<Dim>& G_cur, double E,
        const Mat<Dim>& drive, double dt, double sqrt_dt,
        RNG& rng) const override
    {
        const auto I = Mat<Dim>::Identity();
        double Ec = std::max(E, 0.0);

        // Baseline relaxation
        Mat<Dim> relax = -(G_cur.G - I) / p_.tau_0;

        // Landau self-interaction: E controls the transition
        Mat<Dim> Q = G_cur.traceless();
        double E_crit = (p_.kappa > 1e-6) ? 1.0 / p_.kappa : 1e6;
        double mu_eff = p_.kappa * (Ec - E_crit);
        double Q_sq = (Q.transpose() * Q).trace();
        Mat<Dim> landau = mu_eff * Q - nu_ * Q_sq * Q;

        TensorField<Dim> G_new;
        G_new.G = G_cur.G + (drive + relax + landau) * dt;
        add_noise(G_new, E, sqrt_dt, rng);
        G_new.symmetrize();
        G_new.clamp_eigenvalues(p_.eig_lo, p_.eig_hi);
        return G_new;
    }

private:
    void add_noise(TensorField<Dim>& G, double E, double sqrt_dt, RNG& rng) const {
        if (p_.noise_amp < 1e-8 || E < 1e-8) return;
        std::normal_distribution<double> nd(0.0, 1.0);
        Mat<Dim> xi = Mat<Dim>::Zero();
        for (int a = 0; a < Dim; ++a)
            for (int b = a; b < Dim; ++b) {
                double v = nd(rng);
                xi(a, b) = v;
                xi(b, a) = v;
            }
        double tr = xi.trace();
        for (int a = 0; a < Dim; ++a) xi(a, a) -= tr / Dim;
        G.G += p_.noise_amp * std::sqrt(E) * sqrt_dt * xi;
    }
};

} // namespace aniso
