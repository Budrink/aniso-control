#pragma once

#include "types.hpp"
#include <cmath>
#include <algorithm>
#include <string>

namespace aniso {

// Heater observes local state and decides how much energy to inject.
// Returns scalar power density at each grid point.
template<int Dim>
struct IHeater {
    virtual ~IHeater() = default;

    // Called once per step with grid-wide metrics before per-cell compute().
    virtual void update_global(double barrier_aniso, double confinement,
                               double total_E) {
        (void)barrier_aniso; (void)confinement; (void)total_E;
    }

    // Compute local heating power given position info, local state,
    // energy, tensor, and simulation time.
    virtual double compute(double t, double base_profile,
                           const Vec<Dim>& x, double E,
                           const TensorField<Dim>& G) const = 0;

    virtual void set_power(double) {}
    virtual void set_period(double) {}
    virtual void set_duty(double) {}
    virtual void set_trigger(double) {}
    virtual void set_hysteresis(double) {}
    virtual bool is_active() const { return true; }
    virtual double current_duty() const { return 1.0; }
    virtual std::string type_name() const = 0;
};

// ---------------------------------------------------------------------------
//  1. ConstantHeater — always on at full power × spatial profile
// ---------------------------------------------------------------------------
template<int Dim>
class ConstantHeater : public IHeater<Dim> {
    double power_;
public:
    explicit ConstantHeater(double power) : power_(power) {}
    void set_power(double p) override { power_ = p; }
    std::string type_name() const override { return "constant"; }

    double compute(double, double base_profile,
                   const Vec<Dim>&, double, const TensorField<Dim>&) const override {
        return power_ * base_profile;
    }
};

// ---------------------------------------------------------------------------
//  2. PulsedHeater — periodic on/off with adjustable period and duty
// ---------------------------------------------------------------------------
template<int Dim>
class PulsedHeater : public IHeater<Dim> {
    double power_, period_, duty_;
public:
    PulsedHeater(double power, double period, double duty)
        : power_(power), period_(std::max(period, 0.01)), duty_(std::clamp(duty, 0.0, 1.0)) {}
    void set_power(double p)  override { power_ = p; }
    void set_period(double p) override { period_ = std::max(p, 0.01); }
    void set_duty(double d)   override { duty_ = std::clamp(d, 0.0, 1.0); }
    std::string type_name() const override { return "pulsed"; }

    double compute(double t, double base_profile,
                   const Vec<Dim>&, double, const TensorField<Dim>&) const override {
        double phase = std::fmod(t, period_);
        if (phase >= duty_ * period_) return 0.0;
        return power_ * base_profile;
    }
};

// ---------------------------------------------------------------------------
//  3. EventDrivenHeater — reacts to barrier health
//     Heats when avg barrier anisotropy drops below trigger.
//     Stops when anisotropy rises above trigger * hysteresis.
//     Uses local G anisotropy as proxy for barrier strength.
// ---------------------------------------------------------------------------
template<int Dim>
class EventDrivenHeater : public IHeater<Dim> {
    double power_;
    double trigger_;      // anisotropy threshold to START heating
    double hysteresis_;   // multiplier (>1) — stop when aniso > trigger * hyst
    mutable bool active_ = true;
public:
    EventDrivenHeater(double power, double trigger, double hysteresis)
        : power_(power), trigger_(trigger),
          hysteresis_(std::max(hysteresis, 1.01)) {}
    void set_power(double p)      override { power_ = p; }
    void set_trigger(double t)    override { trigger_ = std::max(t, 0.001); }
    void set_hysteresis(double h) override { hysteresis_ = std::max(h, 1.01); }
    bool is_active() const        override { return active_; }
    std::string type_name() const override { return "event_driven"; }

    double compute(double, double base_profile,
                   const Vec<Dim>&, double,
                   const TensorField<Dim>& G) const override {
        // Local anisotropy as barrier strength indicator
        Eigen::SelfAdjointEigenSolver<Mat<Dim>> solver(G.G);
        auto ev = solver.eigenvalues();
        double lmax = ev.maxCoeff();
        double lmin = std::max(ev.minCoeff(), 1e-6);
        double aniso = lmax / lmin - 1.0;

        if (active_ && aniso > trigger_ * hysteresis_)
            active_ = false;
        if (!active_ && aniso < trigger_)
            active_ = true;

        if (!active_) return 0.0;
        return power_ * base_profile;
    }
};

// ---------------------------------------------------------------------------
//  4. AnisoAwareHeater — heats more where G is isotropic (barrier weak),
//     less where G is already anisotropic (barrier strong).
//     Avoids overheating healthy barrier sections.
// ---------------------------------------------------------------------------
template<int Dim>
class AnisoAwareHeater : public IHeater<Dim> {
    double power_;
public:
    explicit AnisoAwareHeater(double power) : power_(power) {}
    void set_power(double p) override { power_ = p; }
    std::string type_name() const override { return "aniso_aware"; }

    double compute(double, double base_profile,
                   const Vec<Dim>&, double,
                   const TensorField<Dim>& G) const override {
        Eigen::SelfAdjointEigenSolver<Mat<Dim>> solver(G.G);
        auto ev = solver.eigenvalues();
        double lmax = ev.maxCoeff();
        double lmin = std::max(ev.minCoeff(), 1e-6);
        double aniso = lmax / lmin - 1.0;
        // Weight: heat more where barrier is weak (low anisotropy)
        double w = 1.0 / (1.0 + aniso);
        return power_ * base_profile * w;
    }
};

// ---------------------------------------------------------------------------
//  5. GlobalEventHeater — on/off based on grid-wide barrier health
//     Unlike EventDrivenHeater (per-cell), this uses averaged metrics
//     from update_global() to make a single global decision.
// ---------------------------------------------------------------------------
template<int Dim>
class GlobalEventHeater : public IHeater<Dim> {
    double power_;
    double trigger_;      // barrier aniso threshold to START heating
    double hysteresis_;   // stop when barrier > trigger * hysteresis
    mutable bool active_ = true;
    double barrier_ = 0;
public:
    GlobalEventHeater(double power, double trigger, double hysteresis)
        : power_(power), trigger_(trigger),
          hysteresis_(std::max(hysteresis, 1.01)) {}

    void set_power(double p)      override { power_ = p; }
    void set_trigger(double t)    override { trigger_ = std::max(t, 0.001); }
    void set_hysteresis(double h) override { hysteresis_ = std::max(h, 1.01); }
    bool is_active() const        override { return active_; }
    std::string type_name() const override { return "global_event"; }

    void update_global(double barrier_aniso, double, double) override {
        barrier_ = barrier_aniso;
        if (active_ && barrier_ > trigger_ * hysteresis_)
            active_ = false;
        if (!active_ && barrier_ < trigger_)
            active_ = true;
    }

    double compute(double, double base_profile,
                   const Vec<Dim>&, double,
                   const TensorField<Dim>&) const override {
        if (!active_) return 0.0;
        return power_ * base_profile;
    }
};

// ---------------------------------------------------------------------------
//  6. AdaptivePulsedHeater — duty cycle adapts to barrier health
//     Healthy barrier → low duty (save energy, allow cooling).
//     Weak barrier → high duty (inject energy to rebuild).
//     duty = clamp(duty_min + (1 - duty_min) * (1 - barrier/barrier_target), duty_min, 1)
// ---------------------------------------------------------------------------
template<int Dim>
class AdaptivePulsedHeater : public IHeater<Dim> {
    double power_, period_;
    double duty_min_;        // floor duty when barrier is healthy
    double barrier_target_;  // barrier aniso above which duty is minimal
    mutable double duty_ = 0.5;
public:
    AdaptivePulsedHeater(double power, double period,
                         double duty_min, double barrier_target)
        : power_(power), period_(std::max(period, 0.01)),
          duty_min_(std::clamp(duty_min, 0.05, 0.95)),
          barrier_target_(std::max(barrier_target, 0.1)) {}

    void set_power(double p)  override { power_ = p; }
    void set_period(double p) override { period_ = std::max(p, 0.01); }
    void set_duty(double d)   override { duty_min_ = std::clamp(d, 0.05, 0.95); }
    void set_trigger(double t) override { barrier_target_ = std::max(t, 0.1); }
    double current_duty() const override { return duty_; }
    std::string type_name() const override { return "adaptive_pulsed"; }

    void update_global(double barrier_aniso, double, double) override {
        double health = std::clamp(barrier_aniso / barrier_target_, 0.0, 1.0);
        // health=1 → barrier healthy → low duty;  health=0 → weak → full duty
        duty_ = std::clamp(duty_min_ + (1.0 - duty_min_) * (1.0 - health),
                           duty_min_, 1.0);
    }

    double compute(double t, double base_profile,
                   const Vec<Dim>&, double,
                   const TensorField<Dim>&) const override {
        double phase = std::fmod(t, period_);
        if (phase >= duty_ * period_) return 0.0;
        return power_ * base_profile;
    }
};

} // namespace aniso
