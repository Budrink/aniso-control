#pragma once

#include "types.hpp"
#include <memory>
#include <algorithm>
#include <cmath>

namespace aniso {

template<int Dim>
struct IController {
    virtual ~IController() = default;
    virtual Vec<Dim> compute(double t, const Observation<Dim>& obs) const = 0;
    virtual void set_gain(double) {}
    virtual void set_umax(double) {}
    virtual void set_period(double) {}
    virtual void set_duty(double) {}
    virtual void set_trigger(double) {}
    virtual void set_hysteresis(double) {}
    virtual void set_anticipation(double) {}
    virtual bool is_active() const { return true; }
};

// u = -K * y  (ignores G and F entirely)
template<int Dim>
class ProportionalController : public IController<Dim> {
    double gain_, u_max_;
public:
    ProportionalController(double gain, double u_max) : gain_(gain), u_max_(u_max) {}
    void set_gain(double g) override { gain_ = g; }
    void set_umax(double u) override { u_max_ = u; }

    Vec<Dim> compute(double, const Observation<Dim>& obs) const override {
        Vec<Dim> u = -gain_ * obs.y;
        double norm = u.norm();
        if (norm > u_max_) u *= u_max_ / norm;
        return u;
    }
};

// K weighted by G_hat eigenvectors, scaled by Fisher information F
// Pushes harder along well-observed (high F) directions
template<int Dim>
class AnisoAwareController : public IController<Dim> {
    double gain_, u_max_;
public:
    AnisoAwareController(double gain, double u_max) : gain_(gain), u_max_(u_max) {}
    void set_gain(double g) override { gain_ = g; }
    void set_umax(double u) override { u_max_ = u; }

    Vec<Dim> compute(double, const Observation<Dim>& obs) const override {
        Eigen::SelfAdjointEigenSolver<Mat<Dim>> solver(obs.G_hat.G);
        auto ev = solver.eigenvalues();
        auto evec = solver.eigenvectors();

        // Weight by Fisher information eigenvalues (projected onto G_hat eigenbasis)
        // F is large where observation is good → push harder there
        Eigen::SelfAdjointEigenSolver<Mat<Dim>> f_solver(obs.F);
        auto f_ev = f_solver.eigenvalues();
        auto f_evec = f_solver.eigenvectors();

        // Project F into G_hat eigenbasis: effective weight per G-direction
        // Floor at 0.25 to maintain stabilizing gain even in poorly-observed directions
        Vec<Dim> wt;
        for (int i = 0; i < Dim; ++i) {
            Vec<Dim> gi = evec.col(i);
            double fi = (gi.transpose() * obs.F * gi)(0, 0);
            double fi_norm = fi / std::max(f_ev.maxCoeff(), 1e-6);
            wt(i) = std::max(fi_norm, 0.25);
        }
        wt /= wt.maxCoeff();

        Mat<Dim> K = gain_ * evec * wt.asDiagonal() * evec.transpose();
        Vec<Dim> u = -K * obs.y;
        double norm = u.norm();
        if (norm > u_max_) u *= u_max_ / norm;
        return u;
    }
};

template<int Dim>
class PulsedController : public IController<Dim> {
    double gain_, u_max_, period_, duty_;
public:
    PulsedController(double gain, double u_max, double period, double duty)
        : gain_(gain), u_max_(u_max), period_(period), duty_(duty) {}
    void set_gain(double g) override { gain_ = g; }
    void set_umax(double u) override { u_max_ = u; }
    void set_period(double p) override { period_ = std::max(p, 0.01); }
    void set_duty(double d)   override { duty_ = std::clamp(d, 0.0, 1.0); }

    Vec<Dim> compute(double t, const Observation<Dim>& obs) const override {
        double phase = std::fmod(t, period_);
        if (phase >= duty_ * period_) return Vec<Dim>::Zero();
        Vec<Dim> u = -gain_ * obs.y;
        double norm = u.norm();
        if (norm > u_max_) u *= u_max_ / norm;
        return u;
    }
};

// Sleeps while system is stable; wakes with Fisher-weighted AnisoAware logic
template<int Dim>
class EventTriggeredController : public IController<Dim> {
    double gain_, u_max_;
    double trigger_;
    double hysteresis_;
    double anticipation_;
    mutable bool   active_ = false;
    mutable double prev_xn_ = 0.0;

public:
    EventTriggeredController(double gain, double u_max,
                             double trigger, double hysteresis, double anticipation)
        : gain_(gain), u_max_(u_max),
          trigger_(trigger), hysteresis_(hysteresis), anticipation_(anticipation) {}

    void set_gain(double g)          override { gain_ = g; }
    void set_umax(double u)          override { u_max_ = u; }
    void set_trigger(double t)       override { trigger_ = std::max(t, 0.001); }
    void set_hysteresis(double h)    override { hysteresis_ = std::clamp(h, 0.1, 1.0); }
    void set_anticipation(double a)  override { anticipation_ = std::max(a, 0.0); }
    bool is_active() const           override { return active_; }

    Vec<Dim> compute(double, const Observation<Dim>& obs) const override {
        double xn = obs.y.norm();
        double dx = xn - prev_xn_;
        prev_xn_ = xn;

        double threat = xn + anticipation_ * std::max(dx, 0.0);

        if (!active_ && threat > trigger_)
            active_ = true;
        if (active_ && threat < trigger_ * hysteresis_)
            active_ = false;

        if (!active_) return Vec<Dim>::Zero();

        // Fisher-weighted AnisoAware control using estimated G
        Eigen::SelfAdjointEigenSolver<Mat<Dim>> solver(obs.G_hat.G);
        auto ev = solver.eigenvalues();
        auto evec = solver.eigenvectors();

        Eigen::SelfAdjointEigenSolver<Mat<Dim>> f_solver(obs.F);
        auto f_ev = f_solver.eigenvalues();

        Vec<Dim> wt;
        for (int i = 0; i < Dim; ++i) {
            Vec<Dim> gi = evec.col(i);
            double fi = (gi.transpose() * obs.F * gi)(0, 0);
            double fi_norm = fi / std::max(f_ev.maxCoeff(), 1e-6);
            wt(i) = std::max(fi_norm, 0.25);
        }
        wt /= wt.maxCoeff();

        Mat<Dim> K = gain_ * evec * wt.asDiagonal() * evec.transpose();
        Vec<Dim> u = -K * obs.y;
        double norm = u.norm();
        if (norm > u_max_) u *= u_max_ / norm;
        return u;
    }
};

// u = -Kp*y - Ki*integral(y) - Kd*dy/dt
// Classic PID on observed state. D-term amplifies observation noise.
template<int Dim>
class PIDController : public IController<Dim> {
    double kp_, ki_, kd_, u_max_, dt_;
    mutable Vec<Dim> integral_ = Vec<Dim>::Zero();
    mutable Vec<Dim> prev_y_   = Vec<Dim>::Zero();
    mutable bool first_ = true;
public:
    PIDController(double kp, double ki, double kd, double u_max, double dt)
        : kp_(kp), ki_(ki), kd_(kd), u_max_(u_max), dt_(std::max(dt, 1e-6)) {}

    void set_gain(double g) override { kp_ = g; }
    void set_umax(double u) override { u_max_ = u; }

    Vec<Dim> compute(double, const Observation<Dim>& obs) const override {
        Vec<Dim> y = obs.y;

        // Anti-windup: clamp integral
        integral_ += y * dt_;
        double int_norm = integral_.norm();
        double int_max = u_max_ / std::max(ki_, 1e-6);
        if (int_norm > int_max)
            integral_ *= int_max / int_norm;

        Vec<Dim> deriv = Vec<Dim>::Zero();
        if (!first_)
            deriv = (y - prev_y_) / dt_;
        first_ = false;
        prev_y_ = y;

        Vec<Dim> u = -(kp_ * y + ki_ * integral_ + kd_ * deriv);
        double norm = u.norm();
        if (norm > u_max_) u *= u_max_ / norm;
        return u;
    }
};

} // namespace aniso
