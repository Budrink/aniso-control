#pragma once

#include "types.hpp"
#include <memory>
#include <algorithm>
#include <cmath>

namespace aniso {

template<int Dim>
struct IController {
    virtual ~IController() = default;
    virtual Vec<Dim> compute(double t, const Vec<Dim>& y, const TensorField<Dim>& G) const = 0;
    virtual void set_gain(double) {}
    virtual void set_umax(double) {}
    virtual void set_period(double) {}
    virtual void set_duty(double) {}
    virtual void set_trigger(double) {}
    virtual void set_hysteresis(double) {}
    virtual void set_anticipation(double) {}
    virtual bool is_active() const { return true; }
};

template<int Dim>
class ProportionalController : public IController<Dim> {
    double gain_, u_max_;
public:
    ProportionalController(double gain, double u_max) : gain_(gain), u_max_(u_max) {}
    void set_gain(double g) override { gain_ = g; }
    void set_umax(double u) override { u_max_ = u; }

    Vec<Dim> compute(double, const Vec<Dim>& y, const TensorField<Dim>&) const override {
        Vec<Dim> u = -gain_ * y;
        double norm = u.norm();
        if (norm > u_max_) u *= u_max_ / norm;
        return u;
    }
};

template<int Dim>
class AnisoAwareController : public IController<Dim> {
    double gain_, u_max_;
public:
    AnisoAwareController(double gain, double u_max) : gain_(gain), u_max_(u_max) {}
    void set_gain(double g) override { gain_ = g; }
    void set_umax(double u) override { u_max_ = u; }

    Vec<Dim> compute(double, const Vec<Dim>& y, const TensorField<Dim>& G) const override {
        Eigen::SelfAdjointEigenSolver<Mat<Dim>> solver(G.G);
        auto ev = solver.eigenvalues();
        auto evec = solver.eigenvectors();

        Vec<Dim> wt;
        for (int i = 0; i < Dim; ++i)
            wt(i) = 1.0 / std::max(ev(i), 0.5);
        wt /= wt.maxCoeff();

        Mat<Dim> K = gain_ * evec * wt.asDiagonal() * evec.transpose();
        Vec<Dim> u = -K * y;
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

    Vec<Dim> compute(double t, const Vec<Dim>& y, const TensorField<Dim>&) const override {
        double phase = std::fmod(t, period_);
        if (phase >= duty_ * period_) return Vec<Dim>::Zero();
        Vec<Dim> u = -gain_ * y;
        double norm = u.norm();
        if (norm > u_max_) u *= u_max_ / norm;
        return u;
    }
};

// Sleeps while system is stable; wakes with AnisoAware logic when threat detected
template<int Dim>
class EventTriggeredController : public IController<Dim> {
    double gain_, u_max_;
    double trigger_;        // |x| threshold to activate
    double hysteresis_;     // deactivation multiplier (< 1)
    double anticipation_;   // weight on dx/dt trend extrapolation
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

    Vec<Dim> compute(double, const Vec<Dim>& y, const TensorField<Dim>& G) const override {
        double xn = y.norm();
        double dx = xn - prev_xn_;
        prev_xn_ = xn;

        double threat = xn + anticipation_ * std::max(dx, 0.0);

        if (!active_ && threat > trigger_)
            active_ = true;
        if (active_ && threat < trigger_ * hysteresis_)
            active_ = false;

        if (!active_) return Vec<Dim>::Zero();

        // AnisoAware control law
        Eigen::SelfAdjointEigenSolver<Mat<Dim>> solver(G.G);
        auto ev = solver.eigenvalues();
        auto evec = solver.eigenvectors();

        Vec<Dim> wt;
        for (int i = 0; i < Dim; ++i)
            wt(i) = 1.0 / std::max(ev(i), 0.5);
        wt /= wt.maxCoeff();

        Mat<Dim> K = gain_ * evec * wt.asDiagonal() * evec.transpose();
        Vec<Dim> u = -K * y;
        double norm = u.norm();
        if (norm > u_max_) u *= u_max_ / norm;
        return u;
    }
};

} // namespace aniso
