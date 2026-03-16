#pragma once

#include "types.hpp"
#include <memory>

namespace aniso {

template<int Dim>
struct IObserver {
    virtual ~IObserver() = default;
    virtual Vec<Dim> observe(const Vec<Dim>& x, const TensorField<Dim>& G, RNG& rng) const = 0;
    virtual void set_sigma0(double) {}
    virtual void set_beta(double) {}
};

// y = x + (sigma0*I + beta*G) * eta
template<int Dim>
class AdditiveObserver : public IObserver<Dim> {
    double sigma0_, beta_;
public:
    AdditiveObserver(double sigma0, double beta) : sigma0_(sigma0), beta_(beta) {}
    void set_sigma0(double s) override { sigma0_ = s; }
    void set_beta(double b) override  { beta_ = b; }

    Vec<Dim> observe(const Vec<Dim>& x, const TensorField<Dim>& G, RNG& rng) const override {
        std::normal_distribution<double> dist(0.0, 1.0);
        Vec<Dim> eta;
        for (int i = 0; i < Dim; ++i) eta(i) = dist(rng);
        Mat<Dim> amp = sigma0_ * Mat<Dim>::Identity() + beta_ * G.G;
        return x + amp * eta;
    }
};

} // namespace aniso
