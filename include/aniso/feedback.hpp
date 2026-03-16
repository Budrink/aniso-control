#pragma once

#include "types.hpp"
#include <memory>

namespace aniso {

template<int Dim>
struct IFeedback {
    virtual ~IFeedback() = default;
    virtual Vec<Dim> coupling(const TensorField<Dim>& G, const Vec<Dim>& x) const = 0;
};

template<int Dim>
class NoFeedback : public IFeedback<Dim> {
public:
    Vec<Dim> coupling(const TensorField<Dim>&, const Vec<Dim>&) const override {
        return Vec<Dim>::Zero();
    }
};

// kappa * Q * x  where Q = G - tr(G)/Dim * I
template<int Dim>
class TracelessFeedback : public IFeedback<Dim> {
    double kappa_;
public:
    explicit TracelessFeedback(double kappa) : kappa_(kappa) {}

    Vec<Dim> coupling(const TensorField<Dim>& G, const Vec<Dim>& x) const override {
        return kappa_ * G.traceless() * x;
    }
};

// kappa * G * x  (full tensor, not just traceless)
template<int Dim>
class FullFeedback : public IFeedback<Dim> {
    double kappa_;
public:
    explicit FullFeedback(double kappa) : kappa_(kappa) {}

    Vec<Dim> coupling(const TensorField<Dim>& G, const Vec<Dim>& x) const override {
        return kappa_ * G.G * x;
    }
};

} // namespace aniso
