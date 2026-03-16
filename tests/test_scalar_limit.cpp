#include <aniso/config.hpp>
#include <iostream>
#include <cmath>
#include <cassert>

// Manual scalar simulation of:
//   dx/dt = a*x + b*u + w
//   dG/dt = alpha*|u|*I - (G-1)/tau
//   u = clip(-gain*y, u_max)
//   y = x + noise (disabled for comparison: sigma0=0, beta=0)
//
// In dim=1, G is a scalar, Q=0 (traceless of 1x1 is zero), feedback=none.

static double scalar_sim(double a, double b, double w, double x0,
                         double alpha, double tau,
                         double gain, double u_max,
                         double dt, double t_end) {
    double x = x0;
    double G = 1.0;
    int steps = static_cast<int>(t_end / dt + 0.5);

    for (int i = 0; i < steps; ++i) {
        // No noise in comparison
        double y = x;
        double u = -gain * y;
        if (u > u_max)  u = u_max;
        if (u < -u_max) u = -u_max;

        double dx = a * x + b * u + w;
        x += dx * dt;

        double dG = alpha * std::abs(u) - (G - 1.0) / tau;
        G += dG * dt;
        if (G < 0.3) G = 0.3;
        if (G > 5.0) G = 5.0;
    }
    return x;
}

int main() {
    // Build engine from scalar_limit.yaml but with zero noise
    YAML::Node cfg = YAML::LoadFile("configs/scalar_limit.yaml");
    cfg["observation"]["sigma0"] = 0.0;
    cfg["observation"]["beta"]   = 0.0;

    auto engine = aniso::build_engine<1>(cfg);
    auto result = engine.run();

    double x_engine = result.recorder.data().back().x(0);

    double x_scalar = scalar_sim(
        -0.2, 1.0, 0.5, 1.0,  // a, b, w, x0
        0.5, 5.0,              // alpha, tau
        1.5, 3.0,              // gain, u_max
        0.01, 60.0             // dt, t_end
    );

    double err = std::abs(x_engine - x_scalar);
    std::cout << "Engine x_final = " << x_engine << '\n';
    std::cout << "Scalar x_final = " << x_scalar << '\n';
    std::cout << "Absolute error = " << err << '\n';

    // Tolerance: should be essentially identical (both Euler, both deterministic with sigma=0)
    constexpr double tol = 1e-10;
    if (err > tol) {
        std::cerr << "FAIL: scalar limit mismatch (err=" << err << " > " << tol << ")\n";
        return 1;
    }

    std::cout << "PASS: scalar limit matches.\n";
    return 0;
}
