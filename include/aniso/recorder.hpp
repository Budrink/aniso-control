#pragma once

#include "types.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <limits>

namespace aniso {

template<int Dim>
struct Snapshot {
    double t;
    Vec<Dim> x;
    Vec<Dim> u;
    Vec<Dim> G_eigvals;
    double trG;
    double Q_norm;
};

template<int Dim>
struct Metrics {
    // Quality
    double mean_error;
    double peak_error;
    double settling_time;

    // Cost
    double mean_effort;
    double peak_effort;
    double saturation_frac;

    // Degradation
    double mean_aniso;
    double final_aniso;
    double mean_trG;
    double phase_transition_time;   // -1 if no transition

    // Composite
    double efficiency;              // mean_error * mean_effort (lower = better)

    // Stability
    bool   breakdown = false;       // true if error diverged in the tail
    double tail_error = 0.0;        // mean |x| over last 20% of data
};

template<int Dim>
class Recorder {
    std::vector<Snapshot<Dim>> data_;

public:
    void push(double t, const Vec<Dim>& x, const Vec<Dim>& u, const TensorField<Dim>& G) {
        Snapshot<Dim> snap;
        snap.t = t;
        snap.x = x;
        snap.u = u;
        snap.G_eigvals = G.eigenvalues();
        snap.trG = G.trace();
        snap.Q_norm = std::sqrt(G.traceless_norm_sq());
        data_.push_back(snap);
    }

    void clear() { data_.clear(); }
    const std::vector<Snapshot<Dim>>& data() const { return data_; }
    size_t size() const { return data_.size(); }

    Metrics<Dim> compute_metrics(double warmup = 5.0, double u_max = 3.0,
                                  double aniso_threshold = 2.0) const {
        Metrics<Dim> m{};
        if (data_.empty()) return m;

        int count = 0;
        double sum_err = 0, sum_eff = 0, sum_aniso = 0, sum_trace = 0;
        double peak_err = 0, peak_eff = 0;
        int sat_count = 0;
        m.phase_transition_time = -1.0;
        m.settling_time = -1.0;

        double settling_tol = 0.0;
        // First pass: compute mean error for settling time reference
        for (auto& s : data_) {
            if (s.t < warmup) continue;
            settling_tol += s.x.norm();
            ++count;
        }
        if (count > 0) settling_tol = (settling_tol / count) * 1.5;
        else settling_tol = 1.0;

        count = 0;
        bool settled = false;

        for (size_t i = 0; i < data_.size(); ++i) {
            auto& s = data_[i];
            double x_norm = s.x.norm();
            double u_sq = s.u.squaredNorm();
            double u_norm = std::sqrt(u_sq);

            // Settling time: first time |x| drops below 1.5× mean and stays
            if (!settled && s.t > 1.0 && x_norm < settling_tol) {
                bool stays = true;
                for (size_t j = i; j < std::min(i + 100, data_.size()); ++j) {
                    if (data_[j].x.norm() > settling_tol * 2.0) { stays = false; break; }
                }
                if (stays) { m.settling_time = s.t; settled = true; }
            }

            // Phase transition detection: aniso crosses threshold
            if (m.phase_transition_time < 0) {
                double lam_max = s.G_eigvals.maxCoeff();
                double lam_min = std::max(s.G_eigvals.minCoeff(), 1e-6);
                if (lam_max / lam_min > aniso_threshold)
                    m.phase_transition_time = s.t;
            }

            if (s.t < warmup) continue;

            sum_err += x_norm;
            sum_eff += u_sq;
            peak_err = std::max(peak_err, x_norm);
            peak_eff = std::max(peak_eff, u_norm);
            if (u_norm > u_max * 0.99) ++sat_count;

            double lam_max = s.G_eigvals.maxCoeff();
            double lam_min = std::max(s.G_eigvals.minCoeff(), 1e-6);
            sum_aniso += lam_max / lam_min;
            sum_trace += s.trG;
            ++count;
        }

        if (count == 0) return m;
        double n = static_cast<double>(count);

        m.mean_error     = sum_err / n;
        m.peak_error     = peak_err;
        m.mean_effort    = sum_eff / n;
        m.peak_effort    = peak_eff;
        m.saturation_frac = static_cast<double>(sat_count) / n;
        m.mean_aniso     = sum_aniso / n;
        m.mean_trG       = sum_trace / n;
        m.efficiency     = m.mean_error * m.mean_effort;

        // Final anisotropy: average of last 5% of data
        int tail = std::max(1, static_cast<int>(data_.size() * 0.05));
        double fa = 0;
        for (size_t i = data_.size() - tail; i < data_.size(); ++i) {
            double lmax = data_[i].G_eigvals.maxCoeff();
            double lmin = std::max(data_[i].G_eigvals.minCoeff(), 1e-6);
            fa += lmax / lmin;
        }
        m.final_aniso = fa / tail;

        // Breakdown detection: compare tail error vs early error
        int tail20 = std::max(1, static_cast<int>(data_.size() / 5));
        double tail_sum = 0;
        for (size_t i = data_.size() - tail20; i < data_.size(); ++i)
            tail_sum += data_[i].x.norm();
        m.tail_error = tail_sum / tail20;
        m.breakdown = (m.tail_error > m.mean_error * 2.0 && m.tail_error > 0.3);

        return m;
    }

    bool dump_csv(const std::string& path) const {
        std::ofstream f(path);
        if (!f) return false;
        f << std::setprecision(8);
        f << "t";
        for (int i = 0; i < Dim; ++i) f << ",x" << i;
        for (int i = 0; i < Dim; ++i) f << ",u" << i;
        for (int i = 0; i < Dim; ++i) f << ",G_eig" << i;
        f << ",trG,Q_norm\n";
        for (auto& s : data_) {
            f << s.t;
            for (int i = 0; i < Dim; ++i) f << ',' << s.x(i);
            for (int i = 0; i < Dim; ++i) f << ',' << s.u(i);
            for (int i = 0; i < Dim; ++i) f << ',' << s.G_eigvals(i);
            f << ',' << s.trG << ',' << s.Q_norm << '\n';
        }
        return true;
    }
};

} // namespace aniso
