#pragma once

#include "engine.hpp"
#include "config.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <chrono>

namespace aniso {

template<int Dim>
struct BenchResult {
    std::string name;
    Metrics<Dim> metrics;
    double wall_ms;
};

// Run one controller on a given environment config, return metrics
template<int Dim>
BenchResult<Dim> run_one(const YAML::Node& env_cfg,
                         const YAML::Node& ctrl_node,
                         const std::string& name) {
    // Build engine with this specific controller
    EngineParams<Dim> p;

    auto time_node = env_cfg["time"];
    p.dt    = time_node["dt"].as<double>(0.01);
    p.t_end = time_node["t_end"].as<double>(60.0);

    auto plant = env_cfg["plant"];
    p.A  = detail::parse_matrix<Dim>(plant["A"]);
    p.B  = detail::parse_matrix<Dim>(plant["B"]);
    p.w  = detail::parse_vec<Dim>(plant["w"]);
    p.x0 = detail::parse_vec<Dim>(plant["x0"]);

    if (env_cfg["initial_G"] && env_cfg["initial_G"].as<std::string>("identity") != "identity")
        p.G0.G = detail::parse_matrix<Dim>(env_cfg["initial_G"]);

    auto relax = env_cfg["relaxation"];
    p.tau = relax ? relax["tau"].as<double>(5.0) : 5.0;

    if (env_cfg["eigenvalue_clamp"]) {
        p.eig_lo = env_cfg["eigenvalue_clamp"]["lo"].as<double>(0.3);
        p.eig_hi = env_cfg["eigenvalue_clamp"]["hi"].as<double>(5.0);
    }
    p.seed = env_cfg["seed"].as<uint64_t>(42);

    double u_max = ctrl_node["u_max"].as<double>(3.0);

    Engine<Dim> engine(
        std::move(p),
        detail::make_coupling<Dim>(env_cfg["coupling"]),
        detail::make_interaction<Dim>(env_cfg["interaction"]),
        detail::make_feedback<Dim>(env_cfg["feedback"]),
        detail::make_observer<Dim>(env_cfg["observation"], env_cfg["resolution"]),
        detail::make_controller<Dim>(ctrl_node)
    );

    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = engine.run();
    auto t1 = std::chrono::high_resolution_clock::now();
    double wall = std::chrono::duration<double, std::milli>(t1 - t0).count();

    auto metrics = result.recorder.compute_metrics(5.0, u_max);
    return {name, metrics, wall};
}

// Run all controllers listed in cfg["controllers"] against the same environment
template<int Dim>
std::vector<BenchResult<Dim>> run_benchmark(const YAML::Node& cfg) {
    std::vector<BenchResult<Dim>> results;
    auto controllers = cfg["controllers"];
    if (!controllers || !controllers.IsSequence()) return results;

    for (size_t i = 0; i < controllers.size(); ++i) {
        YAML::Node c = controllers[i];
        std::string name = c["name"].as<std::string>("ctrl_" + std::to_string(i));
        results.push_back(run_one<Dim>(cfg, c, name));
    }
    return results;
}

// Auto-grid Pareto: generate many controller configs, run each
template<int Dim>
std::vector<BenchResult<Dim>> run_pareto_grid(const YAML::Node& cfg) {
    std::vector<BenchResult<Dim>> results;

    const double gains[] = {0.3, 0.6, 1.0, 1.5, 2.0, 2.8, 3.5, 5.0, 7.0, 10.0};
    const double umaxs[] = {1.0, 2.0, 3.0, 5.0, 8.0};

    for (double g : gains) {
        for (double um : umaxs) {
            // Proportional
            {
                YAML::Node cn;
                cn["type"] = "proportional";
                cn["gain"] = g;
                cn["u_max"] = um;
                char name[64];
                std::snprintf(name, 64, "P(%.1f,%.0f)", g, um);
                results.push_back(run_one<Dim>(cfg, cn, name));
            }
            // Aniso-aware
            {
                YAML::Node cn;
                cn["type"] = "aniso_aware";
                cn["gain"] = g;
                cn["u_max"] = um;
                char name[64];
                std::snprintf(name, 64, "A(%.1f,%.0f)", g, um);
                results.push_back(run_one<Dim>(cfg, cn, name));
            }
        }
    }

    const double periods[] = {1.0, 3.0, 6.0, 10.0};
    const double duties[] = {0.3, 0.5, 0.7};
    for (double g : {1.5, 3.0, 5.0}) {
        for (double um : {2.0, 3.0, 5.0}) {
            for (double per : periods) {
                for (double dut : duties) {
                    YAML::Node cn;
                    cn["type"] = "pulsed";
                    cn["gain"] = g;
                    cn["u_max"] = um;
                    cn["period"] = per;
                    cn["duty"] = dut;
                    char name[96];
                    std::snprintf(name, 96, "Pul(%.1f,%.0f,%.0f,%.0f%%)",
                        g, um, per, dut*100);
                    results.push_back(run_one<Dim>(cfg, cn, name));
                }
            }
        }
    }

    // Event-triggered
    const double triggers[] = {0.2, 0.5, 1.0};
    const double anticips[] = {2.0, 5.0, 10.0};
    for (double g : {1.5, 3.0, 5.0, 10.0}) {
        for (double um : {3.0, 5.0, 8.0}) {
            for (double trig : triggers) {
                for (double ant : anticips) {
                    YAML::Node cn;
                    cn["type"] = "event_triggered";
                    cn["gain"] = g;
                    cn["u_max"] = um;
                    cn["trigger"] = trig;
                    cn["hysteresis"] = 0.6;
                    cn["anticipation"] = ant;
                    char name[96];
                    std::snprintf(name, 96, "Evt(%.0f,%.0f,%.1f,%.0f)",
                        g, um, trig, ant);
                    results.push_back(run_one<Dim>(cfg, cn, name));
                }
            }
        }
    }

    return results;
}

// Sweep: vary one numeric parameter, run all controllers at each value
template<int Dim>
struct SweepRow {
    double param_value;
    std::vector<BenchResult<Dim>> results;
};

template<int Dim>
std::vector<SweepRow<Dim>> run_sweep(YAML::Node cfg,
                                      const std::string& param_path,
                                      double lo, double hi, double step) {
    std::vector<SweepRow<Dim>> rows;

    // Parse dotted param path (e.g. "relaxation.tau")
    auto set_param = [](YAML::Node root, const std::string& path, double val) {
        size_t dot = path.find('.');
        if (dot == std::string::npos) {
            root[path] = val;
        } else {
            std::string key = path.substr(0, dot);
            std::string rest = path.substr(dot + 1);
            // Support two levels
            size_t dot2 = rest.find('.');
            if (dot2 == std::string::npos) {
                root[key][rest] = val;
            } else {
                root[key][rest.substr(0, dot2)][rest.substr(dot2 + 1)] = val;
            }
        }
    };

    for (double v = lo; v <= hi + step * 0.01; v += step) {
        set_param(cfg, param_path, v);
        auto results = run_benchmark<Dim>(cfg);
        rows.push_back({v, std::move(results)});
    }
    return rows;
}

// Print comparison table to stdout
template<int Dim>
void print_table(const std::vector<BenchResult<Dim>>& results) {
    std::cout << std::left << std::setw(20) << "Controller"
              << std::right
              << std::setw(10) << "Error"
              << std::setw(10) << "PeakErr"
              << std::setw(10) << "Effort"
              << std::setw(10) << "PeakU"
              << std::setw(8)  << "Sat%"
              << std::setw(10) << "Aniso"
              << std::setw(10) << "FinalA"
              << std::setw(10) << "tr(G)"
              << std::setw(10) << "Effic."
              << std::setw(6)  << "OK?"
              << std::setw(8)  << "ms"
              << '\n';
    std::cout << std::string(122, '-') << '\n';

    for (auto& r : results) {
        auto& m = r.metrics;
        std::cout << std::left  << std::setw(20) << r.name
                  << std::right << std::fixed << std::setprecision(4)
                  << std::setw(10) << m.mean_error
                  << std::setw(10) << m.peak_error
                  << std::setw(10) << m.mean_effort
                  << std::setw(10) << m.peak_effort
                  << std::setw(7)  << std::setprecision(1) << m.saturation_frac * 100 << '%'
                  << std::setw(10) << std::setprecision(2) << m.mean_aniso
                  << std::setw(10) << m.final_aniso
                  << std::setw(10) << m.mean_trG
                  << std::setw(10) << std::setprecision(4) << m.efficiency
                  << std::setw(6)  << (m.breakdown ? "FAIL" : "OK")
                  << std::setw(7)  << std::setprecision(1) << r.wall_ms << '\n';
    }
}

// Dump benchmark results to CSV
template<int Dim>
bool dump_bench_csv(const std::vector<BenchResult<Dim>>& results,
                    const std::string& path) {
    std::ofstream f(path);
    if (!f) return false;
    f << std::setprecision(8);
    f << "controller,mean_error,peak_error,settling_time,"
         "mean_effort,peak_effort,saturation_frac,"
         "mean_aniso,final_aniso,mean_trG,phase_transition_time,"
         "efficiency,wall_ms\n";
    for (auto& r : results) {
        auto& m = r.metrics;
        f << r.name
          << ',' << m.mean_error << ',' << m.peak_error << ',' << m.settling_time
          << ',' << m.mean_effort << ',' << m.peak_effort << ',' << m.saturation_frac
          << ',' << m.mean_aniso << ',' << m.final_aniso << ',' << m.mean_trG
          << ',' << m.phase_transition_time
          << ',' << m.efficiency << ',' << r.wall_ms << '\n';
    }
    return true;
}

// Dump sweep results to CSV
template<int Dim>
bool dump_sweep_csv(const std::vector<SweepRow<Dim>>& rows,
                    const std::string& param_name,
                    const std::string& path) {
    std::ofstream f(path);
    if (!f) return false;
    f << std::setprecision(8);
    f << param_name << ",controller,mean_error,mean_effort,mean_aniso,final_aniso,"
         "efficiency,phase_transition_time\n";
    for (auto& row : rows) {
        for (auto& r : row.results) {
            auto& m = r.metrics;
            f << row.param_value << ',' << r.name
              << ',' << m.mean_error << ',' << m.mean_effort
              << ',' << m.mean_aniso << ',' << m.final_aniso
              << ',' << m.efficiency << ',' << m.phase_transition_time << '\n';
        }
    }
    return true;
}

} // namespace aniso
