#include <aniso/benchmark.hpp>
#include <aniso/grid_benchmark.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include <cstring>

// ---- Single simulation mode (original) ----
template<int Dim>
static int run_single(const YAML::Node& cfg) {
    auto engine = aniso::build_engine<Dim>(cfg);
    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = engine.run();
    auto t1 = std::chrono::high_resolution_clock::now();
    double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    double u_max = cfg["controller"]["u_max"].as<double>(3.0);
    auto m = result.recorder.compute_metrics(5.0, u_max);

    std::cout << "Simulation complete (" << Dim << "-D, "
              << result.recorder.size() << " steps, "
              << wall_ms << " ms)\n";
    std::cout << "  mean |x|   = " << m.mean_error
              << "   peak |x|  = " << m.peak_error << '\n';
    std::cout << "  mean |u|^2 = " << m.mean_effort
              << "   peak |u|  = " << m.peak_effort
              << "   sat = " << m.saturation_frac * 100 << "%\n";
    std::cout << "  mean aniso = " << m.mean_aniso
              << "   final     = " << m.final_aniso
              << "   tr(G)     = " << m.mean_trG << '\n';
    std::cout << "  efficiency = " << m.efficiency << '\n';
    if (m.phase_transition_time >= 0)
        std::cout << "  phase transition at t = " << m.phase_transition_time << '\n';

    std::string csv_path = "results.csv";
    if (cfg["output"] && cfg["output"]["csv"])
        csv_path = cfg["output"]["csv"].as<std::string>();
    if (result.recorder.dump_csv(csv_path))
        std::cout << "Output: " << csv_path << '\n';
    return 0;
}

// ---- Benchmark mode: compare controllers ----
template<int Dim>
static int run_bench(const YAML::Node& cfg) {
    std::cout << "=== Controller Benchmark (" << Dim << "-D) ===\n\n";

    auto results = aniso::run_benchmark<Dim>(cfg);
    aniso::print_table(results);

    std::string csv_path = "bench_results.csv";
    if (cfg["output"] && cfg["output"]["csv"])
        csv_path = cfg["output"]["csv"].as<std::string>();
    if (aniso::dump_bench_csv(results, csv_path))
        std::cout << "\nResults saved to " << csv_path << '\n';

    // Determine winner by efficiency
    if (!results.empty()) {
        auto best = std::min_element(results.begin(), results.end(),
            [](auto& a, auto& b) { return a.metrics.efficiency < b.metrics.efficiency; });
        std::cout << "\nBest efficiency: " << best->name
                  << " (" << best->metrics.efficiency << ")\n";
    }
    return 0;
}

// ---- Sweep mode: vary parameter, compare controllers at each value ----
template<int Dim>
static int run_sweep(YAML::Node cfg) {
    auto sweep_node = cfg["sweep"];
    if (!sweep_node) {
        std::cerr << "No 'sweep' section in config\n";
        return 1;
    }
    std::string param = sweep_node["param"].as<std::string>();
    auto range = sweep_node["range"];
    double lo   = range[0].as<double>();
    double hi   = range[1].as<double>();
    double step = range[2].as<double>();

    int n_points = static_cast<int>((hi - lo) / step + 1.5);
    int n_ctrl = static_cast<int>(cfg["controllers"].size());
    std::cout << "=== Sweep: " << param << " [" << lo << " .. " << hi
              << "] step=" << step << " (" << n_points << " points x "
              << n_ctrl << " controllers) ===\n\n";

    auto rows = aniso::run_sweep<Dim>(cfg, param, lo, hi, step);

    // Print summary
    std::cout << std::left << std::setw(10) << param;
    for (auto& r : rows[0].results)
        std::cout << std::setw(22) << r.name;
    std::cout << '\n';

    std::cout << std::setw(10) << "";
    for (size_t i = 0; i < rows[0].results.size(); ++i)
        std::cout << std::setw(11) << "error" << std::setw(11) << "effic.";
    std::cout << '\n';
    std::cout << std::string(10 + 22 * rows[0].results.size(), '-') << '\n';

    for (auto& row : rows) {
        std::cout << std::fixed << std::setprecision(2)
                  << std::setw(10) << row.param_value;
        for (auto& r : row.results) {
            std::cout << std::setprecision(4)
                      << std::setw(11) << r.metrics.mean_error
                      << std::setw(11) << r.metrics.efficiency;
        }
        std::cout << '\n';
    }

    std::string csv_path = "sweep_results.csv";
    if (cfg["output"] && cfg["output"]["csv"])
        csv_path = cfg["output"]["csv"].as<std::string>();
    if (aniso::dump_sweep_csv(rows, param, csv_path))
        std::cout << "\nSweep data saved to " << csv_path << '\n';

    return 0;
}

// ---- Grid sweep mode: vary parameter, compare controllers on GridEngine ----
template<int Dim>
static int run_grid_sweep(YAML::Node cfg) {
    auto sweep_node = cfg["sweep"];
    if (!sweep_node) {
        std::cerr << "No 'sweep' section in config\n";
        return 1;
    }
    std::string param = sweep_node["param"].as<std::string>();
    auto range = sweep_node["range"];
    double lo   = range[0].as<double>();
    double hi   = range[1].as<double>();
    double step = range[2].as<double>();
    int n_steps = sweep_node["steps"].as<int>(5000);
    int warmup  = sweep_node["warmup"].as<int>(500);

    int n_points = static_cast<int>((hi - lo) / step + 1.5);
    int n_ctrl = cfg["controllers"] ? static_cast<int>(cfg["controllers"].size()) : 0;

    std::cout << "=== Grid Sweep: " << param << " [" << lo << " .. " << hi
              << "] step=" << step << " (" << n_points << " pts x "
              << n_ctrl << " ctrl, " << n_steps << " steps each) ===\n\n";

    auto rows = aniso::run_grid_sweep<Dim>(cfg, param, lo, hi, step,
                                            n_steps, warmup);

    // Print summary
    if (!rows.empty() && !rows[0].results.empty()) {
        std::cout << '\n' << std::left << std::setw(10) << param;
        for (auto& r : rows[0].results)
            std::cout << std::setw(28) << r.ctrl_name;
        std::cout << '\n';

        std::cout << std::setw(10) << "";
        for (size_t i = 0; i < rows[0].results.size(); ++i)
            std::cout << std::setw(9) << "avg|x|"
                      << std::setw(9) << "avgE"
                      << std::setw(7) << "OK?"
                      << "   ";
        std::cout << '\n';
        std::cout << std::string(10 + 28 * rows[0].results.size(), '-') << '\n';

        for (auto& row : rows) {
            std::cout << std::fixed << std::setprecision(2)
                      << std::setw(10) << row.param_value;
            for (auto& m : row.results) {
                std::cout << std::setprecision(4)
                          << std::setw(9) << m.avg_x
                          << std::setw(9) << m.avg_E
                          << std::setw(7) << (m.breakdown ? "FAIL" : "OK")
                          << "   ";
            }
            std::cout << '\n';
        }
    }

    std::string csv_path = "grid_sweep.csv";
    if (cfg["output"] && cfg["output"]["csv"])
        csv_path = cfg["output"]["csv"].as<std::string>();
    if (aniso::dump_grid_sweep_csv(rows, param, csv_path))
        std::cout << "\nGrid sweep saved to " << csv_path << '\n';

    return 0;
}

// ---- 2D Grid sweep: phase diagram ----
template<int Dim>
static int run_grid_sweep2d(YAML::Node cfg) {
    auto sw = cfg["sweep2d"];
    if (!sw) { std::cerr << "No 'sweep2d' section in config\n"; return 1; }

    std::string p1 = sw["param1"]["name"].as<std::string>();
    double lo1 = sw["param1"]["range"][0].as<double>();
    double hi1 = sw["param1"]["range"][1].as<double>();
    double st1 = sw["param1"]["range"][2].as<double>();

    std::string p2 = sw["param2"]["name"].as<std::string>();
    double lo2 = sw["param2"]["range"][0].as<double>();
    double hi2 = sw["param2"]["range"][1].as<double>();
    double st2 = sw["param2"]["range"][2].as<double>();

    int n_steps = sw["steps"].as<int>(3000);
    int warmup  = sw["warmup"].as<int>(300);

    int n1 = static_cast<int>((hi1 - lo1) / st1 + 1.5);
    int n2 = static_cast<int>((hi2 - lo2) / st2 + 1.5);
    int n_ctrl = cfg["controllers"] ? static_cast<int>(cfg["controllers"].size()) : 0;

    std::cout << "=== 2D Grid Sweep ===\n"
              << "  " << p1 << ": [" << lo1 << " .. " << hi1 << "] step " << st1
              << " (" << n1 << " pts)\n"
              << "  " << p2 << ": [" << lo2 << " .. " << hi2 << "] step " << st2
              << " (" << n2 << " pts)\n"
              << "  " << n1 * n2 * n_ctrl << " total simulations, "
              << n_steps << " steps each\n"
              << "  Threads: " << std::thread::hardware_concurrency() << "\n\n";

    auto results = aniso::run_grid_sweep2d<Dim>(
        cfg, p1, lo1, hi1, st1, p2, lo2, hi2, st2, n_steps, warmup);

    std::string csv_path = "grid_sweep2d.csv";
    if (cfg["output"] && cfg["output"]["csv"])
        csv_path = cfg["output"]["csv"].as<std::string>();
    if (aniso::dump_grid_sweep2d_csv(results, p1, p2, csv_path))
        std::cout << "\n2D sweep saved to " << csv_path << '\n';

    return 0;
}

// ---- Dispatch ----
template<int Dim>
static int dispatch(const YAML::Node& cfg, const std::string& mode) {
    if (mode == "run")          return run_single<Dim>(cfg);
    if (mode == "bench")        return run_bench<Dim>(cfg);
    if (mode == "sweep")        return run_sweep<Dim>(YAML::Clone(cfg));
    if (mode == "grid_sweep")   return run_grid_sweep<Dim>(YAML::Clone(cfg));
    if (mode == "grid_sweep2d") return run_grid_sweep2d<Dim>(YAML::Clone(cfg));
    std::cerr << "Unknown mode: " << mode << '\n';
    return 1;
}

static void print_usage() {
    std::cerr << "Usage:\n"
              << "  aniso run          <config.yaml>  Run single simulation\n"
              << "  aniso bench        <config.yaml>  Compare controllers\n"
              << "  aniso sweep        <config.yaml>  Parameter sweep (Engine)\n"
              << "  aniso grid_sweep   <config.yaml>  1D sweep (GridEngine)\n"
              << "  aniso grid_sweep2d <config.yaml>  2D phase diagram (GridEngine)\n"
              << "  aniso <config.yaml>               (same as 'run')\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) { print_usage(); return 1; }

    std::string mode = "run";
    std::string config_path;

    if (argc == 2) {
        config_path = argv[1];
    } else {
        mode = argv[1];
        config_path = argv[2];
    }

    try {
        YAML::Node cfg = YAML::LoadFile(config_path);
        int dim = cfg["dim"].as<int>();

        switch (dim) {
            case 1: return dispatch<1>(cfg, mode);
            case 2: return dispatch<2>(cfg, mode);
            case 3: return dispatch<3>(cfg, mode);
            default:
                std::cerr << "Unsupported dim=" << dim << "\n";
                return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
