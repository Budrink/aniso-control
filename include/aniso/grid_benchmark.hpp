#pragma once

#include "grid.hpp"
#include "config.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <future>
#include <mutex>
#include <atomic>

namespace aniso {

struct GridMetrics {
    std::string ctrl_name;
    double param1 = 0, param2 = 0;
    double avg_x      = 0;
    double avg_E      = 0;
    double avg_trG    = 0;
    double max_aniso  = 0;
    double avg_aniso  = 0;    // mean anisotropy across grid (barrier strength)
    double avg_effort = 0;
    double E_gradient = 0;    // center-to-edge energy ratio (confinement quality)
    double tail_x     = 0;
    double tail_aniso = 0;    // mean anisotropy in tail period
    bool   breakdown  = false;
    double wall_ms    = 0;

    // Disruption observables
    double avg_wall_flux     = 0;  // mean energy flux to wall
    double avg_confinement   = 0;  // mean confinement ratio (center_E / edge_E)
    double avg_barrier_aniso = 0;  // mean anisotropy in barrier ring
    double tail_wall_flux    = 0;  // wall flux in tail period
    double tail_barrier      = 0;  // barrier anisotropy in tail period
    bool   disruption        = false; // true if barrier collapsed and energy reached wall
};

template<int Dim>
GridMetrics run_grid_one(const YAML::Node& cfg,
                         const YAML::Node& ctrl_node,
                         const std::string& name,
                         int n_steps = 5000,
                         int warmup_steps = 500) {
    auto grid = build_grid<Dim>(cfg);
    grid.swap_controller(detail::make_controller<Dim>(ctrl_node));
    grid.reset();

    int Nx = grid.Nx(), Ny = grid.Ny();
    int total = Nx * Ny;
    int tail_start = n_steps - n_steps / 5;

    double sum_x = 0, sum_E = 0, sum_tr = 0, sum_u = 0;
    double sum_aniso = 0;
    double sum_tail_x = 0, sum_tail_aniso = 0;
    double peak_aniso = 0;
    double sum_wall_flux = 0, sum_confinement = 0, sum_barrier = 0;
    double sum_tail_wf = 0, sum_tail_barrier = 0;
    int n_accum = 0, n_tail = 0;

    // Count interior (non-wall) cells for averaging
    int n_interior = 0;
    for (int i = 0; i < Nx; ++i)
        for (int j = 0; j < Ny; ++j)
            if (!grid.is_wall(i, j)) ++n_interior;
    if (n_interior == 0) n_interior = total;

    auto t0 = std::chrono::high_resolution_clock::now();

    constexpr double EARLY_STOP_THRESH = 2.0;
    constexpr int    CHECK_INTERVAL    = 50;
    bool early_stop = false;

    for (int step = 0; step < n_steps; ++step) {
        grid.step();
        if (step < warmup_steps) continue;

        bool in_tail = (step >= tail_start);
        if (!in_tail
            && (step - warmup_steps) % CHECK_INTERVAL != 0
            && step != n_steps - 1)
            continue;

        double frame_x = 0, frame_E = 0, frame_tr = 0, frame_u = 0;
        double frame_aniso = 0, frame_max_aniso = 0;
        double E_center = 0, E_edge = 0;
        int n_center = 0, n_edge = 0;

        for (int i = 0; i < Nx; ++i)
            for (int j = 0; j < Ny; ++j) {
                if (grid.is_wall(i, j)) continue;
                double xn = grid.x(i, j).norm();
                double eij = grid.E(i, j);
                double aniso = grid.anisotropy(i, j);
                frame_x  += xn;
                frame_E  += eij;
                frame_tr += grid.G(i, j).trace();
                frame_u  += grid.last_u(i, j).norm();
                frame_aniso += aniso;
                frame_max_aniso = std::max(frame_max_aniso, aniso);

                // Radial bin: center vs edge for energy gradient
                double rx = (Nx > 1) ? (double)i / (Nx - 1) - 0.5 : 0.0;
                double ry = (Ny > 1) ? (double)j / (Ny - 1) - 0.5 : 0.0;
                double r = std::sqrt(rx*rx + ry*ry);
                if (r < 0.15) { E_center += eij; ++n_center; }
                else if (r > 0.3) { E_edge += eij; ++n_edge; }
            }
        frame_x     /= n_interior;
        frame_E     /= n_interior;
        frame_tr    /= n_interior;
        frame_u     /= n_interior;
        frame_aniso /= n_interior;

        double frame_wf = grid.last_wall_flux();
        double frame_conf = grid.confinement_ratio();
        double frame_barrier = grid.barrier_anisotropy();

        sum_x += frame_x; sum_E += frame_E;
        sum_tr += frame_tr; sum_u += frame_u;
        sum_aniso += frame_aniso;
        sum_wall_flux += frame_wf;
        sum_confinement += frame_conf;
        sum_barrier += frame_barrier;
        peak_aniso = std::max(peak_aniso, frame_max_aniso);
        ++n_accum;

        if (in_tail) {
            sum_tail_x     += frame_x;
            sum_tail_aniso += frame_aniso;
            sum_tail_wf    += frame_wf;
            sum_tail_barrier += frame_barrier;
            ++n_tail;
        }

        if (frame_x > EARLY_STOP_THRESH) { early_stop = true; break; }
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    GridMetrics m;
    m.ctrl_name = name;
    m.wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (n_accum > 0) {
        m.avg_x      = sum_x    / n_accum;
        m.avg_E      = sum_E    / n_accum;
        m.avg_trG    = sum_tr   / n_accum;
        m.avg_effort = sum_u    / n_accum;
        m.avg_aniso  = sum_aniso / n_accum;
        m.max_aniso  = peak_aniso;
    }
    if (n_accum > 0) {
        m.avg_wall_flux     = sum_wall_flux   / n_accum;
        m.avg_confinement   = sum_confinement / n_accum;
        m.avg_barrier_aniso = sum_barrier     / n_accum;
    }
    if (n_tail > 0) {
        m.tail_x     = sum_tail_x     / n_tail;
        m.tail_aniso = sum_tail_aniso  / n_tail;
        m.tail_wall_flux = sum_tail_wf    / n_tail;
        m.tail_barrier   = sum_tail_barrier / n_tail;
    }
    // Breakdown: |x| diverging OR barrier completely lost
    m.breakdown = early_stop
               || (m.tail_x > 0.5)
               || (n_tail > 0 && m.tail_aniso < 0.02 && m.tail_x > 0.2);
    // Disruption: barrier collapsed AND energy reaching wall
    m.disruption = m.breakdown
                || (n_tail > 0 && m.tail_barrier < 0.05 && m.tail_wall_flux > 0.01)
                || (n_tail > 0 && m.avg_confinement < 1.5);
    return m;
}

// ── Helpers ──────────────────────────────────────────────────────────

inline void set_yaml_param(YAML::Node root,
                           const std::string& path, double val) {
    size_t dot = path.find('.');
    if (dot == std::string::npos) { root[path] = val; return; }
    std::string key = path.substr(0, dot);
    std::string rest = path.substr(dot + 1);
    size_t dot2 = rest.find('.');
    if (dot2 == std::string::npos)
        root[key][rest] = val;
    else
        root[key][rest.substr(0, dot2)][rest.substr(dot2 + 1)] = val;
}

inline bool is_controller_param(const std::string& path) {
    return path.rfind("controller.", 0) == 0;
}
inline std::string controller_subkey(const std::string& path) {
    return is_controller_param(path) ? path.substr(11) : "";
}

// ── 1D sweep (parallel) ─────────────────────────────────────────────

template<int Dim>
struct GridSweepRow {
    double param_value;
    std::vector<GridMetrics> results;
};

template<int Dim>
std::vector<GridSweepRow<Dim>> run_grid_sweep(
        YAML::Node cfg,
        const std::string& param_path,
        double lo, double hi, double step,
        int n_steps = 5000, int warmup = 500) {

    auto controllers = cfg["controllers"];
    if (!controllers || !controllers.IsSequence()) return {};
    int n_ctrl = static_cast<int>(controllers.size());

    bool is_cp = is_controller_param(param_path);
    std::string ck = controller_subkey(param_path);

    // Build task list: (param_value, ctrl_index)
    struct Task { double val; int ctrl; };
    std::vector<Task> tasks;
    std::vector<double> param_vals;
    for (double v = lo; v <= hi + step * 0.01; v += step) {
        param_vals.push_back(v);
        for (int c = 0; c < n_ctrl; ++c)
            tasks.push_back({v, c});
    }

    std::vector<GridMetrics> all_results(tasks.size());
    std::atomic<int> done_count{0};
    int total_tasks = static_cast<int>(tasks.size());

    auto worker = [&](int idx) {
        auto& t = tasks[idx];
        YAML::Node local_cfg = YAML::Clone(cfg);
        set_yaml_param(local_cfg, param_path, t.val);

        YAML::Node cn = YAML::Clone(controllers[t.ctrl]);
        if (is_cp && !ck.empty()) cn[ck] = t.val;

        std::string name = cn["name"].as<std::string>(
            "ctrl_" + std::to_string(t.ctrl));

        all_results[idx] = run_grid_one<Dim>(
            local_cfg, cn, name, n_steps, warmup);

        int d = ++done_count;
        if (d % n_ctrl == 0) {
            std::cout << "  " << param_path << " = " << t.val
                      << " done  [" << d << "/" << total_tasks << "]\n";
            std::cout.flush();
        }
    };

    // Launch parallel
    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::future<void>> futures;
    futures.reserve(tasks.size());

    for (int i = 0; i < total_tasks; ++i) {
        if (futures.size() >= hw) {
            futures.front().get();
            futures.erase(futures.begin());
        }
        futures.push_back(std::async(std::launch::async, worker, i));
    }
    for (auto& f : futures) f.get();

    // Reassemble into rows
    std::vector<GridSweepRow<Dim>> rows;
    int idx = 0;
    for (double v : param_vals) {
        GridSweepRow<Dim> row;
        row.param_value = v;
        for (int c = 0; c < n_ctrl; ++c)
            row.results.push_back(std::move(all_results[idx++]));
        rows.push_back(std::move(row));
    }
    return rows;
}

// ── 2D sweep (parallel) ─────────────────────────────────────────────

template<int Dim>
std::vector<GridMetrics> run_grid_sweep2d(
        YAML::Node cfg,
        const std::string& param1_path, double lo1, double hi1, double step1,
        const std::string& param2_path, double lo2, double hi2, double step2,
        int n_steps = 3000, int warmup = 300) {

    auto controllers = cfg["controllers"];
    if (!controllers || !controllers.IsSequence()) return {};
    int n_ctrl = static_cast<int>(controllers.size());

    bool is_cp1 = is_controller_param(param1_path);
    bool is_cp2 = is_controller_param(param2_path);
    std::string ck1 = controller_subkey(param1_path);
    std::string ck2 = controller_subkey(param2_path);

    struct Task { double v1, v2; int ctrl; };
    std::vector<Task> tasks;

    for (double v1 = lo1; v1 <= hi1 + step1 * 0.01; v1 += step1)
        for (double v2 = lo2; v2 <= hi2 + step2 * 0.01; v2 += step2)
            for (int c = 0; c < n_ctrl; ++c)
                tasks.push_back({v1, v2, c});

    int total = static_cast<int>(tasks.size());
    std::vector<GridMetrics> results(total);
    std::atomic<int> done_count{0};

    auto worker = [&](int idx) {
        auto& t = tasks[idx];
        YAML::Node local_cfg = YAML::Clone(cfg);
        set_yaml_param(local_cfg, param1_path, t.v1);
        set_yaml_param(local_cfg, param2_path, t.v2);

        YAML::Node cn = YAML::Clone(controllers[t.ctrl]);
        if (is_cp1 && !ck1.empty()) cn[ck1] = t.v1;
        if (is_cp2 && !ck2.empty()) cn[ck2] = t.v2;

        std::string name = cn["name"].as<std::string>(
            "ctrl_" + std::to_string(t.ctrl));

        auto m = run_grid_one<Dim>(local_cfg, cn, name, n_steps, warmup);
        m.param1 = t.v1;
        m.param2 = t.v2;
        results[idx] = std::move(m);

        int d = ++done_count;
        if (d % (n_ctrl * 5) == 0 || d == total) {
            std::cout << "  2D sweep: " << d << "/" << total << "\n";
            std::cout.flush();
        }
    };

    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::future<void>> futures;
    futures.reserve(std::min((unsigned)total, hw));

    for (int i = 0; i < total; ++i) {
        if (futures.size() >= hw) {
            futures.front().get();
            futures.erase(futures.begin());
        }
        futures.push_back(std::async(std::launch::async, worker, i));
    }
    for (auto& f : futures) f.get();

    return results;
}

// ── CSV export ──────────────────────────────────────────────────────

template<int Dim>
bool dump_grid_sweep_csv(const std::vector<GridSweepRow<Dim>>& rows,
                         const std::string& param_name,
                         const std::string& path) {
    std::ofstream f(path);
    if (!f) return false;
    f << std::setprecision(8);
    f << param_name
      << ",controller,avg_x,avg_E,avg_trG,max_aniso,avg_aniso,avg_effort,"
         "tail_x,tail_aniso,breakdown,"
         "avg_wall_flux,avg_confinement,avg_barrier_aniso,"
         "tail_wall_flux,tail_barrier,disruption,wall_ms\n";
    for (auto& row : rows)
        for (auto& m : row.results)
            f << row.param_value << ',' << m.ctrl_name << ','
              << m.avg_x << ',' << m.avg_E << ',' << m.avg_trG << ','
              << m.max_aniso << ',' << m.avg_aniso << ',' << m.avg_effort << ','
              << m.tail_x << ',' << m.tail_aniso << ','
              << (m.breakdown ? 1 : 0) << ','
              << m.avg_wall_flux << ',' << m.avg_confinement << ','
              << m.avg_barrier_aniso << ','
              << m.tail_wall_flux << ',' << m.tail_barrier << ','
              << (m.disruption ? 1 : 0) << ',' << m.wall_ms << '\n';
    return true;
}

inline bool dump_grid_sweep2d_csv(const std::vector<GridMetrics>& results,
                                  const std::string& p1_name,
                                  const std::string& p2_name,
                                  const std::string& path) {
    std::ofstream f(path);
    if (!f) return false;
    f << std::setprecision(8);
    f << p1_name << ',' << p2_name
      << ",controller,avg_x,avg_E,avg_trG,max_aniso,avg_aniso,avg_effort,"
         "tail_x,tail_aniso,breakdown,"
         "avg_wall_flux,avg_confinement,avg_barrier_aniso,"
         "tail_wall_flux,tail_barrier,disruption,wall_ms\n";
    for (auto& m : results)
        f << m.param1 << ',' << m.param2 << ',' << m.ctrl_name << ','
          << m.avg_x << ',' << m.avg_E << ',' << m.avg_trG << ','
          << m.max_aniso << ',' << m.avg_aniso << ',' << m.avg_effort << ','
          << m.tail_x << ',' << m.tail_aniso << ','
          << (m.breakdown ? 1 : 0) << ','
          << m.avg_wall_flux << ',' << m.avg_confinement << ','
          << m.avg_barrier_aniso << ','
          << m.tail_wall_flux << ',' << m.tail_barrier << ','
          << (m.disruption ? 1 : 0) << ',' << m.wall_ms << '\n';
    return true;
}

} // namespace aniso
