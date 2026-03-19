#pragma once

#include "engine.hpp"
#include "chain.hpp"
#include "grid.hpp"
#include "g_response.hpp"
#include "heater.hpp"
#include "resolution.hpp"
#include <yaml-cpp/yaml.h>
#include <string>
#include <stdexcept>

namespace aniso {

namespace detail {

template<int Dim>
Mat<Dim> parse_matrix(const YAML::Node& node) {
    Mat<Dim> m;
    if (node.IsSequence() && node.size() == Dim) {
        for (int i = 0; i < Dim; ++i)
            for (int j = 0; j < Dim; ++j)
                m(i, j) = node[i][j].as<double>();
    } else {
        // flat array
        for (int i = 0; i < Dim; ++i)
            for (int j = 0; j < Dim; ++j)
                m(i, j) = node[i * Dim + j].as<double>();
    }
    return m;
}

template<int Dim>
Vec<Dim> parse_vec(const YAML::Node& node) {
    Vec<Dim> v;
    for (int i = 0; i < Dim; ++i)
        v(i) = node[i].as<double>();
    return v;
}

template<int Dim>
std::unique_ptr<ICoupling<Dim>> make_coupling(const YAML::Node& node) {
    auto type = node["type"].as<std::string>();
    double alpha = node["alpha"].as<double>(0.5);
    double gamma = node["gamma"].as<double>(1.0);
    if (type == "rank1")     return std::make_unique<Rank1Coupling<Dim>>(alpha, gamma);
    if (type == "isotropic") return std::make_unique<IsotropicCoupling<Dim>>(alpha, gamma);
    throw std::runtime_error("Unknown coupling type: " + type);
}

template<int Dim>
std::unique_ptr<IInteraction<Dim>> make_interaction(const YAML::Node& node) {
    if (!node || !node["type"]) return std::make_unique<NoInteraction<Dim>>();
    auto type = node["type"].as<std::string>();
    if (type == "none") return std::make_unique<NoInteraction<Dim>>();
    if (type == "landau") {
        double mu    = node["mu"].as<double>();
        double g_c   = node["g_c"].as<double>();
        double nu    = node["nu"].as<double>();
        double r_max = node["r_max"].as<double>(0.0);
        return std::make_unique<LandauInteraction<Dim>>(mu, g_c, nu, r_max);
    }
    throw std::runtime_error("Unknown interaction type: " + type);
}

template<int Dim>
std::unique_ptr<IFeedback<Dim>> make_feedback(const YAML::Node& node) {
    if (!node || !node["type"]) return std::make_unique<NoFeedback<Dim>>();
    auto type = node["type"].as<std::string>();
    if (type == "none")      return std::make_unique<NoFeedback<Dim>>();
    if (type == "traceless") return std::make_unique<TracelessFeedback<Dim>>(node["kappa"].as<double>());
    if (type == "full")      return std::make_unique<FullFeedback<Dim>>(node["kappa"].as<double>());
    throw std::runtime_error("Unknown feedback type: " + type);
}

template<int Dim>
std::shared_ptr<IResolution<Dim>> make_resolution(const YAML::Node& node) {
    std::string type = "identity";
    double l0 = 0.04;
    double alpha = 1.0;
    if (node) {
        type  = node["type"].as<std::string>("identity");
        l0    = node["l0"].as<double>(0.04);
        alpha = node["alpha"].as<double>(1.0);
    }
    if (type == "identity")
        return std::make_shared<IdentityResolution<Dim>>(l0);
    if (type == "metric")
        return std::make_shared<MetricResolution<Dim>>(l0, alpha);
    throw std::runtime_error("Unknown resolution type: " + type);
}

template<int Dim>
std::unique_ptr<IObserver<Dim>> make_observer(const YAML::Node& obs_node,
                                               const YAML::Node& res_node) {
    auto resolution = make_resolution<Dim>(res_node);
    double sigma_G = 0.3;
    if (obs_node) {
        sigma_G = obs_node["sigma_G"].as<double>(0.3);
    }
    return std::make_unique<ResolutionObserver<Dim>>(std::move(resolution), sigma_G);
}

// Legacy overload for backward compatibility
template<int Dim>
std::unique_ptr<IObserver<Dim>> make_observer(const YAML::Node& obs_node) {
    return make_observer<Dim>(obs_node, YAML::Node());
}

template<int Dim>
std::unique_ptr<IController<Dim>> make_controller(const YAML::Node& node) {
    auto type  = node["type"].as<std::string>();
    double gain  = node["gain"].as<double>(1.0);
    double u_max = node["u_max"].as<double>(3.0);
    if (type == "proportional") return std::make_unique<ProportionalController<Dim>>(gain, u_max);
    if (type == "aniso_aware")  return std::make_unique<AnisoAwareController<Dim>>(gain, u_max);
    if (type == "pulsed") {
        double period = node["period"].as<double>(5.0);
        double duty   = node["duty"].as<double>(0.5);
        return std::make_unique<PulsedController<Dim>>(gain, u_max, period, duty);
    }
    if (type == "event_triggered") {
        double trigger      = node["trigger"].as<double>(0.5);
        double hysteresis   = node["hysteresis"].as<double>(0.6);
        double anticipation = node["anticipation"].as<double>(5.0);
        return std::make_unique<EventTriggeredController<Dim>>(
            gain, u_max, trigger, hysteresis, anticipation);
    }
    if (type == "pid") {
        double kp = node["kp"].as<double>(gain);
        double ki = node["ki"].as<double>(0.5);
        double kd = node["kd"].as<double>(0.1);
        double dt = node["dt"].as<double>(0.01);
        return std::make_unique<PIDController<Dim>>(kp, ki, kd, u_max, dt);
    }
    throw std::runtime_error("Unknown controller type: " + type);
}

template<int Dim>
std::unique_ptr<IGResponse<Dim>> make_g_response(const YAML::Node& node,
                                                  const GResponseParams& defaults)
{
    GResponseParams rp = defaults;
    std::string type = "relax_aniso";
    if (node && node["type"]) type = node["type"].as<std::string>();
    if (node) {
        if (node["tau"])   rp.tau_0     = node["tau"].as<double>(rp.tau_0);
        if (node["kappa"]) rp.kappa     = node["kappa"].as<double>(rp.kappa);
        if (node["noise"]) rp.noise_amp = node["noise"].as<double>(rp.noise_amp);
    }
    if (type == "relax_aniso")
        return std::make_unique<RelaxAnisoResponse<Dim>>(rp);
    if (type == "relax_energy")
        return std::make_unique<RelaxEnergyResponse<Dim>>(rp);
    if (type == "melt")
        return std::make_unique<MeltResponse<Dim>>(rp);
    if (type == "landau_energy") {
        double nu = node ? node["nu"].as<double>(0.5) : 0.5;
        return std::make_unique<LandauEnergyResponse<Dim>>(rp, nu);
    }
    throw std::runtime_error("Unknown g_response type: " + type);
}

template<int Dim>
std::unique_ptr<IHeater<Dim>> make_heater(const YAML::Node& node) {
    std::string type = "constant";
    double power = 1.0;
    if (node) {
        type  = node["type"].as<std::string>("constant");
        power = node["power"].as<double>(1.0);
    }
    if (type == "constant")
        return std::make_unique<ConstantHeater<Dim>>(power);
    if (type == "pulsed") {
        double period = node["period"].as<double>(5.0);
        double duty   = node["duty"].as<double>(0.5);
        return std::make_unique<PulsedHeater<Dim>>(power, period, duty);
    }
    if (type == "event_driven") {
        double trigger    = node["trigger"].as<double>(0.5);
        double hysteresis = node["hysteresis"].as<double>(1.5);
        return std::make_unique<EventDrivenHeater<Dim>>(power, trigger, hysteresis);
    }
    if (type == "aniso_aware")
        return std::make_unique<AnisoAwareHeater<Dim>>(power);
    if (type == "global_event") {
        double trigger    = node["trigger"].as<double>(0.5);
        double hysteresis = node["hysteresis"].as<double>(1.5);
        return std::make_unique<GlobalEventHeater<Dim>>(power, trigger, hysteresis);
    }
    if (type == "adaptive_pulsed") {
        double period         = node["period"].as<double>(1.5);
        double duty_min       = node["duty_min"].as<double>(0.15);
        double barrier_target = node["barrier_target"].as<double>(3.0);
        return std::make_unique<AdaptivePulsedHeater<Dim>>(power, period, duty_min, barrier_target);
    }
    throw std::runtime_error("Unknown heater type: " + type);
}

} // namespace detail

template<int Dim>
Engine<Dim> build_engine(const YAML::Node& cfg) {
    EngineParams<Dim> p;

    auto time_node = cfg["time"];
    p.dt    = time_node["dt"].as<double>(0.01);
    p.t_end = time_node["t_end"].as<double>(60.0);

    auto plant = cfg["plant"];
    p.A  = detail::parse_matrix<Dim>(plant["A"]);
    p.B  = detail::parse_matrix<Dim>(plant["B"]);
    p.w  = detail::parse_vec<Dim>(plant["w"]);
    p.x0 = detail::parse_vec<Dim>(plant["x0"]);

    if (cfg["initial_G"] && cfg["initial_G"].as<std::string>("identity") != "identity") {
        p.G0.G = detail::parse_matrix<Dim>(cfg["initial_G"]);
    }

    auto relax = cfg["relaxation"];
    p.tau = relax ? relax["tau"].as<double>(5.0) : 5.0;

    if (cfg["eigenvalue_clamp"]) {
        p.eig_lo = cfg["eigenvalue_clamp"]["lo"].as<double>(0.3);
        p.eig_hi = cfg["eigenvalue_clamp"]["hi"].as<double>(5.0);
    }

    p.seed = cfg["seed"].as<uint64_t>(42);

    return Engine<Dim>(
        std::move(p),
        detail::make_coupling<Dim>(cfg["coupling"]),
        detail::make_interaction<Dim>(cfg["interaction"]),
        detail::make_feedback<Dim>(cfg["feedback"]),
        detail::make_observer<Dim>(cfg["observation"], cfg["resolution"]),
        detail::make_controller<Dim>(cfg["controller"])
    );
}

template<int Dim>
ChainEngine<Dim> build_chain(const YAML::Node& cfg) {
    ChainParams<Dim> cp;

    auto ch = cfg["chain"];
    if (ch) {
        cp.N            = ch["N"].as<int>(64);
        cp.D_G          = ch["D_G"].as<double>(0.08);
        cp.D_x          = ch["D_x"].as<double>(0.03);
        if (ch["drive"]) {
            cp.drive_center = ch["drive"]["center"].as<double>(0.5);
            cp.drive_width  = ch["drive"]["width"].as<double>(0.2);
            cp.drive_peak   = ch["drive"]["peak"].as<double>(2.5);
        }
        cp.trap   = ch["trap"].as<double>(1.0);
        cp.s_crit = ch["s_crit"].as<double>(1.5);
    }

    auto time_node = cfg["time"];
    cp.dt    = time_node["dt"].as<double>(0.005);
    cp.t_end = time_node["t_end"].as<double>(300.0);

    auto plant = cfg["plant"];
    cp.A  = detail::parse_matrix<Dim>(plant["A"]);
    cp.B  = detail::parse_matrix<Dim>(plant["B"]);
    cp.w  = detail::parse_vec<Dim>(plant["w"]);
    cp.x0 = detail::parse_vec<Dim>(plant["x0"]);

    auto relax = cfg["relaxation"];
    cp.tau = relax ? relax["tau"].as<double>(8.0) : 8.0;

    if (cfg["eigenvalue_clamp"]) {
        cp.eig_lo = cfg["eigenvalue_clamp"]["lo"].as<double>(0.3);
        cp.eig_hi = cfg["eigenvalue_clamp"]["hi"].as<double>(5.0);
    }
    cp.seed = cfg["seed"].as<uint64_t>(42);

    return ChainEngine<Dim>(
        std::move(cp),
        detail::make_coupling<Dim>(cfg["coupling"]),
        detail::make_interaction<Dim>(cfg["interaction"]),
        detail::make_feedback<Dim>(cfg["feedback"]),
        detail::make_observer<Dim>(cfg["observation"], cfg["resolution"]),
        detail::make_controller<Dim>(cfg["controller"])
    );
}

template<int Dim>
GridEngine<Dim> build_grid(const YAML::Node& cfg) {
    GridParams<Dim> gp;

    auto gr = cfg["grid"];
    if (gr) {
        gp.Nx = gr["Nx"].as<int>(48);
        gp.Ny = gr["Ny"].as<int>(48);
        gp.D_E = gr["D_E"].as<double>(0.5);
        gp.D_x = gr["D_x"].as<double>(0.1);
        gp.gamma_diss   = gr["gamma_diss"].as<double>(1.0);
        gp.kappa_tau    = gr["kappa_tau"].as<double>(20.0);
        gp.noise_amp    = gr["noise_amp"].as<double>(0.5);
        gp.D_G          = gr["D_G"].as<double>(0.0);
        gp.g_noise_init = gr["g_noise_init"].as<double>(0.05);
        gp.eta_ctrl     = gr["eta_ctrl"].as<double>(0.3);
        gp.wall_absorb  = gr["wall_absorb"].as<bool>(true);
        gp.wall_radius  = gr["wall_radius"].as<double>(0.45);
        if (gr["heat_profile"]) {
            gp.heat_cx   = gr["heat_profile"]["cx"].as<double>(0.5);
            gp.heat_cy   = gr["heat_profile"]["cy"].as<double>(0.5);
            gp.heat_rx   = gr["heat_profile"]["rx"].as<double>(0.25);
            gp.heat_ry   = gr["heat_profile"]["ry"].as<double>(0.25);
            gp.heat_peak = gr["heat_profile"]["peak"].as<double>(1.0);
        }
    }

    auto time_node = cfg["time"];
    gp.dt = time_node["dt"].as<double>(0.01);

    auto plant = cfg["plant"];
    gp.A  = detail::parse_matrix<Dim>(plant["A"]);
    gp.B  = detail::parse_matrix<Dim>(plant["B"]);
    gp.w  = detail::parse_vec<Dim>(plant["w"]);
    gp.x0 = detail::parse_vec<Dim>(plant["x0"]);

    auto relax = cfg["relaxation"];
    gp.tau_0 = relax ? relax["tau"].as<double>(1.0) : 1.0;

    if (cfg["eigenvalue_clamp"]) {
        gp.eig_lo = cfg["eigenvalue_clamp"]["lo"].as<double>(0.3);
        gp.eig_hi = cfg["eigenvalue_clamp"]["hi"].as<double>(5.0);
    }
    gp.seed = cfg["seed"].as<uint64_t>(42);

    GResponseParams rp{gp.tau_0, gp.kappa_tau, gp.noise_amp, gp.eig_lo, gp.eig_hi};
    auto g_resp = detail::make_g_response<Dim>(cfg["g_response"], rp);
    auto heater = detail::make_heater<Dim>(cfg["heater"]);

    return GridEngine<Dim>(
        std::move(gp),
        detail::make_coupling<Dim>(cfg["coupling"]),
        detail::make_interaction<Dim>(cfg["interaction"]),
        detail::make_feedback<Dim>(cfg["feedback"]),
        detail::make_observer<Dim>(cfg["observation"], cfg["resolution"]),
        detail::make_controller<Dim>(cfg["controller"]),
        std::move(g_resp),
        std::move(heater)
    );
}

} // namespace aniso
