#define NOMINMAX
#define _USE_MATH_DEFINES

#include <aniso/benchmark.hpp>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"

#include <GLFW/glfw3.h>

#include <string>
#include <vector>
#include <cmath>
#include <cfloat>
#include <cstring>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// Ring buffer for scrolling time-series
// ---------------------------------------------------------------------------
struct ScrollBuf {
    std::vector<double> data;
    int offset = 0;
    int size   = 0;
    int cap    = 0;

    void init(int capacity) {
        cap = capacity;
        data.resize(capacity, 0.0);
        offset = 0; size = 0;
    }
    void push(double v) {
        data[offset] = v;
        offset = (offset + 1) % cap;
        if (size < cap) ++size;
    }
    void clear() { offset = 0; size = 0; }
    double operator[](int i) const { return data[(offset - size + i + cap) % cap]; }
    double back() const { return size > 0 ? (*this)[size - 1] : 0.0; }
};

// ---------------------------------------------------------------------------
// Kymograph: scrolling space-time diagram for 1D chain
// ---------------------------------------------------------------------------
struct Kymograph {
    std::vector<float> data;
    int N = 0, max_rows = 400;
    int write_pos = 0, count = 0;

    void init(int n, int rows = 400) {
        N = n; max_rows = rows;
        data.assign(n * rows, 1.0f);
        write_pos = 0; count = 0;
    }
    void push_row(const float* vals) {
        for (int i = 0; i < N; ++i)
            data[write_pos * N + i] = vals[i];
        write_pos = (write_pos + 1) % max_rows;
        if (count < max_rows) ++count;
    }
    float get(int row, int col) const {
        int actual = (write_pos - count + row + max_rows) % max_rows;
        return data[actual * N + col];
    }
    void clear() { write_pos = 0; count = 0; data.assign(N * max_rows, 1.0f); }
};

static ImU32 health_color(float h) {
    h = std::clamp(h, 0.0f, 1.0f);
    int r, g, b;
    if (h > 0.5f) {
        float t = (h - 0.5f) * 2.0f;
        r = (int)(240 - 220*t); g = (int)(210 - 110*t); b = (int)(30 + 190*t);
    } else {
        float t = h * 2.0f;
        r = (int)(220 + 20*t); g = (int)(30 + 180*t); b = 30;
    }
    return IM_COL32(r, g, b, 255);
}

static constexpr int DIM = 2;

struct GuiState {
    std::unique_ptr<aniso::Engine<DIM>> engine;
    bool running  = false;
    bool has_data = false;
    int  steps_per_frame = 10;

    static constexpr int BUF_CAP = 12000;
    ScrollBuf t_buf;
    ScrollBuf x_buf[DIM], u_buf[DIM], eig_buf[DIM];
    ScrollBuf trG_buf, Qnorm_buf, effort_buf, err_buf;

    double ell_cx[65], ell_cy[65];
    ImFont* font_big = nullptr;

    // Trail for phase portrait (last N points)
    static constexpr int TRAIL = 600;
    ScrollBuf trail_x, trail_y;

    // Interactive parameters (sliders drive these, changes trigger rebuild)
    float tau = 5.0f;
    float ctrl_gain = 1.5f;
    float ctrl_umax = 3.0f;
    float coupling_alpha = 0.5f;
    float obs_beta = 0.5f;
    float obs_sigma0 = 0.04f;
    float interact_mu = 0.8f;
    float eig_lo = 0.3f, eig_hi = 5.0f;
    bool params_dirty = false;   // true when a slider changed
    bool auto_run = true;        // auto-start after rebuild

    // Convergence tracking
    double avg_error_recent = 0.0;
    double avg_error_old    = 0.0;
    int    convergence_state = 0; // -1 diverging, 0 unknown, 1 converging, 2 settled
    float  stability_index = 0.5f; // 0 = unstable, 1 = fully stable (composite)

    // Pareto grid results
    std::vector<aniso::BenchResult<DIM>> pareto_grid;
    bool pareto_ran = false;

    std::string config_path = "configs/phase_transition.yaml";
    char config_buf[256] = "configs/phase_transition.yaml";
    YAML::Node current_cfg; // kept for live rebuild

    // Controller switching
    std::vector<std::string> ctrl_names;
    int ctrl_idx = 0;

    // ---- Chain (1D spatial) ----
    std::unique_ptr<aniso::ChainEngine<DIM>> chain;
    bool chain_running = false;
    bool chain_loaded  = false;
    bool chain_just_loaded = false;
    int  chain_steps_per_frame = 20;
    float ch_D_G = 0.05f, ch_D_x = 0.1f;
    float ch_drive_peak = 5.0f, ch_drive_width = 0.15f, ch_drive_center = 0.5f;
    float ch_tau = 5.0f;
    float ch_gain = 1.0f, ch_umax = 3.0f, ch_alpha = 0.7f, ch_mu = 1.5f;
    float ch_trap = 10.0f, ch_s_crit = 2.0f;
    int ch_ctrl_idx = 0;
    Kymograph kymo;

    // ---- Grid (2D spatial) ----
    std::unique_ptr<aniso::GridEngine<DIM>> grid;
    bool grid_running = false;
    bool grid_loaded  = false;
    bool grid_just_loaded = false;
    int  grid_steps_per_frame = 100;
    float gr_D_E = 0.5f, gr_D_x = 0.1f;
    float gr_drive_peak = 5.0f, gr_drive_rx = 0.18f, gr_drive_ry = 0.18f;
    float gr_drive_cx = 0.5f, gr_drive_cy = 0.5f;
    float gr_tau = 1.0f;
    float gr_gain = 3.0f, gr_umax = 20.0f, gr_alpha = 0.5f;
    float gr_gamma_diss = 1.0f;
    float gr_kappa_tau = 20.0f;
    float gr_noise_amp = 0.5f;
    int gr_ctrl_idx = 0;
    int gr_map_mode = 0;   // 0=Health 1=Energy 2=|x| 3=|u|
    float gr_period = 4.0f, gr_duty = 0.5f;
    float gr_trigger = 0.5f, gr_hyst = 0.6f, gr_antic = 5.0f;

    // Time-history ring buffers for Grid tab
    static constexpr int GR_HIST = 600;
    float gr_hist_x[GR_HIST]  = {};
    float gr_hist_E[GR_HIST]  = {};
    float gr_hist_tr[GR_HIST] = {};
    int   gr_hist_pos = 0;

    // Benchmark results
    std::vector<aniso::BenchResult<DIM>> bench_results;
    bool bench_ran = false;

    // Sweep results
    std::vector<aniso::SweepRow<DIM>> sweep_rows;
    bool sweep_ran = false;
    char sweep_param[128] = "relaxation.tau";
    float sweep_lo = 1.0f, sweep_hi = 15.0f, sweep_step = 1.0f;

    void init_buffers() {
        t_buf.init(BUF_CAP); trG_buf.init(BUF_CAP);
        Qnorm_buf.init(BUF_CAP); effort_buf.init(BUF_CAP); err_buf.init(BUF_CAP);
        trail_x.init(TRAIL); trail_y.init(TRAIL);
        for (int d = 0; d < DIM; ++d) {
            x_buf[d].init(BUF_CAP); u_buf[d].init(BUF_CAP); eig_buf[d].init(BUF_CAP);
        }
    }
    void clear_buffers() {
        t_buf.clear(); trG_buf.clear(); Qnorm_buf.clear(); effort_buf.clear(); err_buf.clear();
        trail_x.clear(); trail_y.clear();
        for (int d = 0; d < DIM; ++d) {
            x_buf[d].clear(); u_buf[d].clear(); eig_buf[d].clear();
        }
    }
};

// ---------------------------------------------------------------------------
// ImPlot helpers
// ---------------------------------------------------------------------------
struct ScrollGetter { const ScrollBuf* t; const ScrollBuf* y; };

static ImPlotPoint scroll_getter(int idx, void* data) {
    auto* sg = static_cast<ScrollGetter*>(data);
    return {(*sg->t)[idx], (*sg->y)[idx]};
}
static void plot_scroll(const char* label, ScrollBuf& t_buf, ScrollBuf& y_buf) {
    ScrollGetter sg{&t_buf, &y_buf};
    ImPlot::PlotLineG(label, scroll_getter, &sg, t_buf.size);
}

struct XYGetter { const ScrollBuf* x; const ScrollBuf* y; };
static ImPlotPoint xy_getter(int idx, void* data) {
    auto* g = static_cast<XYGetter*>(data);
    return {(*g->x)[idx], (*g->y)[idx]};
}

// ---------------------------------------------------------------------------
// Compute ellipse points from G matrix
// ---------------------------------------------------------------------------
static void compute_ellipse(const aniso::TensorField<DIM>& G,
                            double* cx, double* cy, int n = 64) {
    Eigen::SelfAdjointEigenSolver<aniso::Mat<DIM>> solver(G.G);
    auto ev = solver.eigenvalues();
    auto evec = solver.eigenvectors();
    double a = std::sqrt(std::max(ev(0), 0.01));
    double b = std::sqrt(std::max(ev(1), 0.01));
    for (int i = 0; i <= n; ++i) {
        double th = 2.0 * M_PI * i / n;
        double lx = a * std::cos(th);
        double ly = b * std::sin(th);
        cx[i] = evec(0,0)*lx + evec(0,1)*ly;
        cy[i] = evec(1,0)*lx + evec(1,1)*ly;
    }
}

// ---------------------------------------------------------------------------
// Arrow helper: draw an arrow from (x0,y0) to (x0+dx, y0+dy) on the current plot
// ---------------------------------------------------------------------------
static void plot_arrow(const char* id, double x0, double y0, double dx, double dy,
                       ImVec4 color, float thickness = 2.5f) {
    double x1 = x0 + dx, y1 = y0 + dy;
    double ax[] = {x0, x1}, ay[] = {y0, y1};
    ImPlot::PushStyleColor(ImPlotCol_Line, color);
    ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, thickness);
    ImPlot::PlotLine(id, ax, ay, 2);
    ImPlot::PopStyleVar();
    ImPlot::PopStyleColor();

    // Arrowhead
    double len = std::sqrt(dx*dx + dy*dy);
    if (len < 1e-6) return;
    double ux = dx/len, uy = dy/len;
    double head = std::min(len * 0.3, 0.25);
    double px = -uy, py = ux;
    double hx[] = {x1, x1 - head*ux + head*0.4*px, x1 - head*ux - head*0.4*px, x1};
    double hy[] = {y1, y1 - head*uy + head*0.4*py, y1 - head*uy - head*0.4*py, y1};
    ImPlot::PushStyleColor(ImPlotCol_Line, color);
    ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, thickness);
    ImPlot::PlotLine(id, hx, hy, 4);
    ImPlot::PopStyleVar();
    ImPlot::PopStyleColor();
}

// ---------------------------------------------------------------------------
// Arc gauge helper for dashboard
// ---------------------------------------------------------------------------
static void draw_gauge(ImDrawList* dl, ImVec2 center, float radius, float value,
                       const char* label, const char* val_text, ImU32 fill_col) {
    const float a0 = (float)M_PI * 0.8f;
    const float a1 = (float)M_PI * 2.2f;
    const float sweep = a1 - a0;

    dl->PathArcTo(center, radius, a0, a1, 40);
    dl->PathStroke(IM_COL32(40, 42, 52, 200), 0, 5.0f);

    float v = std::clamp(value, 0.0f, 1.0f);
    if (v > 0.005f) {
        dl->PathArcTo(center, radius, a0, a0 + sweep * v, std::max(4, (int)(40*v)));
        dl->PathStroke(fill_col, 0, 7.0f);
    }

    for (float f : {0.0f, 0.25f, 0.5f, 0.75f, 1.0f}) {
        float ta = a0 + sweep * f;
        float ri = radius - 10, ro = radius + 2;
        dl->AddLine({center.x + ri*cosf(ta), center.y + ri*sinf(ta)},
                    {center.x + ro*cosf(ta), center.y + ro*sinf(ta)},
                    IM_COL32(70, 72, 80, 150), 1.5f);
    }

    auto vsz = ImGui::CalcTextSize(val_text);
    dl->AddText({center.x - vsz.x*0.5f, center.y - vsz.y*0.6f},
                IM_COL32(225, 225, 230, 255), val_text);

    auto lsz = ImGui::CalcTextSize(label);
    dl->AddText({center.x - lsz.x*0.5f, center.y + lsz.y*0.3f},
                IM_COL32(140, 140, 155, 200), label);
}

// ---------------------------------------------------------------------------
// Sync slider values FROM the loaded config
// ---------------------------------------------------------------------------
static void sync_params_from_config(GuiState& gs) {
    auto& p = gs.engine->params();
    gs.tau = static_cast<float>(p.tau);
    gs.eig_lo = static_cast<float>(p.eig_lo);
    gs.eig_hi = static_cast<float>(p.eig_hi);

    // Extract controller params from YAML
    auto ctrl = gs.current_cfg["controller"];
    if (ctrl) {
        gs.ctrl_gain = ctrl["gain"].as<float>(1.5f);
        gs.ctrl_umax = ctrl["u_max"].as<float>(3.0f);
    }
    auto coupling = gs.current_cfg["coupling"];
    if (coupling) {
        gs.coupling_alpha = coupling["alpha"].as<float>(0.5f);
    }
    auto obs = gs.current_cfg["observation"];
    if (obs) {
        gs.obs_sigma0 = obs["sigma0"].as<float>(0.04f);
        gs.obs_beta = obs["beta"].as<float>(0.5f);
    }
    auto inter = gs.current_cfg["interaction"];
    if (inter) {
        gs.interact_mu = inter["mu"].as<float>(0.8f);
    }
}

static void set_yaml_path(YAML::Node root, const std::string& path, double value) {
    size_t dot = path.find('.');
    if (dot == std::string::npos) {
        root[path] = value;
    } else {
        std::string key = path.substr(0, dot);
        std::string rest = path.substr(dot + 1);
        size_t dot2 = rest.find('.');
        if (dot2 == std::string::npos)
            root[key][rest] = value;
        else
            root[key][rest.substr(0, dot2)][rest.substr(dot2 + 1)] = value;
    }
}

// ---------------------------------------------------------------------------
// Apply slider values INTO the YAML and rebuild engine
// ---------------------------------------------------------------------------
static bool rebuild_engine(GuiState& gs) {
    try {
        YAML::Node& cfg = gs.current_cfg;

        // Push slider values into YAML
        cfg["relaxation"]["tau"] = static_cast<double>(gs.tau);
        cfg["eigenvalue_clamp"]["lo"] = static_cast<double>(gs.eig_lo);
        cfg["eigenvalue_clamp"]["hi"] = static_cast<double>(gs.eig_hi);

        if (cfg["controller"]) {
            cfg["controller"]["gain"] = static_cast<double>(gs.ctrl_gain);
            cfg["controller"]["u_max"] = static_cast<double>(gs.ctrl_umax);
        }
        if (cfg["coupling"]) {
            cfg["coupling"]["alpha"] = static_cast<double>(gs.coupling_alpha);
        }
        if (cfg["observation"]) {
            cfg["observation"]["sigma0"] = static_cast<double>(gs.obs_sigma0);
            cfg["observation"]["beta"] = static_cast<double>(gs.obs_beta);
        }
        if (cfg["interaction"]) {
            cfg["interaction"]["mu"] = static_cast<double>(gs.interact_mu);
        }

        gs.engine = std::make_unique<aniso::Engine<DIM>>(aniso::build_engine<DIM>(cfg));
        gs.engine->reset();
        gs.clear_buffers();
        gs.has_data = false;
        gs.convergence_state = 0;
        gs.avg_error_recent = gs.avg_error_old = 0.0;
        if (gs.auto_run) gs.running = true;
        gs.params_dirty = false;
        return true;
    } catch (...) { return false; }
}

// ---------------------------------------------------------------------------
// Rebuild GridEngine from current config + set GUI sliders
// ---------------------------------------------------------------------------
static bool rebuild_grid(GuiState& gs) {
    try {
        YAML::Node& cfg = gs.current_cfg;
        if (!cfg["grid"]) return false;

        gs.grid = std::make_unique<aniso::GridEngine<DIM>>(
            aniso::build_grid<DIM>(cfg));
        gs.grid->reset();
        gs.grid_loaded = true;
        gs.grid_just_loaded = true;
        gs.grid_running = false;

        auto& gp = gs.grid->params();
        gs.gr_D_E        = (float)gp.D_E;
        gs.gr_D_x        = (float)gp.D_x;
        gs.gr_drive_peak = (float)gp.drive_peak;
        gs.gr_drive_rx   = (float)gp.drive_rx;
        gs.gr_drive_ry   = (float)gp.drive_ry;
        gs.gr_drive_cx   = (float)gp.drive_cx;
        gs.gr_drive_cy   = (float)gp.drive_cy;
        gs.gr_tau        = (float)gp.tau_0;
        gs.gr_gamma_diss = (float)gp.gamma_diss;
        gs.gr_kappa_tau  = (float)gp.kappa_tau;
        gs.gr_noise_amp  = (float)gp.noise_amp;

        auto ctrl_n = cfg["controller"];
        if (ctrl_n) {
            gs.gr_gain = ctrl_n["gain"].as<float>(3.0f);
            gs.gr_umax = ctrl_n["u_max"].as<float>(20.0f);
        }
        auto coup_n = cfg["coupling"];
        if (coup_n) gs.gr_alpha = coup_n["alpha"].as<float>(0.5f);

        std::memset(gs.gr_hist_x, 0, sizeof(gs.gr_hist_x));
        std::memset(gs.gr_hist_E, 0, sizeof(gs.gr_hist_E));
        std::memset(gs.gr_hist_tr, 0, sizeof(gs.gr_hist_tr));
        gs.gr_hist_pos = 0;

        return true;
    } catch (...) { return false; }
}

// ---------------------------------------------------------------------------
// Load config and build engine
// ---------------------------------------------------------------------------
static bool load_config(GuiState& gs) {
    try {
        gs.current_cfg = YAML::LoadFile(gs.config_path);
        int dim = gs.current_cfg["dim"].as<int>(2);
        if (dim != DIM) return false;

        // Build controller list from YAML
        gs.ctrl_names.clear();
        gs.ctrl_idx = 0;
        if (gs.current_cfg["controllers"]
            && gs.current_cfg["controllers"].IsSequence()) {
            for (size_t i = 0; i < gs.current_cfg["controllers"].size(); ++i)
                gs.ctrl_names.push_back(
                    gs.current_cfg["controllers"][i]["name"]
                        .as<std::string>("ctrl_" + std::to_string(i)));
            gs.current_cfg["controller"] = gs.current_cfg["controllers"][0];
        } else if (gs.current_cfg["controller"]) {
            gs.ctrl_names.push_back(
                gs.current_cfg["controller"]["type"].as<std::string>("ctrl"));
        }

        gs.engine = std::make_unique<aniso::Engine<DIM>>(
            aniso::build_engine<DIM>(gs.current_cfg));
        sync_params_from_config(gs);
        gs.engine->reset();
        gs.clear_buffers();
        gs.has_data = false;
        gs.running  = false;
        gs.bench_ran = false;
        gs.bench_results.clear();
        gs.sweep_ran = false;
        gs.sweep_rows.clear();
        gs.convergence_state = 0;

        // Build chain if config has chain section
        gs.chain.reset();
        gs.chain_loaded  = false;
        gs.chain_running = false;
        gs.chain_just_loaded = false;
        if (gs.current_cfg["chain"]) {
            gs.chain = std::make_unique<aniso::ChainEngine<DIM>>(
                aniso::build_chain<DIM>(gs.current_cfg));
            gs.chain->reset();
            gs.chain_loaded = true;
            gs.chain_just_loaded = true;
            auto& cp = gs.chain->params();
            gs.ch_D_G          = (float)cp.D_G;
            gs.ch_D_x          = (float)cp.D_x;
            gs.ch_drive_peak   = (float)cp.drive_peak;
            gs.ch_drive_width  = (float)cp.drive_width;
            gs.ch_drive_center = (float)cp.drive_center;
            gs.ch_tau          = (float)cp.tau;
            gs.ch_trap   = (float)cp.trap;
            gs.ch_s_crit = (float)cp.s_crit;
            auto ctrl_n = gs.current_cfg["controller"];
            if (ctrl_n) {
                gs.ch_gain  = ctrl_n["gain"].as<float>(1.0f);
                gs.ch_umax  = ctrl_n["u_max"].as<float>(3.0f);
            }
            auto coup_n = gs.current_cfg["coupling"];
            if (coup_n) gs.ch_alpha = coup_n["alpha"].as<float>(0.7f);
            auto inter_n = gs.current_cfg["interaction"];
            if (inter_n) gs.ch_mu = inter_n["mu"].as<float>(1.5f);
            gs.kymo.init(cp.N, 400);
        }

        // Build grid if config has grid section
        gs.grid.reset();
        gs.grid_loaded  = false;
        gs.grid_running = false;
        gs.grid_just_loaded = false;
        if (gs.current_cfg["grid"]) {
            gs.grid = std::make_unique<aniso::GridEngine<DIM>>(
                aniso::build_grid<DIM>(gs.current_cfg));
            gs.grid->reset();
            gs.grid_loaded = true;
            gs.grid_just_loaded = true;
            auto& gp = gs.grid->params();
            gs.gr_D_E        = (float)gp.D_E;
            gs.gr_D_x        = (float)gp.D_x;
            gs.gr_drive_peak = (float)gp.drive_peak;
            gs.gr_drive_rx   = (float)gp.drive_rx;
            gs.gr_drive_ry   = (float)gp.drive_ry;
            gs.gr_drive_cx   = (float)gp.drive_cx;
            gs.gr_drive_cy   = (float)gp.drive_cy;
            gs.gr_tau        = (float)gp.tau_0;
            gs.gr_gamma_diss = (float)gp.gamma_diss;
            gs.gr_kappa_tau  = (float)gp.kappa_tau;
            gs.gr_noise_amp  = (float)gp.noise_amp;
            auto ctrl_n = gs.current_cfg["controller"];
            if (ctrl_n) {
                gs.gr_gain = ctrl_n["gain"].as<float>(3.0f);
                gs.gr_umax = ctrl_n["u_max"].as<float>(20.0f);
            }
            auto coup_n = gs.current_cfg["coupling"];
            if (coup_n) gs.gr_alpha = coup_n["alpha"].as<float>(0.5f);
        }

        return true;
    } catch (...) { return false; }
}

// ---------------------------------------------------------------------------
// Advance simulation
// ---------------------------------------------------------------------------
static void sim_step(GuiState& gs) {
    if (!gs.engine || gs.engine->done()) return;
    if (!gs.engine->step()) { gs.running = false; return; }

    auto& st = gs.engine->state();
    auto& u  = gs.engine->last_u();
    auto  ev = st.G.eigenvalues();

    gs.t_buf.push(st.t);
    for (int d = 0; d < DIM; ++d) {
        gs.x_buf[d].push(st.x(d));
        gs.u_buf[d].push(u(d));
        gs.eig_buf[d].push(ev(d));
    }
    gs.trG_buf.push(st.G.trace());
    gs.Qnorm_buf.push(std::sqrt(st.G.traceless_norm_sq()));
    gs.effort_buf.push(u.squaredNorm());
    gs.err_buf.push(st.x.norm());
    gs.trail_x.push(st.x(0));
    gs.trail_y.push(st.x(1));
    gs.has_data = true;
}

// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    if (!glfwInit()) return 1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_FOCUSED, GLFW_TRUE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);

    GLFWwindow* window = glfwCreateWindow(1600, 950,
        "Aniso \xE2\x80\x94 Tensor Degradation Simulator", nullptr, nullptr);
    if (!window) { glfwTerminate(); return 1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glfwShowWindow(window);
    glfwFocusWindow(window);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGui::StyleColorsDark();
    auto& style = ImGui::GetStyle();
    style.FrameRounding = 4.0f;
    style.GrabRounding  = 4.0f;
    style.WindowRounding = 6.0f;

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    ImGui::GetIO().Fonts->AddFontDefault();
    ImFont* _font_big = nullptr;
    { ImFontConfig fc; fc.SizePixels = 24.0f; _font_big = ImGui::GetIO().Fonts->AddFontDefault(&fc); }

    GuiState gs;
    gs.font_big = _font_big;
    gs.init_buffers();
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "--config" || a == "-c") && i + 1 < argc) {
            gs.config_path = argv[++i];
        } else {
            gs.config_path = argv[i];
        }
    }
    std::snprintf(gs.config_buf, sizeof(gs.config_buf), "%s",
                  gs.config_path.c_str());
    load_config(gs);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        if (gs.chain_running && gs.chain && !gs.chain->done()) {
            for (int i = 0; i < gs.chain_steps_per_frame; ++i)
                gs.chain->step();
            if (gs.kymo.N > 0) {
                std::vector<float> row(gs.chain->N());
                for (int j = 0; j < gs.chain->N(); ++j)
                    row[j] = (float)gs.chain->health(j);
                gs.kymo.push_row(row.data());
            }
        }
        if (gs.grid_running && gs.grid) {
            for (int i = 0; i < gs.grid_steps_per_frame; ++i)
                gs.grid->step();
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // ================================================================
        //  LEFT SIDEBAR
        // ================================================================
        ImGui::SetNextWindowPos({0, 0}, ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize({310, 950}, ImGuiCond_FirstUseEver);
        ImGui::Begin("Controls");

        ImGui::SeparatorText("Configuration");
        ImGui::SetNextItemWidth(-1);
        ImGui::InputText("##yaml", gs.config_buf, sizeof(gs.config_buf));
        if (ImGui::Button("Load & Reset", {-1, 0})) {
            gs.config_path = gs.config_buf;
            load_config(gs);
        }

        ImGui::End();

        // ================================================================
        //  MAIN PLOT AREA
        // ================================================================
        ImGui::SetNextWindowPos({320, 0}, ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize({1270, 950}, ImGuiCond_FirstUseEver);
        ImGui::Begin("Plots");

        if (ImGui::BeginTabBar("Tabs")) {

            // REMOVED: Dashboard, Control, Tensor G tabs
#if 0
                {
                    ImVec2 cp = ImGui::GetCursorScreenPos();
                    ImVec2 cs = ImGui::GetContentRegionAvail();
                    float spark_h = 70.0f;
                    float canvas_h = cs.y - spark_h - 4;
                    ImGui::InvisibleButton("##cv", {cs.x, canvas_h});
                    ImDrawList* dl = ImGui::GetWindowDrawList();

                    float mx = cp.x + cs.x * 0.5f;
                    float my = cp.y + canvas_h * 0.52f;
                    float vr = 4.0f;
                    float sc = std::min(cs.x, canvas_h) * 0.40f / vr;

                    // Canvas background tinted by stability index
                    {
                        float si = gs.stability_index;
                        int bg_r = (int)(8 + 30*(1-si));
                        int bg_g = (int)(10 + 18*si);
                        int bg_b = (int)(18 - 6*(1-si));
                        dl->AddRectFilled(cp, {cp.x + cs.x, cp.y + canvas_h},
                            IM_COL32(bg_r, bg_g, bg_b, 255));

                        // Pulsing border when unstable
                        if (si < 0.4f) {
                            float pulse = 0.5f + 0.5f * sinf((float)glfwGetTime() * 4.0f);
                            int ba = (int)((0.4f - si) * pulse * 400);
                            ba = std::clamp(ba, 0, 200);
                            dl->AddRect(cp, {cp.x + cs.x, cp.y + canvas_h},
                                IM_COL32(255, 40, 20, ba), 0, 0, 2.5f);
                        } else if (si > 0.75f) {
                            dl->AddRect(cp, {cp.x + cs.x, cp.y + canvas_h},
                                IM_COL32(30, 180, 60, 40), 0, 0, 1.5f);
                        }
                    }

                    for (int i = -4; i <= 4; ++i) {
                        float gx = mx + i * sc;
                        float gy = my + i * sc;
                        if (gx > cp.x && gx < cp.x + cs.x)
                            dl->AddLine({gx, cp.y}, {gx, cp.y + canvas_h},
                                IM_COL32(25, 30, 40, 60));
                        if (gy > cp.y && gy < cp.y + canvas_h)
                            dl->AddLine({cp.x, gy}, {cp.x + cs.x, gy},
                                IM_COL32(25, 30, 40, 60));
                    }

                    for (int i = 1; i <= 8; ++i) {
                        float r = i * 0.5f * sc;
                        int alpha = std::max(8, 55 - i * 6);
                        dl->AddCircle({mx, my}, r,
                            IM_COL32(30, 140, 65, alpha), 64, 1.2f);
                    }

                    dl->AddLine({cp.x, my}, {cp.x + cs.x, my},
                        IM_COL32(30, 55, 38, 50));
                    dl->AddLine({mx, cp.y}, {mx, cp.y + canvas_h},
                        IM_COL32(30, 55, 38, 50));

                    dl->AddCircleFilled({mx, my}, 5,
                        IM_COL32(50, 190, 75, 200));
                    dl->AddCircle({mx, my}, 5,
                        IM_COL32(90, 255, 110, 80));

                    if (gs.has_data && gs.engine) {
                        auto& st = gs.engine->state();
                        auto& u_vec = gs.engine->last_u();

                        // --- FOG: G tensor as translucent ellipse ---
                        {
                            Eigen::SelfAdjointEigenSolver<aniso::Mat<DIM>> sol(st.G.G);
                            auto ev = sol.eigenvalues();
                            auto evc = sol.eigenvectors();

                            float a_ax = sqrtf(std::max((float)ev(0), 0.01f)) * sc;
                            float b_ax = sqrtf(std::max((float)ev(1), 0.01f)) * sc;
                            float aniso_q = (float)std::sqrt(st.G.traceless_norm_sq());
                            float tr_ex = std::max(0.0f,
                                (float)(st.G.trace() - DIM) / DIM);
                            float bad = std::min(1.0f,
                                aniso_q * 0.4f + tr_ex * 0.3f);

                            int fog_a = (int)(std::min(0.45f,
                                bad * 0.45f + 0.04f) * 255);
                            ImU32 fog_fill = IM_COL32(
                                (int)(35 + 185*bad),
                                (int)(55 + 25*(1-bad)),
                                (int)(105 - 65*bad), fog_a);

                            const int NE = 48;
                            ImVec2 epts[49];
                            float rot = atan2f((float)evc(1,0),
                                               (float)evc(0,0));
                            float cr = cosf(rot), sr = sinf(rot);
                            for (int i = 0; i <= NE; ++i) {
                                float th = 2.0f * (float)M_PI * i / NE;
                                float lx = a_ax * cosf(th);
                                float ly = b_ax * sinf(th);
                                epts[i] = {mx + lx*cr - ly*sr,
                                           my - (lx*sr + ly*cr)};
                            }
                            dl->AddConvexPolyFilled(epts, NE + 1, fog_fill);

                            ImU32 fog_line = IM_COL32(
                                (int)(75 + 145*bad),
                                (int)(95 + 35*(1-bad)),
                                (int)(135 - 80*bad), 110);
                            dl->AddPolyline(epts, NE + 1, fog_line,
                                ImDrawFlags_Closed, 1.5f);
                        }

                        // --- TRAIL: recent trajectory ---
                        {
                            int tn = gs.trail_x.size;
                            if (tn > 1) {
                                for (int i = 1; i < tn; ++i) {
                                    float ta = (float)i / tn;
                                    int al = (int)(ta * ta * 150);
                                    dl->AddLine(
                                        {mx + (float)gs.trail_x[i-1]*sc,
                                         my - (float)gs.trail_y[i-1]*sc},
                                        {mx + (float)gs.trail_x[i]*sc,
                                         my - (float)gs.trail_y[i]*sc},
                                        IM_COL32(85, 135, 225, al), 1.8f);
                                }
                            }
                        }

                        // --- BALL: current state position ---
                        float bx = mx + (float)st.x(0) * sc;
                        float by = my - (float)st.x(1) * sc;
                        float err = (float)st.x.norm();
                        float ef = std::min(err / 3.0f, 1.0f);

                        dl->AddCircleFilled({bx, by}, 16,
                            IM_COL32(255, 255, 255, 12));
                        dl->AddCircleFilled({bx, by}, 11,
                            IM_COL32(255, 255, 255, 30));

                        ImU32 ball_c = IM_COL32(
                            (int)(75 + 180*ef),
                            (int)(235 - 185*ef),
                            (int)(75 - 35*ef), 255);
                        dl->AddCircleFilled({bx, by}, 8, ball_c);
                        dl->AddCircle({bx, by}, 8,
                            IM_COL32(255, 255, 255, 150), 0, 1.5f);

                        // --- CONTROL ARROW ---
                        float un = (float)u_vec.norm();
                        if (un > 0.01f) {
                            float asc = sc * 0.35f;
                            float adx = (float)u_vec(0) * asc;
                            float ady = -(float)u_vec(1) * asc;
                            float efr = std::min(un / gs.ctrl_umax, 1.0f);
                            ImU32 acol = IM_COL32(
                                (int)(55 + 200*efr),
                                (int)(215 - 175*efr), 55, 220);

                            dl->AddLine({bx, by}, {bx+adx, by+ady},
                                acol, 2.8f);

                            float alen = sqrtf(adx*adx + ady*ady);
                            if (alen > 4.0f) {
                                float anx = adx/alen, any = ady/alen;
                                float hd = std::min(alen*0.3f, 11.0f);
                                float pnx = -any, pny = anx;
                                ImVec2 tip = {bx+adx, by+ady};
                                ImVec2 lft = {tip.x - hd*anx + hd*0.4f*pnx,
                                              tip.y - hd*any + hd*0.4f*pny};
                                ImVec2 rgt = {tip.x - hd*anx - hd*0.4f*pnx,
                                              tip.y - hd*any - hd*0.4f*pny};
                                dl->AddTriangleFilled(tip, lft, rgt, acol);
                            }
                        }

                        // --- Distance label near ball ---
                        char dist_txt[32];
                        std::snprintf(dist_txt, 32, "%.2f", err);
                        auto dsz = ImGui::CalcTextSize(dist_txt);
                        dl->AddText({bx - dsz.x*0.5f, by - 22},
                            IM_COL32(200, 200, 210, 170), dist_txt);

                        // --- BIG STATUS TEXT with stability % ---
                        char stxt_buf[64];
                        const char* stxt;
                        ImU32 scol;
                        float si = gs.stability_index;
                        switch (gs.convergence_state) {
                            case 2:
                                std::snprintf(stxt_buf, 64,
                                    "STABLE  %.0f%%", si*100);
                                scol = IM_COL32(45, 235, 65, 255);
                                break;
                            case 1:
                                std::snprintf(stxt_buf, 64,
                                    "CONVERGING  %.0f%%", si*100);
                                scol = IM_COL32(75, 210, 115, 210);
                                break;
                            case -1:
                                std::snprintf(stxt_buf, 64,
                                    "UNSTABLE  %.0f%%", si*100);
                                scol = IM_COL32(255, 55, 35, 255);
                                break;
                            default:
                                std::snprintf(stxt_buf, 64,
                                    "MARGINAL  %.0f%%", si*100);
                                scol = IM_COL32(215, 185, 35, 210);
                                break;
                        }
                        stxt = stxt_buf;
                        if (gs.font_big) {
                            auto tsz = gs.font_big->CalcTextSizeA(
                                24, FLT_MAX, 0, stxt);
                            dl->AddText(gs.font_big, 24,
                                {mx - tsz.x*0.5f, cp.y + 10}, scol, stxt);
                        } else {
                            auto tsz = ImGui::CalcTextSize(stxt);
                            dl->AddText({mx - tsz.x*0.5f, cp.y + 14},
                                scol, stxt);
                        }

                        // --- Legend (bottom of canvas, small) ---
                        const char* legend =
                            "Ring=target  Ball=state  Fog=degradation  "
                            "Arrow=control (green=easy, red=hard)";
                        auto lgsz = ImGui::CalcTextSize(legend);
                        dl->AddText(
                            {mx - lgsz.x*0.5f, cp.y + canvas_h - 18},
                            IM_COL32(100, 100, 115, 140), legend);

                    } else {
                        const char* msg =
                            "Load a configuration and press Run";
                        auto msz = ImGui::CalcTextSize(msg);
                        dl->AddText({mx - msz.x*0.5f, my - msz.y*0.5f},
                            IM_COL32(110, 110, 120, 180), msg);
                    }

                    // ---- SPARKLINE (below canvas) ----
                    ImVec2 sp = ImGui::GetCursorScreenPos();
                    ImVec2 ss = {cs.x, spark_h};
                    ImGui::InvisibleButton("##spk", ss);

                    dl->AddRectFilled(sp,
                        {sp.x + ss.x, sp.y + ss.y},
                        IM_COL32(12, 14, 22, 255));

                    int en = gs.err_buf.size;
                    if (en > 2) {
                        float y_max = 4.0f;
                        int step = std::max(1, en / (int)ss.x);
                        float px_per = ss.x / (float)(en / step);

                        for (int i = step; i < en; i += step) {
                            float x0 = sp.x + (float)((i-step)/step)
                                * px_per;
                            float x1 = sp.x + (float)(i/step) * px_per;
                            float e0 = (float)gs.err_buf[i - step];
                            float e1 = (float)gs.err_buf[i];
                            float fy0 = sp.y + ss.y
                                - (e0/y_max) * (ss.y - 4);
                            float fy1 = sp.y + ss.y
                                - (e1/y_max) * (ss.y - 4);
                            fy0 = std::clamp(fy0, sp.y, sp.y + ss.y);
                            fy1 = std::clamp(fy1, sp.y, sp.y + ss.y);

                            float emid = (e0 + e1) * 0.5f;
                            float ec = std::min(emid / 2.0f, 1.0f);
                            ImU32 lc = IM_COL32(
                                (int)(75 + 180*ec),
                                (int)(215 - 165*ec), 55, 200);
                            dl->AddLine({x0, fy0}, {x1, fy1}, lc, 1.5f);
                        }

                        dl->AddLine(
                            {sp.x, sp.y + ss.y - 1},
                            {sp.x + ss.x, sp.y + ss.y - 1},
                            IM_COL32(40, 42, 50, 100));

                        dl->AddText({sp.x + 6, sp.y + 4},
                            IM_COL32(150, 150, 160, 170),
                            "Error |x| over time");
                    }
                }
                ImGui::EndChild();

                ImGui::SameLine();

                // ---- GAUGES PANEL (right) ----
                ImGui::BeginChild("##dash_gauges", {gauge_w, main_h},
                    true, ImGuiWindowFlags_NoScrollbar);
                {
                    if (gs.has_data && gs.engine) {
                        auto& st = gs.engine->state();
                        auto& u_vec = gs.engine->last_u();
                        ImDrawList* gdl = ImGui::GetWindowDrawList();
                        ImVec2 gp = ImGui::GetCursorScreenPos();
                        float gw = ImGui::GetContentRegionAvail().x;

                        float err = (float)st.x.norm();
                        float effort = (float)u_vec.norm()
                            / std::max(gs.ctrl_umax, 0.01f);
                        float trace_g = (float)st.G.trace();
                        float health = 1.0f - std::min(1.0f,
                            (trace_g - DIM) /
                            std::max(gs.eig_hi * DIM - DIM, 0.01f));

                        float gr = std::min(gw * 0.32f, 48.0f);
                        float gcx = gp.x + gw * 0.5f;
                        float spacing = gr * 2.4f;

                        // Gauge 1: Accuracy
                        float acc = 1.0f - std::min(err / 3.0f, 1.0f);
                        float gy1 = gp.y + gr + 30;
                        ImU32 acc_col = IM_COL32(
                            (int)(240*(1-acc)), (int)(200*acc), 50, 255);
                        char acc_txt[16];
                        std::snprintf(acc_txt, 16, "%.0f%%", acc*100);
                        draw_gauge(gdl, {gcx, gy1}, gr, acc,
                            "ACCURACY", acc_txt, acc_col);

                        // Gauge 2: Effort
                        float eff_v = std::min(effort, 1.0f);
                        float gy2 = gy1 + spacing;
                        ImU32 eff_col = IM_COL32(
                            (int)(60 + 195*eff_v),
                            (int)(200 - 160*eff_v), 50, 255);
                        char eff_txt[16];
                        std::snprintf(eff_txt, 16, "%.0f%%", eff_v*100);
                        draw_gauge(gdl, {gcx, gy2}, gr, eff_v,
                            "EFFORT", eff_txt, eff_col);

                        // Gauge 3: Field Health
                        float gy3 = gy2 + spacing;
                        ImU32 hp_col = IM_COL32(
                            (int)(230*(1-health)),
                            (int)(200*health), 60, 255);
                        char hp_txt[16];
                        std::snprintf(hp_txt, 16, "%.0f%%", health*100);
                        draw_gauge(gdl, {gcx, gy3}, gr, health,
                            "HEALTH", hp_txt, hp_col);

                        // Metrics text below gauges
                        float ty = gy3 + gr + 25;
                        ImGui::SetCursorScreenPos({gp.x + 8, ty});

                        ImGui::TextColored({0.6f,0.65f,0.7f,1},
                            "t = %.1f / %.0f", st.t,
                            gs.engine->params().t_end);
                        ImGui::Text("|x| = %.3f", err);
                        ImGui::Text("|u| = %.2f / %.1f",
                            (float)u_vec.norm(), gs.ctrl_umax);
                        ImGui::Text("tr(G) = %.2f", trace_g);
                        auto evals = st.G.eigenvalues();
                        ImGui::Text("lam = %.2f, %.2f",
                            evals(0), evals(1));

                        if (st.t > 5.0) {
                            auto m = gs.engine->recorder()
                                .compute_metrics(5.0, gs.ctrl_umax);
                            ImGui::Separator();
                            ImGui::Text("Efficiency %.4f", m.efficiency);
                            ImGui::Text("Aniso %.3f", m.mean_aniso);
                        }
                    } else {
                        ImGui::TextColored({0.5f,0.5f,0.5f,1},
                            "No data");
                    }
                }
                ImGui::EndChild();

                ImGui::EndTabItem();
            }

            // ============================================================
            //  TAB: CONTROL DETAIL
            // ============================================================
            if (ImGui::BeginTabItem("Control")) {
                float w_half = (ImGui::GetContentRegionAvail().x - style.ItemSpacing.x) * 0.5f;
                float h_full = ImGui::GetContentRegionAvail().y;

                // Left: u0(t), u1(t) stacked
                ImGui::BeginGroup();
                if (gs.has_data && ImPlot::BeginPlot("u0(t)##u0", ImVec2(w_half, h_full * 0.32f))) {
                    ImPlot::SetupAxes("", "u0");
                    ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.3f,0.8f,1.0f,1));
                    plot_scroll("u0", gs.t_buf, gs.u_buf[0]);
                    ImPlot::PopStyleColor();
                    ImPlot::EndPlot();
                }
                if (gs.has_data && ImPlot::BeginPlot("u1(t)##u1", ImVec2(w_half, h_full * 0.32f))) {
                    ImPlot::SetupAxes("", "u1");
                    ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1.0f,0.5f,0.2f,1));
                    plot_scroll("u1", gs.t_buf, gs.u_buf[1]);
                    ImPlot::PopStyleColor();
                    ImPlot::EndPlot();
                }
                if (gs.has_data && ImPlot::BeginPlot("Effort |u|^2##eff",
                        ImVec2(w_half, h_full * 0.32f))) {
                    ImPlot::SetupAxes("t", "|u|^2");
                    ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1,0.3f,0.3f,1));
                    ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.25f);
                    plot_scroll("|u|^2", gs.t_buf, gs.effort_buf);
                    ImPlot::PopStyleVar();
                    ImPlot::PopStyleColor();
                    ImPlot::EndPlot();
                }
                ImGui::EndGroup();

                ImGui::SameLine();

                // Right: 2D control vector plot
                ImGui::BeginGroup();
                if (gs.has_data && ImPlot::BeginPlot("Control Vector##cvec",
                        ImVec2(w_half, h_full * 0.65f), ImPlotFlags_Equal)) {
                    ImPlot::SetupAxes("u0", "u1");
                    ImPlot::SetupAxesLimits(-3.5, 3.5, -3.5, 3.5, ImPlotCond_Always);

                    // u_max circle
                    double uc_x[65], uc_y[65];
                    for (int i = 0; i <= 64; ++i) {
                        double th = 2.0 * M_PI * i / 64;
                        uc_x[i] = 3.0 * std::cos(th);
                        uc_y[i] = 3.0 * std::sin(th);
                    }
                    ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.5f,0.5f,0.5f,0.3f));
                    ImPlot::PlotLine("|u|=u_max", uc_x, uc_y, 65);
                    ImPlot::PopStyleColor();

                    // Control trail (recent u vectors as scatter)
                    if (gs.u_buf[0].size > 1) {
                        int trail_n = std::min(gs.u_buf[0].size, 200);
                        std::vector<double> tx(trail_n), ty(trail_n);
                        int start = gs.u_buf[0].size - trail_n;
                        for (int i = 0; i < trail_n; ++i) {
                            tx[i] = gs.u_buf[0][start + i];
                            ty[i] = gs.u_buf[1][start + i];
                        }
                        ImPlot::PushStyleColor(ImPlotCol_MarkerFill, ImVec4(0.4f,0.6f,0.9f,0.15f));
                        ImPlot::PushStyleColor(ImPlotCol_MarkerOutline, ImVec4(0.4f,0.6f,0.9f,0.05f));
                        ImPlot::PushStyleVar(ImPlotStyleVar_MarkerSize, 2.0f);
                        ImPlot::PlotScatter("history", tx.data(), ty.data(), trail_n);
                        ImPlot::PopStyleVar();
                        ImPlot::PopStyleColor(2);
                    }

                    // Current control arrow from origin
                    auto& u = gs.engine->last_u();
                    double u_norm = u.norm();
                    float r = static_cast<float>(std::min(u_norm / 3.0, 1.0));
                    ImVec4 acol(r, 1.0f - r, 0.2f, 1.0f);
                    plot_arrow("##u_arrow", 0, 0, u(0), u(1), acol, 3.5f);

                    // Current u dot
                    double cu[] = {u(0)}, cv[] = {u(1)};
                    ImPlot::PushStyleColor(ImPlotCol_MarkerFill, acol);
                    ImPlot::PushStyleVar(ImPlotStyleVar_MarkerSize, 6.0f);
                    ImPlot::PlotScatter("u now", cu, cv, 1);
                    ImPlot::PopStyleVar();
                    ImPlot::PopStyleColor();

                    ImPlot::EndPlot();
                }

                // Legend / explanation
                ImGui::TextWrapped(
                    "Arrow: current control vector u.\n"
                    "Green = low effort, Red = near u_max.\n"
                    "Gray circle = saturation limit |u|=3.");
                ImGui::EndGroup();

                ImGui::EndTabItem();
            }

            // ============================================================
            //  TAB: TENSOR DETAIL
            // ============================================================
            if (ImGui::BeginTabItem("Tensor G")) {
                if (gs.has_data && ImPlot::BeginPlot("Eigenvalues##eig", ImVec2(-1, 250))) {
                    ImPlot::SetupAxes("t", "lambda");
                    plot_scroll("lam0", gs.t_buf, gs.eig_buf[0]);
                    plot_scroll("lam1", gs.t_buf, gs.eig_buf[1]);
                    ImPlot::EndPlot();
                }
                if (gs.has_data && ImPlot::BeginPlot("tr(G) & |Q|##trq", ImVec2(-1, 250))) {
                    ImPlot::SetupAxes("t", "");
                    plot_scroll("tr(G)", gs.t_buf, gs.trG_buf);
                    plot_scroll("|Q|",   gs.t_buf, gs.Qnorm_buf);
                    ImPlot::EndPlot();
                }
                ImGui::EndTabItem();
#endif // removed tabs

            // ============================================================
            //  TAB: BENCHMARK — compare controllers
            // ============================================================
            if (ImGui::BeginTabItem("Benchmark")) {
                if (ImGui::Button("Run Benchmark", {160, 0})) {
                    try {
                        YAML::Node bcfg = YAML::LoadFile(gs.config_path);
                        if (bcfg["controllers"] && bcfg["controllers"].IsSequence()) {
                            gs.bench_results = aniso::run_benchmark<DIM>(bcfg);
                            gs.bench_ran = true;
                        }
                    } catch (...) {}
                }
                ImGui::SameLine();
                if (ImGui::Button("Run Pareto Grid", {160, 0})) {
                    try {
                        YAML::Node bcfg = YAML::LoadFile(gs.config_path);
                        gs.pareto_grid = aniso::run_pareto_grid<DIM>(bcfg);
                        gs.pareto_ran = true;
                        if (!gs.bench_ran && bcfg["controllers"]) {
                            gs.bench_results = aniso::run_benchmark<DIM>(bcfg);
                            gs.bench_ran = true;
                        }
                    } catch (...) {}
                }
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("~200 configs: P, A, Pulsed across gain/u_max grid");
                ImGui::SameLine();
                ImGui::TextDisabled("(loads controllers from current YAML)");

                if (gs.bench_ran && !gs.bench_results.empty()) {
                    int n = static_cast<int>(gs.bench_results.size());
                    float w_half = (ImGui::GetContentRegionAvail().x - style.ItemSpacing.x) * 0.5f;

                    // Prepare bar chart data
                    std::vector<double> err_vals(n), eff_vals(n), aniso_vals(n), effic_vals(n);
                    std::vector<const char*> labels(n);
                    for (int i = 0; i < n; ++i) {
                        labels[i] = gs.bench_results[i].name.c_str();
                        err_vals[i]   = gs.bench_results[i].metrics.mean_error;
                        eff_vals[i]   = gs.bench_results[i].metrics.mean_effort;
                        aniso_vals[i] = gs.bench_results[i].metrics.final_aniso;
                        effic_vals[i] = gs.bench_results[i].metrics.efficiency;
                    }
                    std::vector<double> positions(n);
                    for (int i = 0; i < n; ++i) positions[i] = static_cast<double>(i);

                    // --- Bar chart: Tracking Error ---
                    if (ImPlot::BeginPlot("Tracking Error by Controller##bar_err",
                            ImVec2(w_half, 280))) {
                        ImPlot::SetupAxes("", "Mean |x|");
                        ImPlot::SetupAxisTicks(ImAxis_X1, positions.data(), n, labels.data());
                        ImPlot::PushStyleColor(ImPlotCol_Fill, ImVec4(0.2f,0.6f,1.0f,0.8f));
                        ImPlot::PlotBars("Error", positions.data(), err_vals.data(), n, 0.6);
                        ImPlot::PopStyleColor();
                        ImPlot::EndPlot();
                    }

                    ImGui::SameLine();

                    // --- Bar chart: Control Effort ---
                    if (ImPlot::BeginPlot("Control Effort by Controller##bar_eff",
                            ImVec2(w_half, 280))) {
                        ImPlot::SetupAxes("", "Mean |u|^2");
                        ImPlot::SetupAxisTicks(ImAxis_X1, positions.data(), n, labels.data());
                        ImPlot::PushStyleColor(ImPlotCol_Fill, ImVec4(1.0f,0.4f,0.2f,0.8f));
                        ImPlot::PlotBars("Effort", positions.data(), eff_vals.data(), n, 0.6);
                        ImPlot::PopStyleColor();
                        ImPlot::EndPlot();
                    }

                    // --- Pareto: Effort vs Error with frontier ---
                    if (ImPlot::BeginPlot("Pareto: Effort vs Error (lower-left = better)##pareto",
                            ImVec2(w_half, 300))) {
                        ImPlot::SetupAxes("Mean Effort |u|^2", "Mean Error |x|");

                        // Compute Pareto frontier (sort by effort, keep non-dominated)
                        struct ParetoPoint { double x, y; int idx; };
                        std::vector<ParetoPoint> pts(n);
                        for (int i = 0; i < n; ++i)
                            pts[i] = {eff_vals[i], err_vals[i], i};
                        std::sort(pts.begin(), pts.end(),
                            [](auto& a, auto& b) { return a.x < b.x; });

                        std::vector<double> front_x, front_y;
                        double best_y = 1e30;
                        for (auto& p : pts) {
                            if (p.y <= best_y) {
                                front_x.push_back(p.x);
                                front_y.push_back(p.y);
                                best_y = p.y;
                            }
                        }

                        // Draw frontier as step line
                        if (front_x.size() > 1) {
                            std::vector<double> step_x, step_y;
                            for (size_t i = 0; i < front_x.size(); ++i) {
                                if (i > 0) {
                                    step_x.push_back(front_x[i]);
                                    step_y.push_back(front_y[i-1]);
                                }
                                step_x.push_back(front_x[i]);
                                step_y.push_back(front_y[i]);
                            }
                            ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.3f,0.8f,0.3f,0.6f));
                            ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 2.0f);
                            ImPlot::PlotLine("Pareto front", step_x.data(), step_y.data(),
                                static_cast<int>(step_x.size()));
                            ImPlot::PopStyleVar();
                            ImPlot::PopStyleColor();
                        }

                        ImVec4 colors[] = {
                            {0.2f,0.7f,1.0f,1}, {1.0f,0.4f,0.2f,1}, {0.2f,0.9f,0.3f,1},
                            {1.0f,0.8f,0.1f,1}, {0.8f,0.3f,0.9f,1}, {0.9f,0.6f,0.4f,1},
                            {0.5f,0.5f,1.0f,1}, {1.0f,0.6f,0.8f,1}
                        };
                        for (int i = 0; i < n; ++i) {
                            ImPlot::PushStyleColor(ImPlotCol_MarkerFill, colors[i % 8]);
                            ImPlot::PushStyleVar(ImPlotStyleVar_MarkerSize, 8.0f);
                            ImPlot::PlotScatter(labels[i], &eff_vals[i], &err_vals[i], 1);
                            ImPlot::PopStyleVar();
                            ImPlot::PopStyleColor();
                        }
                        ImPlot::EndPlot();
                    }

                    ImGui::SameLine();

                    // --- Bar chart: Efficiency (composite) ---
                    if (ImPlot::BeginPlot("Efficiency (error x effort, lower = better)##bar_effic",
                            ImVec2(w_half, 300))) {
                        ImPlot::SetupAxes("", "Efficiency");
                        ImPlot::SetupAxisTicks(ImAxis_X1, positions.data(), n, labels.data());

                        // Color each bar: best = green, worst = red
                        double min_e = *std::min_element(effic_vals.begin(), effic_vals.end());
                        double max_e = *std::max_element(effic_vals.begin(), effic_vals.end());
                        double range = std::max(max_e - min_e, 1e-6);
                        for (int i = 0; i < n; ++i) {
                            float t = static_cast<float>((effic_vals[i] - min_e) / range);
                            ImVec4 col(0.2f + 0.8f*t, 0.8f - 0.6f*t, 0.2f, 0.85f);
                            ImPlot::PushStyleColor(ImPlotCol_Fill, col);
                            ImPlot::PlotBars(labels[i], &positions[i], &effic_vals[i], 1, 0.6);
                            ImPlot::PopStyleColor();
                        }
                        ImPlot::EndPlot();
                    }

                    // --- Dense Pareto grid (if computed) ---
                    if (gs.pareto_ran && !gs.pareto_grid.empty()) {
                        int ng = static_cast<int>(gs.pareto_grid.size());
                        std::vector<double> gx(ng), gy(ng);
                        for (int i = 0; i < ng; ++i) {
                            gx[i] = gs.pareto_grid[i].metrics.mean_effort;
                            gy[i] = gs.pareto_grid[i].metrics.mean_error;
                        }

                        // Compute Pareto frontier over ALL points (grid + named)
                        struct PP { double x, y; };
                        std::vector<PP> all_pts;
                        for (int i = 0; i < ng; ++i)
                            all_pts.push_back({gx[i], gy[i]});
                        for (int i = 0; i < n; ++i)
                            all_pts.push_back({eff_vals[i], err_vals[i]});
                        std::sort(all_pts.begin(), all_pts.end(),
                            [](auto& a, auto& b){ return a.x < b.x; });
                        std::vector<double> fx, fy;
                        double by = 1e30;
                        for (auto& p : all_pts) {
                            if (p.y <= by) { fx.push_back(p.x); fy.push_back(p.y); by = p.y; }
                        }

                        ImGui::Text("Pareto grid: %d configurations", ng);
                        if (ImPlot::BeginPlot("Dense Pareto Front##pareto_dense",
                                ImVec2(-1, 380))) {
                            ImPlot::SetupAxes("Mean Effort |u|^2", "Mean Error |x|");

                            // Grid points (small, semi-transparent)
                            ImPlot::PushStyleColor(ImPlotCol_MarkerFill,
                                ImVec4(0.35f, 0.45f, 0.65f, 0.25f));
                            ImPlot::PushStyleColor(ImPlotCol_MarkerOutline,
                                ImVec4(0.4f, 0.5f, 0.7f, 0.15f));
                            ImPlot::PushStyleVar(ImPlotStyleVar_MarkerSize, 3.0f);
                            ImPlot::PlotScatter("grid", gx.data(), gy.data(), ng);
                            ImPlot::PopStyleVar();
                            ImPlot::PopStyleColor(2);

                            // Frontier line
                            if (fx.size() > 1) {
                                std::vector<double> sx, sy;
                                for (size_t i = 0; i < fx.size(); ++i) {
                                    if (i > 0) { sx.push_back(fx[i]); sy.push_back(fy[i-1]); }
                                    sx.push_back(fx[i]); sy.push_back(fy[i]);
                                }
                                ImPlot::PushStyleColor(ImPlotCol_Line,
                                    ImVec4(0.2f, 0.9f, 0.3f, 0.7f));
                                ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 2.5f);
                                ImPlot::PlotLine("Pareto front", sx.data(), sy.data(),
                                    static_cast<int>(sx.size()));
                                ImPlot::PopStyleVar();
                                ImPlot::PopStyleColor();
                            }

                            // Named controllers (large, bright)
                            ImVec4 colors[] = {
                                {0.2f,0.7f,1.0f,1}, {1.0f,0.4f,0.2f,1},
                                {0.2f,0.9f,0.3f,1}, {1.0f,0.8f,0.1f,1},
                                {0.8f,0.3f,0.9f,1}, {0.9f,0.6f,0.4f,1},
                                {0.5f,0.5f,1.0f,1}, {1.0f,0.6f,0.8f,1}
                            };
                            for (int i = 0; i < n; ++i) {
                                ImPlot::PushStyleColor(ImPlotCol_MarkerFill, colors[i%8]);
                                ImPlot::PushStyleVar(ImPlotStyleVar_MarkerSize, 8.0f);
                                ImPlot::PlotScatter(labels[i],
                                    &eff_vals[i], &err_vals[i], 1);
                                ImPlot::PopStyleVar();
                                ImPlot::PopStyleColor();
                            }
                            ImPlot::EndPlot();
                        }
                    }

                    // --- Metrics table ---
                    ImGui::Separator();
                    if (ImGui::BeginTable("bench_table", 10,
                            ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                            ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingFixedFit)) {

                        ImGui::TableSetupColumn("Controller", 0, 140);
                        ImGui::TableSetupColumn("Error",      0, 70);
                        ImGui::TableSetupColumn("PeakErr",    0, 70);
                        ImGui::TableSetupColumn("Effort",     0, 70);
                        ImGui::TableSetupColumn("Sat%",       0, 55);
                        ImGui::TableSetupColumn("Aniso",      0, 65);
                        ImGui::TableSetupColumn("FinalA",     0, 65);
                        ImGui::TableSetupColumn("Effic.",     0, 70);
                        ImGui::TableSetupColumn("Status",     0, 45);
                        ImGui::TableSetupColumn("ms",         0, 55);
                        ImGui::TableHeadersRow();

                        double best_eff = 1e30;
                        for (auto& r : gs.bench_results)
                            if (!r.metrics.breakdown)
                                best_eff = std::min(best_eff, r.metrics.efficiency);

                        for (auto& r : gs.bench_results) {
                            auto& m = r.metrics;
                            bool is_best = (!m.breakdown && m.efficiency <= best_eff * 1.01);
                            ImGui::TableNextRow();
                            if (m.breakdown)
                                ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg1,
                                    ImGui::GetColorU32(ImVec4(0.5f, 0.1f, 0.1f, 0.4f)));
                            else if (is_best)
                                ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg1,
                                    ImGui::GetColorU32(ImVec4(0.1f, 0.4f, 0.1f, 0.4f)));
                            ImGui::TableNextColumn(); ImGui::Text("%s", r.name.c_str());
                            ImGui::TableNextColumn(); ImGui::Text("%.4f", m.mean_error);
                            ImGui::TableNextColumn(); ImGui::Text("%.4f", m.peak_error);
                            ImGui::TableNextColumn(); ImGui::Text("%.4f", m.mean_effort);
                            ImGui::TableNextColumn(); ImGui::Text("%.0f%%", m.saturation_frac * 100);
                            ImGui::TableNextColumn(); ImGui::Text("%.2f", m.mean_aniso);
                            ImGui::TableNextColumn(); ImGui::Text("%.2f", m.final_aniso);
                            ImGui::TableNextColumn(); ImGui::Text("%.4f", m.efficiency);
                            ImGui::TableNextColumn();
                            if (m.breakdown)
                                ImGui::TextColored({1,0.3f,0.3f,1}, "FAIL");
                            else
                                ImGui::TextColored({0.3f,1,0.3f,1}, "OK");
                            ImGui::TableNextColumn(); ImGui::Text("%.1f", r.wall_ms);
                        }
                        ImGui::EndTable();
                    }
                } else if (gs.bench_ran) {
                    ImGui::TextColored({1,0.5f,0.3f,1},
                        "No 'controllers' section found in YAML. Use benchmark config.");
                }
                ImGui::EndTabItem();
            }

            // ============================================================
            //  TAB: CHAIN — 1D spatial self-organization
            // ============================================================
            {
                ImGuiTabItemFlags chain_flags = 0;
                if (gs.chain_just_loaded) {
                    chain_flags = ImGuiTabItemFlags_SetSelected;
                    gs.chain_just_loaded = false;
                }
                if (gs.chain_loaded && ImGui::BeginTabItem("Chain", nullptr, chain_flags)) {
                ImVec2 avail = ImGui::GetContentRegionAvail();
                int N = gs.chain->N();

                // ---- Top control bar ----
                {
                    float bw = 70;
                    if (ImGui::Button(gs.chain_running ? "Pause##ch" : " Run ##ch", {bw, 0}))
                        gs.chain_running = !gs.chain_running;
                    ImGui::SameLine();
                    if (ImGui::Button("Reset##ch", {bw, 0})) {
                        gs.chain->reset(); gs.chain_running = false; gs.kymo.clear();
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("Step##ch", {bw, 0})) {
                        gs.chain->step();
                        if (gs.kymo.N > 0) {
                            std::vector<float> row(N);
                            for (int j = 0; j < N; ++j) row[j] = (float)gs.chain->health(j);
                            gs.kymo.push_row(row.data());
                        }
                    }
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(120);
                    ImGui::SliderInt("Steps/f##ch", &gs.chain_steps_per_frame, 1, 200);
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("Simulation steps per rendered frame");

                    ImGui::SameLine();
                    char ov[64];
                    std::snprintf(ov, 64, "t = %.1f / %.0f",
                        gs.chain->t(), gs.chain->params().t_end);
                    float progress = (float)(gs.chain->t() / gs.chain->params().t_end);
                    ImGui::ProgressBar(progress, {180, 0}, ov);
                    if (gs.chain->done()) { ImGui::SameLine(); ImGui::TextColored({0.3f,1,0.3f,1}, "DONE"); }
                }

                // ---- Parameter sliders (two rows) ----
                {
                    bool drive_changed = false;
                    ImGui::PushItemWidth(105);

                    // Row 1: physics
                    if (ImGui::SliderFloat("D_G##ch", &gs.ch_D_G, 0.0f, 0.3f, "%.3f"))
                        gs.chain->params().D_G = gs.ch_D_G;
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Tensor diffusion between neighbors.\nHigher = smoother profile, wider barriers.\nLower = sharper barriers.");
                    ImGui::SameLine();
                    if (ImGui::SliderFloat("D_x##ch", &gs.ch_D_x, 0.0f, 1.0f, "%.2f"))
                        gs.chain->params().D_x = gs.ch_D_x;
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("State coupling between neighbors");
                    ImGui::SameLine();
                    if (ImGui::SliderFloat("tau##ch", &gs.ch_tau, 0.5f, 30.0f, "%.1f"))
                        gs.chain->params().tau = gs.ch_tau;
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Relaxation time: how fast G returns to Identity (order restores)");
                    ImGui::SameLine();
                    if (ImGui::SliderFloat("alpha##ch", &gs.ch_alpha, 0.0f, 2.0f, "%.2f"))
                        gs.chain->coup().set_alpha(gs.ch_alpha);
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Coupling strength: how much control degrades tensor");
                    ImGui::SameLine();
                    if (ImGui::SliderFloat("mu##ch", &gs.ch_mu, 0.0f, 4.0f, "%.1f"))
                        gs.chain->inter().set_mu(gs.ch_mu);
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Landau self-interaction: anisotropy amplification above threshold");

                    // Row 2: bistability + drive profile + controller
                    if (ImGui::SliderFloat("trap##ch", &gs.ch_trap, 0.0f, 30.0f, "%.1f"))
                        gs.chain->params().trap = gs.ch_trap;
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Hysteresis strength: slows recovery above s_crit.\n0 = linear, 10+ = strong phase memory");
                    ImGui::SameLine();
                    if (ImGui::SliderFloat("s_crit##ch", &gs.ch_s_crit, 1.0f, 4.0f, "%.2f"))
                        gs.chain->params().s_crit = gs.ch_s_crit;
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Degradation threshold: trace(G)/Dim.\nBelow = normal recovery. Above = slow recovery (hysteresis).");
                    ImGui::SameLine();
                    if (ImGui::SliderFloat("Peak##ch", &gs.ch_drive_peak, 0.5f, 10.0f, "%.1f")) {
                        gs.chain->params().drive_peak = gs.ch_drive_peak; drive_changed = true;
                    }
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Drive intensity at center: higher = more degradation (overheating)");
                    ImGui::SameLine();
                    if (ImGui::SliderFloat("Width##ch", &gs.ch_drive_width, 0.02f, 0.8f, "%.2f")) {
                        gs.chain->params().drive_width = gs.ch_drive_width; drive_changed = true;
                    }
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Drive profile width: how wide the heated zone is");
                    ImGui::SameLine();
                    if (ImGui::SliderFloat("Center##ch", &gs.ch_drive_center, 0.0f, 1.0f, "%.2f")) {
                        gs.chain->params().drive_center = gs.ch_drive_center; drive_changed = true;
                    }
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Drive center position along the chain");
                    ImGui::SameLine();
                    if (ImGui::SliderFloat("Gain##ch", &gs.ch_gain, 0.1f, 10.0f, "%.1f"))
                        gs.chain->ctrl().set_gain(gs.ch_gain);
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Controller gain: higher = more control effort = more tensor degradation");
                    ImGui::SameLine();
                    if (ImGui::SliderFloat("u_max##ch", &gs.ch_umax, 0.5f, 10.0f, "%.1f"))
                        gs.chain->ctrl().set_umax(gs.ch_umax);
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Maximum control signal: saturation limit");

                    // Controller type selector
                    ImGui::SameLine();
                    {
                        const char* ctrl_types[] = {"proportional", "aniso_aware", "pulsed", "event_triggered"};
                        ImGui::SetNextItemWidth(140);
                        if (ImGui::Combo("Ctrl##ch_ctrl", &gs.ch_ctrl_idx, ctrl_types, 4)) {
                            YAML::Node cn;
                            cn["type"]  = std::string(ctrl_types[gs.ch_ctrl_idx]);
                            cn["gain"]  = (double)gs.ch_gain;
                            cn["u_max"] = (double)gs.ch_umax;
                            if (gs.ch_ctrl_idx == 2) { cn["period"] = 4.0; cn["duty"] = 0.5; }
                            if (gs.ch_ctrl_idx == 3) {
                                cn["trigger"] = 0.5; cn["hysteresis"] = 0.6;
                                cn["anticipation"] = 5.0;
                            }
                            gs.chain->swap_controller(
                                aniso::detail::make_controller<DIM>(cn));
                        }
                    }

                    ImGui::PopItemWidth();
                    if (drive_changed) gs.chain->rebuild_drive_profile();
                }

                ImGui::Separator();
                float remaining_h = ImGui::GetContentRegionAvail().y;
                float kymo_h   = remaining_h * 0.48f;
                float strip_h  = 22.0f;
                float plots_h  = remaining_h - kymo_h - strip_h - 30;

                // ---- KYMOGRAPH: space-time diagram ----
                {
                    ImVec2 kp = ImGui::GetCursorScreenPos();
                    float kw = ImGui::GetContentRegionAvail().x;
                    ImGui::InvisibleButton("##kymo", {kw, kymo_h});
                    ImDrawList* dl = ImGui::GetWindowDrawList();

                    dl->AddRectFilled(kp, {kp.x + kw, kp.y + kymo_h}, IM_COL32(10, 12, 20, 255));

                    int rows = gs.kymo.count;
                    if (rows > 0) {
                        float cell_w = kw / N;
                        float cell_h = kymo_h / std::max(rows, 1);
                        cell_h = std::min(cell_h, kymo_h / 50.0f);
                        int visible = std::min(rows, (int)(kymo_h / std::max(cell_h, 0.5f)));
                        cell_h = kymo_h / visible;
                        int start_row = rows - visible;

                        for (int r = 0; r < visible; ++r) {
                            float y0 = kp.y + kymo_h - (r + 1) * cell_h;
                            float y1 = y0 + cell_h;
                            int data_row = start_row + r;
                            for (int c = 0; c < N; ++c) {
                                float v = gs.kymo.get(data_row, c);
                                dl->AddRectFilled(
                                    {kp.x + c * cell_w, y0},
                                    {kp.x + (c+1) * cell_w + 1, y1 + 1},
                                    health_color(v));
                            }
                        }

                        // Time axis labels
                        dl->AddText({kp.x + 4, kp.y + 2},
                            IM_COL32(200, 200, 220, 200), "now");
                        dl->AddText({kp.x + 4, kp.y + kymo_h - 16},
                            IM_COL32(200, 200, 220, 120), "past");
                    } else {
                        const char* msg = "Press Run to start \xe2\x80\x94 space-time diagram will appear here";
                        auto msz = ImGui::CalcTextSize(msg);
                        dl->AddText({kp.x + kw*0.5f - msz.x*0.5f, kp.y + kymo_h*0.5f - msz.y*0.5f},
                            IM_COL32(120, 120, 140, 180), msg);
                    }

                    // Axis labels
                    dl->AddText({kp.x + kw*0.5f - 50, kp.y + kymo_h - 16},
                        IM_COL32(140, 140, 160, 150), "node position \xe2\x86\x92");
                    float arrow_x = kp.x + kw - 24;
                    dl->AddText({arrow_x, kp.y + kymo_h*0.5f - 6},
                        IM_COL32(140, 140, 160, 150), "\xe2\x86\x91 t");

                    // Color legend
                    float lx = kp.x + kw - 200, ly = kp.y + 2;
                    for (int i = 0; i < 100; ++i) {
                        float v = (float)i / 99.0f;
                        dl->AddRectFilled({lx + i*1.2f, ly}, {lx + (i+1)*1.2f, ly + 10}, health_color(v));
                    }
                    dl->AddText({lx - 30, ly - 1}, IM_COL32(220, 60, 40, 200), "hot");
                    dl->AddText({lx + 124, ly - 1}, IM_COL32(30, 100, 220, 200), "cool");
                }

                // ---- Current state strip + drive overlay ----
                {
                    ImVec2 sp = ImGui::GetCursorScreenPos();
                    float sw = ImGui::GetContentRegionAvail().x;
                    ImGui::InvisibleButton("##strip", {sw, strip_h});
                    ImDrawList* dl = ImGui::GetWindowDrawList();
                    float cell_w = sw / N;

                    for (int i = 0; i < N; ++i) {
                        dl->AddRectFilled(
                            {sp.x + i*cell_w, sp.y},
                            {sp.x + (i+1)*cell_w, sp.y + strip_h},
                            health_color((float)gs.chain->health(i)));
                    }
                    // Drive profile as white line overlay
                    for (int i = 1; i < N; ++i) {
                        float d0 = std::clamp(((float)gs.chain->drive(i-1) - 1.0f) / std::max(gs.ch_drive_peak - 1.0f, 0.01f), 0.0f, 1.0f);
                        float d1 = std::clamp(((float)gs.chain->drive(i) - 1.0f) / std::max(gs.ch_drive_peak - 1.0f, 0.01f), 0.0f, 1.0f);
                        dl->AddLine(
                            {sp.x + (i-0.5f)*cell_w, sp.y + strip_h*(1.0f - d0*0.9f)},
                            {sp.x + (i+0.5f)*cell_w, sp.y + strip_h*(1.0f - d1*0.9f)},
                            IM_COL32(255, 255, 255, 120), 1.5f);
                    }
                    dl->AddText({sp.x + 4, sp.y + 3},
                        IM_COL32(255, 255, 255, 180), "Current state");
                    dl->AddText({sp.x + sw - 170, sp.y + 3},
                        IM_COL32(255, 255, 255, 120), "white line = drive");
                }

                // ---- Profile plots (bottom) ----
                if (plots_h > 80) {
                    std::vector<double> pos(N), trG(N), aniso(N), health_v(N), drive(N);
                    for (int i = 0; i < N; ++i) {
                        pos[i]     = (double)i / (N - 1);
                        trG[i]     = gs.chain->G(i).trace();
                        aniso[i]   = gs.chain->anisotropy(i);
                        health_v[i]= gs.chain->health(i);
                        drive[i]   = gs.chain->drive(i);
                    }
                    float pw = (ImGui::GetContentRegionAvail().x - style.ItemSpacing.x) * 0.5f;
                    float ph = plots_h * 0.5f;

                    if (ImPlot::BeginPlot("trace(G) & Drive##ch_trG", ImVec2(pw, ph))) {
                        ImPlot::SetupAxes("position", "");
                        ImPlot::SetupAxesLimits(0, 1, 0, gs.ch_drive_peak * 2 + 2, ImPlotCond_Once);
                        ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1.0f, 0.4f, 0.3f, 1));
                        ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 2.5f);
                        ImPlot::PlotLine("tr(G)", pos.data(), trG.data(), N);
                        ImPlot::PopStyleVar(); ImPlot::PopStyleColor();
                        ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1.0f, 0.85f, 0.3f, 0.6f));
                        ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 1.5f);
                        ImPlot::PlotLine("drive", pos.data(), drive.data(), N);
                        ImPlot::PopStyleVar(); ImPlot::PopStyleColor();
                        double id_x[] = {0, 1}, id_y[] = {(double)DIM, (double)DIM};
                        ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.5f, 0.5f, 0.5f, 0.4f));
                        ImPlot::PlotLine("G=I", id_x, id_y, 2);
                        ImPlot::PopStyleColor();
                        ImPlot::EndPlot();
                    }
                    ImGui::SameLine();
                    if (ImPlot::BeginPlot("Anisotropy##ch_aniso", ImVec2(pw, ph))) {
                        ImPlot::SetupAxes("position", "anisotropy");
                        ImPlot::SetupAxesLimits(0, 1, 0, 4, ImPlotCond_Once);
                        ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.7f, 0.3f, 0.9f, 1));
                        ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 2.5f);
                        ImPlot::PlotLine("aniso", pos.data(), aniso.data(), N);
                        ImPlot::PopStyleVar(); ImPlot::PopStyleColor();
                        ImPlot::EndPlot();
                    }
                    if (ImPlot::BeginPlot("Health##ch_health", ImVec2(pw, ph))) {
                        ImPlot::SetupAxes("position", "health [0=degraded, 1=pristine]");
                        ImPlot::SetupAxesLimits(0, 1, -0.05, 1.05, ImPlotCond_Once);
                        ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.2f, 0.9f, 0.3f, 1));
                        ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 2.5f);
                        ImPlot::PlotLine("health", pos.data(), health_v.data(), N);
                        ImPlot::PopStyleVar(); ImPlot::PopStyleColor();
                        ImPlot::EndPlot();
                    }
                    ImGui::SameLine();
                    // Summary metrics
                    ImGui::BeginGroup();
                    {
                        double avg_h = 0, min_h = 1, max_a = 0, avg_tr = 0;
                        for (int i = 0; i < N; ++i) {
                            avg_h  += health_v[i];
                            min_h   = std::min(min_h, health_v[i]);
                            max_a   = std::max(max_a, aniso[i]);
                            avg_tr += trG[i];
                        }
                        avg_h /= N; avg_tr /= N;
                        int degraded = 0;
                        for (int i = 0; i < N; ++i)
                            if (health_v[i] < 0.5) ++degraded;

                        ImGui::TextColored({0.7f,0.75f,0.8f,1}, "Chain Summary");
                        ImGui::Separator();
                        ImGui::Text("Avg health:  %.2f", avg_h);
                        ImGui::Text("Min health:  %.2f", min_h);
                        ImGui::Text("Degraded:    %d / %d nodes", degraded, N);
                        ImGui::Text("Max aniso:   %.2f", max_a);
                        ImGui::Text("Avg tr(G):   %.2f", avg_tr);
                        ImGui::Spacing();
                        if (degraded > 0 && degraded < N)
                            ImGui::TextColored({0.3f,1.0f,0.4f,1}, "BARRIER FORMED");
                        else if (degraded == 0)
                            ImGui::TextColored({0.3f,0.7f,1.0f,1}, "ALL PRISTINE");
                        else
                            ImGui::TextColored({1.0f,0.3f,0.2f,1}, "FULLY DEGRADED");
                    }
                    ImGui::EndGroup();
                }

                ImGui::EndTabItem();
            }
            }

            // ============================================================
            //  TAB: GRID 2D — spatial self-organization on NxM lattice
            // ============================================================
            {
                ImGuiTabItemFlags grid_flags = 0;
                if (gs.grid_just_loaded) {
                    grid_flags = ImGuiTabItemFlags_SetSelected;
                    gs.grid_just_loaded = false;
                }
                if (gs.grid_loaded && ImGui::BeginTabItem("Grid 2D", nullptr, grid_flags)) {
                    ImVec2 avail = ImGui::GetContentRegionAvail();
                    int Nx = gs.grid->Nx(), Ny = gs.grid->Ny();

                    // ---- Top control bar ----
                    {
                        float bw = 70;
                        if (ImGui::Button(gs.grid_running ? "Pause##gr" : " Run ##gr", {bw, 0}))
                            gs.grid_running = !gs.grid_running;
                        ImGui::SameLine();
                        if (ImGui::Button("Reset##gr", {bw, 0})) {
                            gs.grid->reset(); gs.grid_running = false;
                            std::memset(gs.gr_hist_x, 0, sizeof(gs.gr_hist_x));
                            std::memset(gs.gr_hist_E, 0, sizeof(gs.gr_hist_E));
                            std::memset(gs.gr_hist_tr, 0, sizeof(gs.gr_hist_tr));
                            gs.gr_hist_pos = 0;
                        }
                        ImGui::SameLine();
                        if (ImGui::Button("Step##gr", {bw, 0})) gs.grid->step();
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(80);
                        ImGui::SliderInt("##spd", &gs.grid_steps_per_frame, 1, 500, "%d st/f");
                        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Simulation steps per rendered frame.\nHigher = faster simulation.");
                        ImGui::SameLine();
                        ImGui::TextDisabled("t=%.0f", gs.grid->t());
                        ImGui::SameLine();
                        {
                            const char* modes[] = {"Health","Energy","|x|","|u|"};
                            ImGui::SetNextItemWidth(90);
                            ImGui::Combo("Map##gm", &gs.gr_map_mode, modes, 4);
                        }
                    }

                    // ---- Parameter sliders ----
                    {
                        bool drive_changed = false;
                        ImGui::PushItemWidth(100);

                        // Row 1: Controller
                        if (ImGui::SliderFloat("Gain##gr", &gs.gr_gain, 0.1f, 50.0f, "%.2f",
                                ImGuiSliderFlags_Logarithmic))
                            gs.grid->ctrl().set_gain(gs.gr_gain);
                        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Controller gain (log scale).\nControl injects energy into the medium.");
                        ImGui::SameLine();
                        if (ImGui::SliderFloat("u_max##gr", &gs.gr_umax, 0.5f, 50.0f, "%.1f"))
                            gs.grid->ctrl().set_umax(gs.gr_umax);
                        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Control saturation");
                        ImGui::SameLine();
                        {
                            const char* ctrl_types[] = {"proportional", "aniso_aware", "pulsed", "event_triggered"};
                            ImGui::SetNextItemWidth(140);
                            if (ImGui::Combo("Ctrl##gr_ctrl", &gs.gr_ctrl_idx, ctrl_types, 4)) {
                                YAML::Node cn;
                                cn["type"]  = std::string(ctrl_types[gs.gr_ctrl_idx]);
                                cn["gain"]  = (double)gs.gr_gain;
                                cn["u_max"] = (double)gs.gr_umax;
                                if (gs.gr_ctrl_idx == 2) {
                                    cn["period"] = (double)gs.gr_period;
                                    cn["duty"]   = (double)gs.gr_duty;
                                }
                                if (gs.gr_ctrl_idx == 3) {
                                    cn["trigger"]      = (double)gs.gr_trigger;
                                    cn["hysteresis"]   = (double)gs.gr_hyst;
                                    cn["anticipation"] = (double)gs.gr_antic;
                                }
                                gs.grid->swap_controller(
                                    aniso::detail::make_controller<DIM>(cn));
                            }
                        }
                        ImGui::SameLine();
                        if (ImGui::SliderFloat("alpha##gr", &gs.gr_alpha, 0.0f, 2.0f, "%.2f"))
                            gs.grid->coup().set_alpha(gs.gr_alpha);
                        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Control-to-heating coupling.\nHigher = more energy injected per |u|.");

                        // Pulsed-specific sliders
                        if (gs.gr_ctrl_idx == 2) {
                            if (ImGui::SliderFloat("Period##gr", &gs.gr_period, 0.1f, 20.0f, "%.1f"))
                                gs.grid->ctrl().set_period(gs.gr_period);
                            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Pulse period (time units).");
                            ImGui::SameLine();
                            if (ImGui::SliderFloat("Duty##gr", &gs.gr_duty, 0.05f, 1.0f, "%.2f"))
                                gs.grid->ctrl().set_duty(gs.gr_duty);
                            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Duty cycle: fraction of period\nwhen controller is ON.");
                        }

                        // EventTriggered-specific sliders
                        if (gs.gr_ctrl_idx == 3) {
                            if (ImGui::SliderFloat("Trigger##gr", &gs.gr_trigger, 0.01f, 3.0f, "%.2f"))
                                gs.grid->ctrl().set_trigger(gs.gr_trigger);
                            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Activation threshold on |x| + anticipation*d|x|.\nLower = more sensitive.");
                            ImGui::SameLine();
                            if (ImGui::SliderFloat("Hyst##gr", &gs.gr_hyst, 0.1f, 1.0f, "%.2f"))
                                gs.grid->ctrl().set_hysteresis(gs.gr_hyst);
                            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Deactivation level = Trigger * Hyst.\nLower = stays on longer.");
                            ImGui::SameLine();
                            if (ImGui::SliderFloat("Antic##gr", &gs.gr_antic, 0.0f, 20.0f, "%.1f"))
                                gs.grid->ctrl().set_anticipation(gs.gr_antic);
                            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Anticipation weight on d|x|/dt.\nHigher = triggers earlier.");
                            bool evt_on = gs.grid->ctrl().is_active();
                            ImGui::SameLine();
                            ImGui::TextColored(evt_on ? ImVec4(0.2f,1,0.2f,1) : ImVec4(0.5f,0.5f,0.5f,1),
                                evt_on ? "[ON]" : "[off]");
                        }

                        // Row 2: Energy & relaxation
                        if (ImGui::SliderFloat("D_E##gr", &gs.gr_D_E, 0.0f, 3.0f, "%.2f"))
                            gs.grid->params().D_E = gs.gr_D_E;
                        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Energy diffusion coefficient.\nEnergy flows through G^{-1}.");
                        ImGui::SameLine();
                        if (ImGui::SliderFloat("diss##gr", &gs.gr_gamma_diss, 0.01f, 5.0f, "%.2f"))
                            gs.grid->params().gamma_diss = gs.gr_gamma_diss;
                        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Energy dissipation rate.\nHigher = energy drains faster.");
                        ImGui::SameLine();
                        if (ImGui::SliderFloat("tau##gr", &gs.gr_tau, 0.1f, 30.0f, "%.2f",
                                ImGuiSliderFlags_Logarithmic))
                            gs.grid->params().tau_0 = gs.gr_tau;
                        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Base G relaxation time (log scale).\nSmaller = faster relaxation.\nMust compete with drive.");
                        ImGui::SameLine();
                        if (ImGui::SliderFloat("k_tau##gr", &gs.gr_kappa_tau, 0.0f, 100.0f, "%.1f"))
                            gs.grid->params().kappa_tau = gs.gr_kappa_tau;
                        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Anisotropy slows relaxation.\ntau_eff = tau_0 * (1 + k_tau * aniso^2).\nHigher = barriers self-sustain longer.");
                        ImGui::SameLine();
                        if (ImGui::SliderFloat("Noise##gr", &gs.gr_noise_amp, 0.0f, 3.0f, "%.2f"))
                            gs.grid->params().noise_amp = gs.gr_noise_amp;
                        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Thermal noise amplitude.\nScaled by sqrt(local energy).\nDestroys structure at high E.");
                        ImGui::SameLine();
                        if (ImGui::SliderFloat("D_x##gr", &gs.gr_D_x, 0.0f, 1.0f, "%.2f"))
                            gs.grid->params().D_x = gs.gr_D_x;
                        if (ImGui::IsItemHovered()) ImGui::SetTooltip("State diffusion through G^{-1}.\nBarrier blocks state flow.");

                        // Row 3: Drive geometry
                        if (ImGui::SliderFloat("Peak##gr", &gs.gr_drive_peak, 0.5f, 15.0f, "%.1f")) {
                            gs.grid->params().drive_peak = gs.gr_drive_peak; drive_changed = true;
                        }
                        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Heating zone peak intensity");
                        ImGui::SameLine();
                        if (ImGui::SliderFloat("Rx##gr", &gs.gr_drive_rx, 0.02f, 0.8f, "%.2f")) {
                            gs.grid->params().drive_rx = gs.gr_drive_rx; drive_changed = true;
                        }
                        ImGui::SameLine();
                        if (ImGui::SliderFloat("Ry##gr", &gs.gr_drive_ry, 0.02f, 0.8f, "%.2f")) {
                            gs.grid->params().drive_ry = gs.gr_drive_ry; drive_changed = true;
                        }
                        ImGui::SameLine();
                        if (ImGui::SliderFloat("Cx##gr", &gs.gr_drive_cx, 0.0f, 1.0f, "%.2f")) {
                            gs.grid->params().drive_cx = gs.gr_drive_cx; drive_changed = true;
                        }
                        ImGui::SameLine();
                        if (ImGui::SliderFloat("Cy##gr", &gs.gr_drive_cy, 0.0f, 1.0f, "%.2f")) {
                            gs.grid->params().drive_cy = gs.gr_drive_cy; drive_changed = true;
                        }

                        ImGui::PopItemWidth();
                        if (drive_changed) gs.grid->rebuild_drive_profile();
                    }

                    ImGui::Separator();
                    float remaining_h = ImGui::GetContentRegionAvail().y;
                    float map_sz = std::min(remaining_h - 10, avail.x * 0.62f);
                    map_sz = std::max(map_sz, 200.0f);

                    // ---- 2D Map (switchable: Health/Energy/|x|/|u|) ----

                    // Pre-compute max for normalization
                    float map_max_E = 0.01f, map_max_xn = 0.01f, map_max_un = 0.01f;
                    for (int i = 0; i < Nx; ++i)
                        for (int j = 0; j < Ny; ++j) {
                            map_max_E  = std::max(map_max_E,  (float)gs.grid->E(i,j));
                            map_max_xn = std::max(map_max_xn, (float)gs.grid->x(i,j).norm());
                            map_max_un = std::max(map_max_un, (float)gs.grid->last_u(i,j).norm());
                        }

                    ImGui::BeginChild("##grid_left", {map_sz + 30, remaining_h}, false);
                    {
                        ImVec2 mp = ImGui::GetCursorScreenPos();
                        ImGui::InvisibleButton("##gmap", {map_sz, map_sz});
                        ImDrawList* dl = ImGui::GetWindowDrawList();
                        float cell_w = map_sz / Nx;
                        float cell_h = map_sz / Ny;

                        dl->AddRectFilled(mp, {mp.x + map_sz, mp.y + map_sz},
                            IM_COL32(10, 12, 20, 255));

                        for (int i = 0; i < Nx; ++i) {
                            for (int j = 0; j < Ny; ++j) {
                                ImU32 col;
                                if (gs.gr_map_mode == 0) {
                                    col = health_color((float)gs.grid->health(i,j));
                                } else {
                                    float t;
                                    if (gs.gr_map_mode == 1)
                                        t = (float)(gs.grid->E(i,j) / map_max_E);
                                    else if (gs.gr_map_mode == 2)
                                        t = (float)(gs.grid->x(i,j).norm() / map_max_xn);
                                    else
                                        t = (float)(gs.grid->last_u(i,j).norm() / map_max_un);
                                    t = std::clamp(t, 0.0f, 1.0f);
                                    int r = (int)(20 + 235*t);
                                    int g = (int)(20 + 200*t*(1-t)*4);
                                    int b = (int)(80*(1-t));
                                    col = IM_COL32(r, g, b, 255);
                                }
                                dl->AddRectFilled(
                                    {mp.x + i * cell_w, mp.y + j * cell_h},
                                    {mp.x + (i+1) * cell_w + 0.5f,
                                     mp.y + (j+1) * cell_h + 0.5f},
                                    col);
                            }
                        }

                        // Ellipse overlay: shows G tensor orientation at sampled nodes
                        int estep = std::max(1, Nx / 12);
                        for (int i = estep/2; i < Nx; i += estep) {
                            for (int j = estep/2; j < Ny; j += estep) {
                                auto& Gij = gs.grid->G(i, j);
                                Eigen::SelfAdjointEigenSolver<aniso::Mat<DIM>> sol(Gij.G);
                                auto ev = sol.eigenvalues();
                                auto evc = sol.eigenvectors();

                                float ecx = mp.x + (i + 0.5f) * cell_w;
                                float ecy = mp.y + (j + 0.5f) * cell_h;
                                float sc_e = cell_w * estep * 0.35f;
                                float a_ax = sqrtf(std::max((float)ev(0), 0.01f)) * sc_e;
                                float b_ax = sqrtf(std::max((float)ev(1), 0.01f)) * sc_e;
                                float rot = atan2f((float)evc(1,0), (float)evc(0,0));
                                float cr = cosf(rot), sr = sinf(rot);

                                const int NE = 16;
                                ImVec2 epts[17];
                                for (int k = 0; k <= NE; ++k) {
                                    float th = 2.0f * (float)M_PI * k / NE;
                                    float lx = a_ax * cosf(th);
                                    float ly = b_ax * sinf(th);
                                    epts[k] = {ecx + lx*cr - ly*sr,
                                               ecy + lx*sr + ly*cr};
                                }
                                float aniso_f = (float)gs.grid->anisotropy(i, j);
                                int ea = std::min(255, (int)(60 + 195 *
                                    std::min(aniso_f / 3.0f, 1.0f)));
                                dl->AddPolyline(epts, NE + 1,
                                    IM_COL32(255, 255, 255, ea),
                                    ImDrawFlags_Closed, 1.2f);
                            }
                        }

                        // Drive contour overlay (1- and 2-sigma ellipses)
                        {
                            float dcx = mp.x + gs.gr_drive_cx * map_sz;
                            float dcy = mp.y + gs.gr_drive_cy * map_sz;
                            float drx = gs.gr_drive_rx * map_sz;
                            float dry = gs.gr_drive_ry * map_sz;
                            const int NP = 32;
                            ImVec2 pts[33];
                            for (int k = 0; k <= NP; ++k) {
                                float th = 2.0f * (float)M_PI * k / NP;
                                pts[k] = {dcx + drx * cosf(th),
                                          dcy + dry * sinf(th)};
                            }
                            dl->AddPolyline(pts, NP + 1,
                                IM_COL32(255, 255, 255, 60),
                                ImDrawFlags_None, 1.0f);
                            for (int k = 0; k <= NP; ++k) {
                                float th = 2.0f * (float)M_PI * k / NP;
                                pts[k] = {dcx + drx*2 * cosf(th),
                                          dcy + dry*2 * sinf(th)};
                            }
                            dl->AddPolyline(pts, NP + 1,
                                IM_COL32(255, 255, 255, 30),
                                ImDrawFlags_None, 0.8f);
                        }

                        dl->AddText({mp.x + map_sz * 0.5f - 6, mp.y + map_sz + 2},
                            IM_COL32(150, 150, 170, 180), "x");
                        dl->AddText({mp.x - 14, mp.y + map_sz * 0.5f - 6},
                            IM_COL32(150, 150, 170, 180), "y");

                        // Color legend (mode-aware)
                        float lx = mp.x + 4, ly = mp.y + map_sz + 16;
                        for (int i = 0; i < 80; ++i) {
                            float v = (float)i / 79.0f;
                            ImU32 lc;
                            if (gs.gr_map_mode == 0) {
                                lc = health_color(v);
                            } else {
                                int r = (int)(20 + 235*v);
                                int g = (int)(20 + 200*v*(1-v)*4);
                                int b = (int)(80*(1-v));
                                lc = IM_COL32(r, g, b, 255);
                            }
                            dl->AddRectFilled({lx + i*1.5f, ly},
                                {lx + (i+1)*1.5f, ly + 8}, lc);
                        }
                        const char* lo[] = {"degraded","low E","low |x|","low |u|"};
                        const char* hi[] = {"pristine","high E","high |x|","high |u|"};
                        dl->AddText({lx - 2, ly + 9},
                            IM_COL32(200, 200, 200, 200), lo[gs.gr_map_mode]);
                        dl->AddText({lx + 95, ly + 9},
                            IM_COL32(200, 200, 200, 200), hi[gs.gr_map_mode]);
                    }
                    ImGui::EndChild();

                    ImGui::SameLine();

                    // ---- Right panel: Summary + Anisotropy map ----
                    ImGui::BeginChild("##grid_right", {0, remaining_h}, true);
                    {
                        ImGui::TextColored({0.7f,0.75f,0.8f,1}, "Grid Summary");
                        ImGui::Separator();

                        double avg_h = 0, min_h = 1, max_a = 0, avg_tr = 0;
                        double avg_E = 0, max_E = 0;
                        double avg_xn = 0, avg_un = 0;
                        int degraded = 0;
                        for (int i = 0; i < Nx; ++i) {
                            for (int j = 0; j < Ny; ++j) {
                                double h = gs.grid->health(i, j);
                                avg_h += h;
                                min_h = std::min(min_h, h);
                                max_a = std::max(max_a, gs.grid->anisotropy(i, j));
                                avg_tr += gs.grid->G(i, j).trace();
                                double e = gs.grid->E(i, j);
                                avg_E += e;
                                max_E = std::max(max_E, e);
                                avg_xn += gs.grid->x(i, j).norm();
                                avg_un += gs.grid->last_u(i, j).norm();
                                if (h < 0.5) ++degraded;
                            }
                        }
                        int total = Nx * Ny;
                        avg_h /= total; avg_tr /= total;
                        avg_E /= total; avg_xn /= total; avg_un /= total;

                        ImGui::TextColored({1.0f,0.9f,0.3f,1},
                            "E: avg %.2f  max %.2f", avg_E, max_E);
                        ImGui::Text("|x|: %.3f  |u|: %.3f", avg_xn, avg_un);
                        ImGui::Text("tr(G): %.2f  Aniso: %.2f", avg_tr, max_a);
                        ImGui::Text("Health: %.2f (min %.2f)", avg_h, min_h);
                        ImGui::Text("Degraded: %d/%d", degraded, total);

                        // Push data into ring buffer
                        if (gs.grid_running) {
                            int& p = gs.gr_hist_pos;
                            gs.gr_hist_x[p  % GuiState::GR_HIST] = (float)avg_xn;
                            gs.gr_hist_E[p  % GuiState::GR_HIST] = (float)avg_E;
                            gs.gr_hist_tr[p % GuiState::GR_HIST] = (float)avg_tr;
                            ++p;
                        }

                        ImGui::Spacing();
                        ImGui::Separator();
                        ImGui::TextColored({0.4f,0.85f,1.0f,1}, "Time History");

                        auto mini_plot = [&](const char* label, const float* buf,
                                             int buf_len, int pos,
                                             ImVec4 color, float plot_h,
                                             float cur_val) {
                            ImGui::TextColored(color, "%s: %.3f", label, cur_val);
                            int count = std::min(pos, buf_len);
                            int offset = (pos >= buf_len) ? (pos % buf_len) : 0;
                            ImGui::PushStyleColor(ImGuiCol_PlotLines,
                                ImGui::ColorConvertFloat4ToU32(color));
                            ImGui::PlotLines(("##" + std::string(label)).c_str(),
                                buf, count, offset,
                                nullptr, FLT_MAX, FLT_MAX,
                                {ImGui::GetContentRegionAvail().x, plot_h});
                            ImGui::PopStyleColor();
                        };

                        float ph = std::max(
                            (ImGui::GetContentRegionAvail().y - 50) / 3.0f,
                            36.0f);

                        mini_plot("avg |x|", gs.gr_hist_x,
                            GuiState::GR_HIST, gs.gr_hist_pos,
                            {0.3f, 1.0f, 0.3f, 1.0f}, ph, (float)avg_xn);
                        mini_plot("avg E", gs.gr_hist_E,
                            GuiState::GR_HIST, gs.gr_hist_pos,
                            {1.0f, 0.7f, 0.2f, 1.0f}, ph, (float)avg_E);
                        mini_plot("avg tr(G)", gs.gr_hist_tr,
                            GuiState::GR_HIST, gs.gr_hist_pos,
                            {0.5f, 0.5f, 1.0f, 1.0f}, ph, (float)avg_tr);
                    }
                    ImGui::EndChild();

                    ImGui::EndTabItem();
                }
            }

            // ============================================================
            //  TAB: SWEEP — parameter sweep curves (like control.ipynb)
            // ============================================================
            if (ImGui::BeginTabItem("Sweep")) {
                ImGui::Text("Parameter sweep: vary one parameter, compare controllers");
                ImGui::Separator();

                ImGui::SetNextItemWidth(200);
                ImGui::InputText("Param path", gs.sweep_param, sizeof(gs.sweep_param));
                ImGui::SameLine();
                {
                    const char* presets[] = {
                        "relaxation.tau", "coupling.alpha", "coupling.gamma",
                        "interaction.mu", "observation.sigma0", "observation.beta",
                        "controller.gain", "controller.u_max"
                    };
                    ImGui::SetNextItemWidth(150);
                    if (ImGui::BeginCombo("##sw_preset", "Presets...")) {
                        for (auto& p : presets) {
                            if (ImGui::Selectable(p))
                                std::snprintf(gs.sweep_param,
                                    sizeof(gs.sweep_param), "%s", p);
                        }
                        ImGui::EndCombo();
                    }
                }
                ImGui::SameLine();
                ImGui::SetNextItemWidth(80);
                ImGui::InputFloat("Lo", &gs.sweep_lo, 0, 0, "%.2f");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(80);
                ImGui::InputFloat("Hi", &gs.sweep_hi, 0, 0, "%.2f");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(80);
                ImGui::InputFloat("Step", &gs.sweep_step, 0, 0, "%.2f");
                ImGui::SameLine();

                if (ImGui::Button("Run Sweep", {120, 0})) {
                    try {
                        YAML::Node scfg = YAML::LoadFile(gs.config_path);
                        if (scfg["controllers"] && scfg["controllers"].IsSequence()) {
                            // Try to load sweep params from YAML if defaults
                            if (scfg["sweep"]) {
                                auto sw = scfg["sweep"];
                                if (sw["param"])
                                    std::snprintf(gs.sweep_param, sizeof(gs.sweep_param),
                                        "%s", sw["param"].as<std::string>().c_str());
                                if (sw["range"] && sw["range"].IsSequence() && sw["range"].size() >= 3) {
                                    gs.sweep_lo   = sw["range"][0].as<float>();
                                    gs.sweep_hi   = sw["range"][1].as<float>();
                                    gs.sweep_step = sw["range"][2].as<float>();
                                }
                            }
                            gs.sweep_rows = aniso::run_sweep<DIM>(scfg,
                                gs.sweep_param,
                                static_cast<double>(gs.sweep_lo),
                                static_cast<double>(gs.sweep_hi),
                                static_cast<double>(gs.sweep_step));
                            gs.sweep_ran = true;
                        }
                    } catch (...) {}
                }

                if (gs.sweep_ran && !gs.sweep_rows.empty()) {
                    int n_rows = static_cast<int>(gs.sweep_rows.size());
                    int n_ctrl = static_cast<int>(gs.sweep_rows[0].results.size());
                    if (n_ctrl > 0 && n_rows > 1) {
                        ImVec4 colors[] = {
                            {0.2f,0.7f,1.0f,1}, {1.0f,0.4f,0.2f,1}, {0.2f,0.9f,0.3f,1},
                            {1.0f,0.8f,0.1f,1}, {0.8f,0.3f,0.9f,1}, {0.9f,0.6f,0.4f,1},
                            {0.5f,0.5f,1.0f,1}, {1.0f,0.6f,0.8f,1}
                        };

                        // Prepare per-controller data vectors
                        std::vector<std::vector<double>> param_vals(n_ctrl);
                        std::vector<std::vector<double>> err_curves(n_ctrl);
                        std::vector<std::vector<double>> eff_curves(n_ctrl);
                        std::vector<std::string> ctrl_names(n_ctrl);

                        for (int c = 0; c < n_ctrl; ++c) {
                            ctrl_names[c] = gs.sweep_rows[0].results[c].name;
                            param_vals[c].resize(n_rows);
                            err_curves[c].resize(n_rows);
                            eff_curves[c].resize(n_rows);
                            for (int r = 0; r < n_rows; ++r) {
                                param_vals[c][r] = gs.sweep_rows[r].param_value;
                                err_curves[c][r] = gs.sweep_rows[r].results[c].metrics.mean_error;
                                eff_curves[c][r] = gs.sweep_rows[r].results[c].metrics.mean_effort;
                            }
                        }

                        float w_half = (ImGui::GetContentRegionAvail().x - style.ItemSpacing.x) * 0.5f;
                        float h_plot = std::max(
                            (ImGui::GetContentRegionAvail().y - 20.0f) * 0.5f, 200.0f);

                        // --- Plot 1: Error vs Parameter (control-precision limit) ---
                        std::string xlabel = std::string(gs.sweep_param);
                        if (ImPlot::BeginPlot("Tracking Error vs Parameter##sweep_err",
                                ImVec2(w_half, h_plot))) {
                            ImPlot::SetupAxes(xlabel.c_str(), "Mean Error |x|");
                            for (int c = 0; c < n_ctrl; ++c) {
                                ImPlot::PushStyleColor(ImPlotCol_Line, colors[c % 8]);
                                ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 2.5f);
                                ImPlot::PlotLine(ctrl_names[c].c_str(),
                                    param_vals[c].data(), err_curves[c].data(), n_rows);
                                ImPlot::PopStyleVar();
                                ImPlot::PopStyleColor();
                            }
                            ImPlot::EndPlot();
                        }

                        ImGui::SameLine();

                        // --- Plot 2: Effort vs Parameter ---
                        if (ImPlot::BeginPlot("Control Effort vs Parameter##sweep_eff",
                                ImVec2(w_half, h_plot))) {
                            ImPlot::SetupAxes(xlabel.c_str(), "Mean Effort |u|^2");
                            for (int c = 0; c < n_ctrl; ++c) {
                                ImPlot::PushStyleColor(ImPlotCol_Line, colors[c % 8]);
                                ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 2.5f);
                                ImPlot::PlotLine(ctrl_names[c].c_str(),
                                    param_vals[c].data(), eff_curves[c].data(), n_rows);
                                ImPlot::PopStyleVar();
                                ImPlot::PopStyleColor();
                            }
                            ImPlot::EndPlot();
                        }

                        // --- Plot 3: Error vs Effort (trade-off curves) ---
                        if (ImPlot::BeginPlot("Error vs Effort (trade-off)##sweep_tradeoff",
                                ImVec2(w_half, h_plot))) {
                            ImPlot::SetupAxes("Mean Effort |u|^2", "Mean Error |x|");
                            for (int c = 0; c < n_ctrl; ++c) {
                                ImPlot::PushStyleColor(ImPlotCol_Line, colors[c % 8]);
                                ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 2.5f);
                                ImPlot::PlotLine(ctrl_names[c].c_str(),
                                    eff_curves[c].data(), err_curves[c].data(), n_rows);
                                ImPlot::PopStyleVar();
                                ImPlot::PopStyleColor();
                            }
                            ImPlot::EndPlot();
                        }

                        ImGui::SameLine();

                        // --- Plot 4: Efficiency vs Parameter ---
                        if (ImPlot::BeginPlot("Efficiency vs Parameter##sweep_effic",
                                ImVec2(w_half, h_plot))) {
                            ImPlot::SetupAxes(xlabel.c_str(), "Efficiency (lower=better)");
                            for (int c = 0; c < n_ctrl; ++c) {
                                std::vector<double> effic(n_rows);
                                for (int r = 0; r < n_rows; ++r)
                                    effic[r] = gs.sweep_rows[r].results[c].metrics.efficiency;
                                ImPlot::PushStyleColor(ImPlotCol_Line, colors[c % 8]);
                                ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 2.5f);
                                ImPlot::PlotLine(ctrl_names[c].c_str(),
                                    param_vals[c].data(), effic.data(), n_rows);
                                ImPlot::PopStyleVar();
                                ImPlot::PopStyleColor();
                            }
                            ImPlot::EndPlot();
                        }

                        // --- Best parameters summary ---
                        int best_ctrl = -1, best_row = -1;
                        double best_eff_val = 1e30;
                        for (int r = 0; r < n_rows; ++r) {
                            for (int c = 0; c < n_ctrl; ++c) {
                                double e = gs.sweep_rows[r].results[c].metrics.efficiency;
                                if (e < best_eff_val) {
                                    best_eff_val = e;
                                    best_ctrl = c;
                                    best_row = r;
                                }
                            }
                        }
                        if (best_ctrl >= 0) {
                            ImGui::Separator();
                            auto& br = gs.sweep_rows[best_row].results[best_ctrl];
                            ImGui::TextColored({0.3f, 1.0f, 0.4f, 1},
                                "BEST: %s @ %s = %.3f  "
                                "(error=%.4f, effort=%.4f, efficiency=%.5f)",
                                br.name.c_str(), gs.sweep_param,
                                gs.sweep_rows[best_row].param_value,
                                br.metrics.mean_error, br.metrics.mean_effort,
                                br.metrics.efficiency);

                            for (int c = 0; c < n_ctrl; ++c) {
                                int br_row = -1;
                                double be = 1e30;
                                for (int r = 0; r < n_rows; ++r) {
                                    double e = gs.sweep_rows[r].results[c].metrics.efficiency;
                                    if (e < be) { be = e; br_row = r; }
                                }
                                if (br_row >= 0) {
                                    ImGui::Text("  %s: best @ %s=%.3f (eff=%.5f)",
                                        ctrl_names[c].c_str(), gs.sweep_param,
                                        gs.sweep_rows[br_row].param_value, be);
                                }
                            }

                            if (ImGui::Button("Launch Best in Grid", {180, 0})) {
                                set_yaml_path(gs.current_cfg, gs.sweep_param,
                                    gs.sweep_rows[best_row].param_value);
                                auto controllers = gs.current_cfg["controllers"];
                                if (controllers && controllers.IsSequence()
                                    && best_ctrl < (int)controllers.size()) {
                                    gs.current_cfg["controller"] =
                                        controllers[best_ctrl];
                                }
                                rebuild_grid(gs);
                            }
                            if (ImGui::IsItemHovered())
                                ImGui::SetTooltip(
                                    "Load best sweep parameters into Grid 2D tab");
                        }
                    }
                } else if (gs.sweep_ran) {
                    ImGui::TextColored({1,0.5f,0.3f,1},
                        "No 'controllers' or 'sweep' section found. Use sweep config.");
                }
                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }
        ImGui::End();

        // ---- Render ----
        ImGui::Render();
        int dw, dh;
        glfwGetFramebufferSize(window, &dw, &dh);
        glViewport(0, 0, dw, dh);
        glClearColor(0.08f, 0.08f, 0.10f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    ImPlot::DestroyContext();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
