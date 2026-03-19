#pragma once

#include "types.hpp"
#include "coupling.hpp"
#include "interaction.hpp"
#include "feedback.hpp"
#include "observer.hpp"
#include "controller.hpp"
#include "recorder.hpp"
#include <memory>

namespace aniso {

template<int Dim>
struct EngineParams {
    Mat<Dim> A = Mat<Dim>::Zero();
    Mat<Dim> B = Mat<Dim>::Identity();
    Vec<Dim> w = Vec<Dim>::Zero();
    Vec<Dim> x0 = Vec<Dim>::Zero();
    TensorField<Dim> G0;
    double tau = 5.0;
    double dt = 0.01;
    double t_end = 60.0;
    double eig_lo = 0.3;
    double eig_hi = 5.0;
    uint64_t seed = 42;
};

template<int Dim>
struct SimResult {
    Recorder<Dim> recorder;
    Metrics<Dim> metrics;
};

template<int Dim>
class Engine {
    std::unique_ptr<ICoupling<Dim>>    coupling_;
    std::unique_ptr<IInteraction<Dim>> interaction_;
    std::unique_ptr<IFeedback<Dim>>    feedback_;
    std::unique_ptr<IObserver<Dim>>    observer_;
    std::unique_ptr<IController<Dim>>  controller_;

    EngineParams<Dim> params_;
    SimState<Dim> state_;
    RNG rng_;
    Recorder<Dim> recorder_;
    Vec<Dim> last_u_ = Vec<Dim>::Zero();
    bool initialized_ = false;

public:
    Engine(EngineParams<Dim> p,
           std::unique_ptr<ICoupling<Dim>>    coupling,
           std::unique_ptr<IInteraction<Dim>> interaction,
           std::unique_ptr<IFeedback<Dim>>    feedback,
           std::unique_ptr<IObserver<Dim>>    observer,
           std::unique_ptr<IController<Dim>>  controller)
        : coupling_(std::move(coupling))
        , interaction_(std::move(interaction))
        , feedback_(std::move(feedback))
        , observer_(std::move(observer))
        , controller_(std::move(controller))
        , params_(std::move(p))
    {}

    void reset() {
        state_.x = params_.x0;
        state_.G = params_.G0;
        state_.t = 0.0;
        rng_.seed(params_.seed);
        recorder_.clear();
        last_u_ = Vec<Dim>::Zero();
        initialized_ = true;
    }

    bool step() {
        if (!initialized_) reset();
        if (state_.t > params_.t_end + params_.dt * 0.5) return false;

        const double dt = params_.dt;
        const auto I = Mat<Dim>::Identity();

        auto obs = observer_->observe(state_.x, state_.G, rng_);
        Vec<Dim> u = controller_->compute(state_.t, obs);
        last_u_ = u;
        recorder_.push(state_.t, state_.x, u, state_.G);

        Vec<Dim> fb = feedback_->coupling(state_.G, state_.x);
        Vec<Dim> dx = params_.A * state_.x + fb + params_.B * u + params_.w;
        state_.x += dx * dt;

        Mat<Dim> drive = coupling_->drive(u);
        Mat<Dim> relax = -(state_.G.G - I) / params_.tau;
        Mat<Dim> inter = interaction_->compute(state_.G);
        state_.G.G += (drive + relax + inter) * dt;
        state_.G.symmetrize();
        state_.G.clamp_eigenvalues(params_.eig_lo, params_.eig_hi);

        state_.t += dt;
        return true;
    }

    SimResult<Dim> run() {
        reset();
        while (step()) {}
        auto metrics = recorder_.compute_metrics(5.0);
        return {std::move(recorder_), metrics};
    }

    const SimState<Dim>&  state()    const { return state_; }
    const Recorder<Dim>&  recorder() const { return recorder_; }
    const Vec<Dim>&       last_u()   const { return last_u_; }
    const EngineParams<Dim>& params() const { return params_; }
    EngineParams<Dim>&    params()         { return params_; }
    bool                  done()     const { return state_.t > params_.t_end + params_.dt * 0.5; }

    IController<Dim>&  ctrl()   { return *controller_; }
    ICoupling<Dim>&    coup()   { return *coupling_; }
    IObserver<Dim>&    obs()    { return *observer_; }
    IInteraction<Dim>& inter()  { return *interaction_; }

    void swap_controller(std::unique_ptr<IController<Dim>> c) {
        controller_ = std::move(c);
    }
};

} // namespace aniso
