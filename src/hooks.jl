using StatsBase
using ProgressMeter

export EmptyHook, EmpiricalPolicyEvaluationHook, ProgressMeterHook

"""
    EmptyHook()

Empty hook that does nothing.
"""
struct EmptyHook <: AbstractHook end


"""
    EmpiricalPolicyEvaluationHook(π, γ, horizon, n, sample_size; env=nothing)

Hook that evaluates the policy `π` every `n` episodes in the experiment using `sample_size` trajectory samples. The mean returns are stored in the `returns` list of the hook. The evaluation is done on the environment `env`. If no environment is provided, the environment used in the experiment is used instead. The trajectories are truncated at `horizon` steps and the discount factor for recording returns is `γ`. Internally, the `interact` function is used to generate the trajectories and the mean return is computed as: `mean(interact(env, π, γ, horizon, sample_size)[1])`.
"""
struct EmpiricalPolicyEvaluationHook <: AbstractHook
    π::AbstractPolicy
    γ::Real
    horizon::Real
    n::Int
    sample_size::Int
    env::Union{Nothing, AbstractMDP}
    returns::Vector{Float64}
    EmpiricalPolicyEvaluationHook(π::AbstractPolicy, γ::Real, horizon::Real, n::Int, sample_size::Int; env::Union{Nothing, AbstractMDP}=nothing) = new(π, γ, horizon, n, sample_size, env, Float64[])
end

function preexperiment(peh::EmpiricalPolicyEvaluationHook; env, kwargs...)
    _env = isnothing(peh.env) ? env : peh.env
    push!(peh.returns, mean(interact(_env, peh.π, peh.γ, peh.horizon, peh.sample_size)[1]))
end

function postepisode(peh::EmpiricalPolicyEvaluationHook; env, returns, kwargs...)
    if length(returns) % peh.n == 0
        _env = isnothing(peh.env) ? env : peh.env
        push!(peh.returns, mean(interact(_env, peh.π, peh.γ, peh.horizon, peh.sample_size)[1]))
    end
end




"""
    ProgressMeterHook(; kwargs...)

Hook that uses a ProgressMeter to display progress as the experiment is running. Progress is defined as the number of episodes completed divided by `max_trials`. The hook can be initialized with any keyword arguments accepted by `ProgressMeter.Progress`.
"""
struct ProgressMeterHook <: AbstractHook
    progress::Progress
    ProgressMeterHook(; kwargs...) = new(Progress(0; kwargs...))
end

function postepisode(pmh::ProgressMeterHook; max_trials, kwargs...)
    pmh.progress.n = max_trials
    next!(pmh.progress)   
end

function postexperiment(pmh::ProgressMeterHook; kwargs...)
    finish!(pmh.progress)
end