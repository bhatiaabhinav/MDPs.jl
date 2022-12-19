using StatsBase
using ProgressMeter

export EmptyHook, EmpiricalPolicyEvaluationHook, ProgressMeterHook


struct EmptyHook <: AbstractHook end


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