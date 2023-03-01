export OneHotStateReprWrapper, FrameStackWrapper, NormalizeWrapper
import Statistics

mutable struct OneHotStateReprWrapper{T<:AbstractFloat} <: AbstractMDP{Vector{T}, Int}
    env::AbstractMDP{Int, Int}
    ss::VectorSpace{T}
    state::Vector{T}
    function OneHotStateReprWrapper{T}(env::AbstractMDP{Int, Int}) where {T<:AbstractFloat}
        n = length(state_space(env))
        ss = VectorSpace{T}(T(0), T(1), (n, ))
        new{T}(env, ss, zeros(T, n))
    end
end

function factory_reset!(env::OneHotStateReprWrapper)
    factory_reset!(env.env)
end


@inline action_space(env::OneHotStateReprWrapper) = action_space(env.env)
@inline state_space(env::OneHotStateReprWrapper) = env.ss
@inline action_meaning(env::OneHotStateReprWrapper, a::Int) = action_meaning(env.env, a)


state(env::OneHotStateReprWrapper) = env.state
action(env::OneHotStateReprWrapper) = action(env.env)
reward(env::OneHotStateReprWrapper) = reward(env.env)

function reset!(env::OneHotStateReprWrapper; rng::AbstractRNG=Random.GLOBAL_RNG)
    reset!(env.env; rng=rng)
    env.state = to_onehot(env, state(env.env))
    nothing
end

function step!(env::OneHotStateReprWrapper, a::Int; rng::AbstractRNG=Random.GLOBAL_RNG)
    step!(env.env, a; rng=rng)
    env.state = to_onehot(env, state(env.env))
    nothing
end

@inline in_absorbing_state(env::OneHotStateReprWrapper) = in_absorbing_state(env.env)
@inline truncated(env::OneHotStateReprWrapper) = truncated(env.env)

@inline visualize(env::OneHotStateReprWrapper, args...; kwargs...) = visualize(env.env, args...; kwargs...)

function to_onehot(env::OneHotStateReprWrapper{T}, s::Int) where T
    _s = zeros(T, length(state_space(env.env)))
    _s[s] = 1
    return _s
end

function to_onehot(x::Int, max_x::Int, T=Float32)
    onehot_x = zeros(T, max_x)
    onehot_x[x] = 1
    return onehot_x
end





struct FrameStackWrapper{T, A} <: AbstractMDP{Vector{T}, A}
    env::AbstractMDP{Vector{T}, A}
    ss::VectorSpace{T}
    state::Vector{T}
    function FrameStackWrapper(env::AbstractMDP{Vector{T}, A}, k::Int=4) where {T, A}
        env_ss = state_space(env)
        ss = VectorSpace{T}(repeat(env_ss.lows, k), repeat(env_ss.highs, k))
        return new{T, A}(env, ss, repeat(state(env), k))
    end
end

function factory_reset!(env::FrameStackWrapper)
    factory_reset!(env.env)
end

@inline action_space(env::FrameStackWrapper) = action_space(env.env)
@inline state_space(env::FrameStackWrapper) = env.ss
@inline action_meaning(env::FrameStackWrapper, a) = action_meaning(env.env, a)


@inline state(env::FrameStackWrapper) = env.state
@inline action(env::FrameStackWrapper) = action(env.env)
@inline reward(env::FrameStackWrapper) = reward(env.env)

function reset!(env::FrameStackWrapper; rng::AbstractRNG=Random.GLOBAL_RNG)
    reset!(env.env; rng=rng)
    m = size(state_space(env.env), 1)
    env.state[1:end-m] .= 0
    env.state[end-m+1:end] = state(env.env)
    nothing
end

function step!(env::FrameStackWrapper{T, A}, a::A; rng::AbstractRNG=Random.GLOBAL_RNG) where {T, A}
    step!(env.env, a; rng=rng)
    m = size(state_space(env.env), 1)
    env.state[1:end-m] = @view env.state[m+1:end]
    env.state[end-m+1:end] = state(env.env)
    nothing
end

@inline in_absorbing_state(env::FrameStackWrapper) = in_absorbing_state(env.env)
@inline truncated(env::FrameStackWrapper) = truncated(env.env)

@inline visualize(env::FrameStackWrapper, args...; kwargs...) = visualize(env.env, args...; kwargs...)



Base.@kwdef mutable struct RunningMeanVariance{T, N}
    const shape::NTuple{N, Int}
    const μ::Array{Float64, N} = zeros(Float64, shape...)
    const M₂::Array{Float64, N} = zeros(Float64, shape...)
    n::Int = 0
end

function reset_rmv!(rmv::RunningMeanVariance)
    rmv.n = 0
    fill!(rmv.μ, 0)
    fill!(rmv.M₂, 0)
    nothing
end

function increment_rmv!(rmv::RunningMeanVariance{T, N}, x::Union{AbstractArray{T, N}, T}) where {T, N}
    rmv.n += 1
    Δ = x .- rmv.μ
    rmv.μ .+= Δ / rmv.n
    rmv.M₂ .+= Δ .* (x .- rmv.μ)
    nothing
end

function Statistics.mean(rmv::RunningMeanVariance{T, N})::Union{T, Array{T, N}} where {T, N}
    if rmv.n < 1
        μ = fill(NaN, rmv.shape...)
    else
        μ = rmv.μ
    end
    if N == 0
        return T(μ[])
    else
        return convert(Array{T, N}, μ)
    end
end

function Statistics.var(rmv::RunningMeanVariance{T, N}; corrected::Bool=true)::Union{T, Array{T, N}} where {T, N}
    if rmv.n < 2
        σ² = fill(NaN, rmv.shape...)
    else
        σ² = rmv.M₂ / (rmv.n - Int(corrected))
    end
    if N == 0
        return T(σ²[])
    else
        return convert(Array{T, N}, σ²)
    end
end

function Statistics.std(rmv::RunningMeanVariance{T, N}; corrected::Bool=true)::Union{T, Array{T, N}} where {T, N}
    if rmv.n < 2
        σ = N == 0 ? NaN : fill(NaN, rmv.shape...)
    else
        σ = sqrt.(rmv.M₂ / (rmv.n - Int(corrected)))
    end
    if N == 0
        return T(σ) # sqrt converts 0-dim Array to scalar. No need to do σ[]
    else
        return convert(Array{T, N}, σ)
    end
end

Base.@kwdef struct NormalizeWrapper{T, N, A} <: AbstractMDP{Array{T, N}, A}
    env::AbstractMDP{Array{T, N}, A}
    obs_rmv::RunningMeanVariance{T, N} = RunningMeanVariance{T, N}(shape=size(state_space(env))[1:end-1])
    rew_rmv::RunningMeanVariance{Float64, 0} = RunningMeanVariance{Float64, 0}(shape=())
    normalize_obs::Bool = true
    normalize_reward::Bool = true
    clip_obs::T = T(100.0)
    clip_reward::Float64 = 100.0
    γ::Float64 = 0.99
    ϵ::Float64 = 1e-4
end

function NormalizeWrapper(env::AbstractMDP{Array{T, N}, A}; kwargs...) where {T, N, A}
    # println("here")
    NormalizeWrapper{T, N, A}(env=env; kwargs...)
end
function factory_reset!(env::NormalizeWrapper)
    reset_rmv!(env.obs_rmv)
    reset_rmv!(env.rew_rmv)
    factory_reset!(env.env)
end

@inline action_space(env::NormalizeWrapper) = action_space(env.env)
function state_space(env::NormalizeWrapper)
    ss = deepcopy(state_space(env.env))
    ss.lows .= -env.clip_obs
    ss.highs .= env.clip_obs
    return ss
end
@inline action_meaning(env::NormalizeWrapper, a) = action_meaning(env.env, a)
@inline action(env::NormalizeWrapper) = action(env.env)


function state(env::NormalizeWrapper{T, N, A})::Array{T, N} where {T, N, A}
    if env.normalize_obs
        if env.obs_rmv.n < 2
            return state(env.env)
        else
            return clamp.((state(env.env) .- mean(env.obs_rmv)) ./ (std(env.obs_rmv) .+ T(env.ϵ)), -env.clip_obs, env.clip_obs)
        end
    else
        return state(env.env)
    end
end

function reward(env::NormalizeWrapper)
    if env.normalize_reward
        if env.rew_rmv.n < 2
            return reward(env.env)
        else
            return clamp((reward(env.env) - mean(env.rew_rmv)) / (std(env.rew_rmv) + env.ϵ), -env.clip_reward, env.clip_reward)
        end
    else
        return reward(env.env)
    end
end

function reset!(env::NormalizeWrapper{T, N, A}; rng::AbstractRNG=Random.GLOBAL_RNG) where {T, N, A}
    reset!(env.env; rng=rng)
    increment_rmv!(env.obs_rmv, state(env.env))
    nothing
end

function step!(env::NormalizeWrapper{T, N, A}, a::A; rng::AbstractRNG=Random.GLOBAL_RNG) where {T, N, A}
    step!(env.env, a; rng=rng)
    increment_rmv!(env.obs_rmv, state(env.env))
    increment_rmv!(env.rew_rmv, reward(env.env))
    nothing
end

@inline in_absorbing_state(env::NormalizeWrapper) = in_absorbing_state(env.env)
@inline truncated(env::NormalizeWrapper) = truncated(env.env)

@inline visualize(env::NormalizeWrapper, args; kwargs...) = visualize(env.env, args...; kwargs...)
