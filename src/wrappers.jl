export OneHotStateReprWrapper, FrameStackWrapper, NormalizeWrapper, EvidenceObservationWrapper, TimeLimitWrapper
import Statistics

"""
    OneHotStateReprWrapper{T}(env::AbstractMDP{Int, Int}) where {T<:AbstractFloat}

Wrapper that converts the state representation of an MDP from integers to one-hot vectors. The state space of the wrapped MDP is a `VectorSpace{T}`. Each element of the new state space is a one-hot vector of length `n`, where `n` is the number of states in the wrapped MDP.
"""
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
    fill!(env.state, 0)
    env.state[state(env.env)] = 1
    nothing
end

function step!(env::OneHotStateReprWrapper, a::Int; rng::AbstractRNG=Random.GLOBAL_RNG)
    step!(env.env, a; rng=rng)
    fill!(env.state, 0)
    env.state[state(env.env)] = 1
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

"""
    to_onehot(x::Int, max_x::Int, T=Float32)

Converts an integer `x` to a one-hot vector of length `max_x` with eltype `T`.
"""
function to_onehot(x::Int, max_x::Int, T=Float32)
    onehot_x = zeros(T, max_x)
    onehot_x[x] = 1
    return onehot_x
end




"""
    FrameStackWrapper{T, A}(env::AbstractMDP{Vector{T}, A}, k::Int=4) where {T, A}

Wrapper that stacks the last `k` observations of an MDP's state space into a single vector. The state space of the wrapped MDP is a `VectorSpace{T}`. Each element of the new state space is a vector of length `n*k`, where `n` is the length of the states in the wrapped MDP.
"""
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



"""
    NormalizeWrapper{T, N, A}(env::AbstractMDP{Vector{T, N}, A}, normalize_obs::Bool=true, normalize_reward::Bool=true, clip_obs::T=100.0, clip_reward::Float64=100.0, γ::Float64=0.99, ϵ::Float64=1e-4) where {T, N, A}

    NormalizeWrapper(env::AbstractMDP; kwargs...)

Wrapper that normalizes the observations and rewards of an MDP. The states of the wrapped MDP are of type `Array{T, N}` and actions are of type `A`. The observations are normalized by subtracting the mean and dividing by the standard deviation. The rewards are normalized by dividing by the standard deviation of a rolling discounted sum of the rewards.

Note: It is important to save the normalization statistics after training and load them before testing. Otherwise, the agent will not behave correctly. At testing time, you should not update the normalization statistics. You can do this by setting `wrapper_env.update_stats=false` or by passing `update_stats=false` to the constructor.

Note: If you wish to wrap multiple environments with the same normalization statistics, you can do so by passing the same `obs_rmv`, `rew_rmv` and `ret_rmv` to each wrapper. This is useful if you want to train multiple agents in parallel with the same normalization statistics. For example, you can do the following:

```julia
env = ... # some environment with states of type Array{T, N}`
obs_rmv = RunningMeanVariance{T, N}(shape=size(state_space(env))[1:end-1])
rew_rmv = RunningMeanVariance{Float64}()
ret_rmv = RunningMeanVariance{Float64}()
env = NormalizeWrapper(env, obs_rmv=obs_rmv, rew_rmv=rew_rmv, ret_rmv=ret_rmv, kwargs...)
# or
env = NormalizeWrapper(env, kwargs...)
obs_rmv, rew_rmv, ret_rmv = env.obs_rmv, env.rew_rmv, env.ret_rmv

# then you can wrap other environments with the same normalization statistics
env1 = NormalizeWrapper(env1, obs_rmv=obs_rmv, rew_rmv=rew_rmv, ret_rmv=ret_rmv, kwargs...)
env2 = NormalizeWrapper(env2, obs_rmv=obs_rmv, rew_rmv=rew_rmv, ret_rmv=ret_rmv, kwargs...)
...
```

# Arguments
- `env::AbstractMDP`: the environment to wrap
- `normalize_obs::Bool=true`: whether to normalize the observations
- `normalize_reward::Bool=true`: whether to normalize the rewards
- `normalize_reward_by_reward_std::Bool=false`: If true, reward are normalized by standard deviation of the rewards. By default, rewards are normalized by the standard deviation of a rolling discounted sum of the rewards.
- `clip_obs::T=100.0`: the absolute value to clip the observations to
- `clip_reward::Float64=100.0`: the absolute value to clip the rewards to
- `γ::Float64=0.99`: the discount factor
- `ϵ::Float64=1e-4`: a small value to avoid division by zero
- `update_stats=true`: whether to update the normalization statistics. Set this to `false` at testing time.
- `obs_rmv=RunningMeanVariance{T, N}(shape=size(state_space(env))[1:end-1])`: a data structure to maintain the running mean and variance of the observations
- `ret_rmv=RunningMeanVariance{Float64}()`: a data structure to maintain the running variance of rolling discounted returns
- `rew_rmv=RunningMeanVariance{Float64}()`: a data structure to maintain the running variance of the rewards


# References
-  Stable Baselines3 implementation of VecNormalize: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/vec_normalize.py)
""" 
Base.@kwdef mutable struct NormalizeWrapper{T, N, A} <: AbstractMDP{Array{T, N}, A}
    const env::AbstractMDP{Array{T, N}, A}
    normalize_obs::Bool = true
    normalize_reward::Bool = true
    normalize_reward_by_reward_std::Bool = false  # if false, normalize by return std
    const clip_obs::T = T(100.0)
    const clip_reward::Float64 = 100.0
    const γ::Float64 = 0.99
    const ϵ::Float64 = 1e-4
    update_stats = true
    obs_rmv::RunningMeanVariance{T, N} = RunningMeanVariance{T, N}(shape=size(state_space(env))[1:end-1])
    ret_rmv::RunningMeanVariance{Float64, 0} = RunningMeanVariance{Float64}()
    rew_rmv::RunningMeanVariance{Float64, 0} = RunningMeanVariance{Float64}()
    ret::Float64 = 0.0
end

function NormalizeWrapper(env::AbstractMDP{Array{T, N}, A}; kwargs...) where {T, N, A}
    NormalizeWrapper{T, N, A}(; env=env, kwargs...)
end
function factory_reset!(env::NormalizeWrapper)
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
        if env.normalize_reward_by_reward_std
            divisor = std(env.rew_rmv)
            if env.rew_rmv.n < 2 || divisor == 0
                divisor = 1
            end
        else
            divisor = std(env.ret_rmv)
            if env.ret_rmv.n < 2 || divisor == 0
                divisor = 1
            end
        end
        return clamp(reward(env.env) / (divisor + env.ϵ), -env.clip_reward, env.clip_reward)
    else
        return reward(env.env)
    end
end

function reset!(env::NormalizeWrapper{T, N, A}; rng::AbstractRNG=Random.GLOBAL_RNG) where {T, N, A}
    reset!(env.env; rng=rng)
    env.ret = 0
    env.update_stats && push!(env.obs_rmv, state(env.env))
    nothing
end

function step!(env::NormalizeWrapper{T, N, A}, a::A; rng::AbstractRNG=Random.GLOBAL_RNG) where {T, N, A}
    step!(env.env, a; rng=rng)
    r::Float64 = reward(env.env)
    env.ret = env.γ * env.ret + r
    if env.update_stats
        push!(env.obs_rmv, state(env.env))
        push!(env.ret_rmv, env.ret)
        push!(env.rew_rmv, r)
    end
    nothing
end

@inline in_absorbing_state(env::NormalizeWrapper) = in_absorbing_state(env.env)
@inline truncated(env::NormalizeWrapper) = truncated(env.env)

@inline visualize(env::NormalizeWrapper, args...; kwargs...) = visualize(env.env, args...; kwargs...)




"""
    EvidenceObservationWrapper{T}(env::AbstractMDP{S, A}) where {T <: AbstractFloat, S, A}

A wrapper that emits an evidence vector as the observation. An evidence vector is a concatenation of the latest action, the latest reward, a flag indicating whether the current state marks the start of a new episode, and the current state. This is useful for solving POMDPs. In deep RL, this wrapper is used in conjunction with a recurrent neural network that takes in an evidence vector as input at each step and outputs a policy.
"""

struct EvidenceObservationWrapper{T, S, A} <: AbstractMDP{Vector{T}, A}
    env::AbstractMDP{S, A}
    evidence::Vector{T}  # current evidence vector. An evidence vector is a concatenation of the latest action, the latest reward, a flag indicating whether the current state marks the start of a new episode, and the current state.
    𝕊::VectorSpace{T}
    function EvidenceObservationWrapper{T}(env::AbstractMDP{S, A}) where {T <: AbstractFloat, S, A}
        m, n = size(state_space(env), 1), size(action_space(env), 1)
        return new{T, S, A}(env, Vector{T}(undef, 1+n+1+m), evidence_state_space(env, T))
    end
end

function set_evidence!(env::EvidenceObservationWrapper{T, S, A}, new_episode_flag::Bool, latest_action::A, latest_reward::Float64, latest_state::S) where {T, S, A}
    m, n = size(state_space(env.env), 1), size(action_space(env.env), 1)
    env.evidence[n+2] = T(new_episode_flag)
    if new_episode_flag
        env.evidence[1:n+1] .= 0
    else
        if A == Int
            env.evidence[1:n] .= 0
            env.evidence[latest_action] = 1
        else
            env.evidence[1:n] .= latest_action
        end
        env.evidence[1+n] = latest_reward
    end
    if S == Int
        env.evidence[end-m+1:end] .= 0
        env.evidence[end-m+latest_state] = 1
    else
        env.evidence[end-m+1:end] .= latest_state
    end
    nothing
end

state(env::EvidenceObservationWrapper) = env.evidence

function factory_reset!(env::EvidenceObservationWrapper)
    factory_reset!(env.env)
    nothing
end

function reset!(env::EvidenceObservationWrapper{T, S, A}; rng::AbstractRNG=Random.GLOBAL_RNG) where {T, S, A}
    reset!(env.env; rng=rng)
    set_evidence!(env, true, action(env.env), reward(env.env), state(env.env))
    nothing
end

function step!(env::EvidenceObservationWrapper{T, S, A}, a::A; rng::AbstractRNG=Random.GLOBAL_RNG) where {T, S, A}
    step!(env.env, a; rng=rng)
    set_evidence!(env, false, action(env.env), reward(env.env), state(env.env))
    nothing
end

function evidence_state_space(wrapped_env::AbstractMDP{S, A}, T) where {S, A}
    flag_low, flag_high = 0, 1
    if A == Int
        action_low, action_high = zeros(size(action_space(wrapped_env), 1)), ones(size(action_space(wrapped_env), 1)) # one-hot encoding
    else
        action_low, action_high = action_space(wrapped_env).lows, action_space(wrapped_env).highs
    end
    reward_low, reward_high = -Inf, Inf
    if S == Int
        state_low, state_high = zeros(size(state_space(wrapped_env), 1)), ones(size(state_space(wrapped_env), 1)) # one-hot encoding
    else
        state_low, state_high = state_space(wrapped_env).lows, state_space(wrapped_env).highs
    end
    lows = convert(Vector{T}, vcat(action_low, reward_low, flag_low, state_low))
    highs = convert(Vector{T}, vcat(action_high, reward_high, flag_high, state_high))
    return VectorSpace{T}(lows, highs)
end

@inline state_space(env::EvidenceObservationWrapper) = env.𝕊
@inline action_space(env::EvidenceObservationWrapper) = action_space(env.env)
@inline action_meaning(env::EvidenceObservationWrapper, a) = action_meaning(env.env, a)
@inline action(env::EvidenceObservationWrapper) = action(env.env)
@inline reward(env::EvidenceObservationWrapper) = reward(env.env)
@inline in_absorbing_state(env::EvidenceObservationWrapper) = in_absorbing_state(env.env)
@inline truncated(env::EvidenceObservationWrapper) = truncated(env.env)
@inline visualize(env::EvidenceObservationWrapper, args...; kwargs...) = visualize(env.env, args...; kwargs...)





# """
#     TimeLimitWrapper(env::AbstractMDP, time_limit::Int)

# A wrapper that _truncates_ an episode after a fixed number of steps. Note that this does not mean that the MDP transitions to an absorbing state after the time limit is reached. At the end of the time limit, `in_absorbing_state` will return false and `truncated` will return true. `in_absorbing_state` _could_ return true if the underlying MDP transitions to an absorbing state coincidently with the time limit being reached.
# """
# mutable struct TimeLimitWrapper{S, A} <: AbstractMDP{S, A}
#     const env::AbstractMDP{S, A}
#     const time_limit::Int
#     t::Int
#     function TimeLimitWrapper(env::AbstractMDP{S, A}, time_limit::Int) where {S, A}
#         return new{S, A}(env, time_limit, 0)
#     end
# end

# @inline state_space(env::TimeLimitWrapper) = state_space(env.env)
# @inline action_space(env::TimeLimitWrapper) = action_space(env.env)
# @inline action_meaning(env::TimeLimitWrapper, a) = action_meaning(env.env, a)
# @inline action_meanings(env::TimeLimitWrapper) = action_meanings(env.env)

# @inline start_state_support(env::TimeLimitWrapper) = start_state_support(env.env)
# @inline start_state_probability(env::TimeLimitWrapper, s) = start_state_probability(env.env, s)
# @inline start_state_distribution(env::TimeLimitWrapper, support) = start_state_distribution(env.env, support)
# @inline transition_support(env::TimeLimitWrapper, s, a) = transition_support(env.env, s, a)
# @inline transition_probability(env::TimeLimitWrapper, s, a, s′) = transition_probability(env.env, s, a, s′)
# @inline transition_distribution(env::TimeLimitWrapper, s, a, support) = transition_distribution(env.env, s, a, support)
# @inline reward(env::TimeLimitWrapper, s, a, s′) = reward(env.env, s, a, s′)
# @inline is_absorbing(env::TimeLimitWrapper, s) = is_absorbing(env.env, s)
# @inline visualize(env::TimeLimitWrapper, s; kwargs...) = visualize(env.env, s; kwargs...)

# @inline state(env::TimeLimitWrapper) = state(env.env)
# @inline action(env::TimeLimitWrapper) = action(env.env)
# @inline reward(env::TimeLimitWrapper) = reward(env.env)
# @inline in_absorbing_state(env::TimeLimitWrapper) = in_absorbing_state(env.env)
# @inline visualize(env::TimeLimitWrapper; kwargs...) = visualize(env.env; kwargs...)
# truncated(env::TimeLimitWrapper) = env.t >= env.time_limit

# function factory_reset!(env::TimeLimitWrapper)
#     factory_reset!(env.env)
#     env.t = 0
#     nothing
# end

# function reset!(env::TimeLimitWrapper; rng::AbstractRNG=Random.GLOBAL_RNG)
#     reset!(env.env; rng=rng)
#     env.t = 0
#     nothing
# end

# function step!(env::TimeLimitWrapper{S, A}, a::A; rng::AbstractRNG=Random.GLOBAL_RNG) where {S, A}
#     step!(env.env, a; rng=rng)
#     env.t += 1
#     nothing
# end


"""
    TimeLimitWrapper(env::AbstractMDP, time_limit::Int)

A wrapper that _truncates_ an episode after a fixed number of steps. Note that this does not mean that the MDP transitions to an absorbing state after the time limit is reached. At the end of the time limit, `in_absorbing_state` will return false and `truncated` will return true. `in_absorbing_state` _could_ return true if the underlying MDP transitions to an absorbing state coincidently with the time limit being reached.
"""
mutable struct TimeLimitWrapper{S, A} <: AbstractWrapper{S, A}
    const env::AbstractMDP{S, A}
    const time_limit::Int
    t::Int
    function TimeLimitWrapper(env::AbstractMDP{S, A}, time_limit::Int) where {S, A}
        return new{S, A}(env, time_limit, 0)
    end
end

truncated(env::TimeLimitWrapper) = env.t >= env.time_limit

function factory_reset!(env::TimeLimitWrapper)
    factory_reset!(env.env)
    env.t = 0
    nothing
end

function reset!(env::TimeLimitWrapper; rng::AbstractRNG=Random.GLOBAL_RNG)
    reset!(env.env; rng=rng)
    env.t = 0
    nothing
end

function step!(env::TimeLimitWrapper{S, A}, a::A; rng::AbstractRNG=Random.GLOBAL_RNG) where {S, A}
    step!(env.env, a; rng=rng)
    env.t += 1
    nothing
end
