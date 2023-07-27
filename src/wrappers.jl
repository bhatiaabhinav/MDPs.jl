export OneHotStateReprWrapper, FrameStackWrapper, NormalizeWrapper, EvidenceObservationWrapper, TimeLimitWrapper, Uint8ToFloatWrapper, ReshapeObservationWrapper, FlattenObservationWrapper, FireResetWrapper, VideoRecorderWrapper, ActionRepeatWrapper
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
    FrameStackWrapper(env::AbstractMDP{Array{T, N}, A}, k::Int=4) where {T, N, A}

Wrapper that stacks the last `k` observations of an MDP's state space into a single array. The state space of the wrapped MDP is a `TensorSpace{T, N}`. Each element of the new space is of shape (d1, d2, .., dN * k) where (d1, d2, .., dN) is the shape of the states in the wrapped MDP.
"""
struct FrameStackWrapper{T, N, A} <: AbstractWrapper{Array{T, N}, A}
    env::AbstractMDP{Array{T, N}, A}
    𝕊::TensorSpace{T, N}
    state::Array{T, N}
    function FrameStackWrapper(env::AbstractMDP{Array{T, N}, A}, k::Int=4) where {T, N, A}
        @assert k > 0 "k must be positive"
        if k < 2
            @warn "For framestack=1, you should simply use the original environment instead of FrameStackWrapper"
        end
        sspace::TensorSpace{T, N} = state_space(env)
        lows = repeat(sspace.lows, outer=(ones(Int, N-1)..., k))
        highs = repeat(sspace.highs, outer=(ones(Int, N-1)..., k))
        sspace = TensorSpace{T, N}(lows, highs)
        s = zeros(T, size(sspace)[1:N])
        return new{T, N, A}(env, sspace, s)
    end
end

function factory_reset!(env::FrameStackWrapper{T, N, A}) where {T, N, A}
    factory_reset!(env.env)
    env.state .= zeros(T, size(state_space(env))[1:N])
end

@inline state_space(env::FrameStackWrapper) = env.𝕊
@inline state(env::FrameStackWrapper) = env.state

function reset!(env::FrameStackWrapper{T, N, A}; rng::AbstractRNG=Random.GLOBAL_RNG) where {T, N, A}
    reset!(env.env; rng=rng)
    dN = size(state_space(env.env), N)
    k = size(state_space(env), N) ÷ dN
    if k == 1
        env.state .= state(env.env)
    else
        selectdim(env.state, N, 1:(k-1)*dN) .= 0
        selectdim(env.state, N, (k-1)*dN+1:k*dN) .= state(env.env)
    end
    nothing
end

function step!(env::FrameStackWrapper{T, N, A}, a::A; rng::AbstractRNG=Random.GLOBAL_RNG) where {T, N, A}
    step!(env.env, a; rng=rng)
    
    dN = size(state_space(env.env), N)
    k = size(state_space(env), N) ÷ dN
    if k == 1
        env.state .= state(env.env)
    else
        selectdim(env.state, N, 1:(k-1)*dN) .= selectdim(env.state, N, dN+1:k*dN)
        selectdim(env.state, N, (k-1)*dN+1:k*dN) .= state(env.env)
    end
    nothing
end


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


"""
    Uint8ToFloatWrapper{T}(env::AbstractMDP{Array{UInt8, N}, A}) where {T<:AbstractFloat, N, A}

A wrapper that converts the state space from `Array{UInt8, N}` to `Array{T, N}` where `T <: AbstractFloat`. The values are by dividing by `255` (the maximum value of a `UInt8`).
"""
struct Uint8ToFloatWrapper{T<:AbstractFloat, N, A} <: AbstractWrapper{Array{T, N}, A}
    env::AbstractMDP{Array{UInt8, N}, A}
    𝕊::TensorSpace{T, N}
    state::Array{T, N}
    function Uint8ToFloatWrapper{T}(env::AbstractMDP{Array{UInt8, N}, A}) where {T<:AbstractFloat, N, A}
        sspace = state_space(env)
        lows = convert(Array{T, N}, sspace.lows) / 255
        highs = convert(Array{T, N}, sspace.highs) / 255
        𝕊 = TensorSpace{T, N}(lows, highs)
        s = convert(Array{T, N}, state(env)) / 255
        return new{T, N, A}(env, 𝕊, s)
    end
end

@inline state_space(env::Uint8ToFloatWrapper) = env.𝕊
@inline state(env::Uint8ToFloatWrapper) = env.state

function factory_reset!(env::Uint8ToFloatWrapper{T, N, A}) where {T<:AbstractFloat, N, A}
    factory_reset!(env.env)
    env.state .= convert(Array{T, N}, state(env.env)) / 255
    nothing
end

function reset!(env::Uint8ToFloatWrapper{T, N, A}; rng::AbstractRNG=Random.GLOBAL_RNG) where {T<:AbstractFloat, N, A}
    reset!(env.env; rng=rng)
    env.state .= convert(Array{T, N}, state(env.env)) / 255
    nothing
end

function step!(env::Uint8ToFloatWrapper{T, N, A}, a::A; rng::AbstractRNG=Random.GLOBAL_RNG) where {T<:AbstractFloat, N, A}
    step!(env.env, a; rng=rng)
    env.state .= convert(Array{T, N}, state(env.env)) / 255
    nothing
end


"""
    ReshapeObservationWrapper(env::AbstractMDP{Array{T, M}, A}, newshape::NTuple{N, Int}) where {T, N, M, A}

A wrapper that reshapes the state space from `Array{T, M}` to `Array{T, N}`.
"""
struct ReshapeObservationWrapper{T, N, M, A} <: AbstractWrapper{Array{T, N}, A}
    env::AbstractMDP{Array{T, M}, A}
    newshape::NTuple{N, Int}
    function ReshapeObservationWrapper(env::AbstractMDP{Array{T, M}, A}, newshape::NTuple{N, Int}) where {T, N, M, A}
        return new{T, N, M, A}(env, newshape)
    end
end

function state_space(env::ReshapeObservationWrapper{T, N, M, A})::TensorSpace{T, N} where {T, N, M, A}
    sspace::TensorSpace{T, M} = state_space(env.env)
    lows, highs = sspace.lows, sspace.highs
    return TensorSpace{T, N}(reshape(lows, env.newshape), reshape(highs, env.newshape))
end

@inline state(env::ReshapeObservationWrapper) = reshape(state(env.env), env.newshape)


"""
    FlattenObservationWrapper(env::AbstractMDP{Array{T, N}, A}) where {T, N, A}

A wrapper that reshapes the state space from `Array{T, N}` to `Vector{T}`.
"""
FlattenObservationWrapper(env) = ReshapeObservationWrapper(env, (prod(size(state_space(env))[1:end-1]), ))


"""
    FireResetWrapper(env::AbstractMDP{S, A}) where {S, A}

A wrapper that performs the following steps on `reset!`: (1) reset the environment, (2) take a random action, (3) repeat until the game starts i.e., the state changes. This is useful for games like Atari where the game starts only after a specific button is pressed.
"""
struct FireResetWrapper{S, A} <: AbstractWrapper{S, A}
    env::AbstractMDP{S, A}
    function FireResetWrapper(env::AbstractMDP{S, A}) where {S, A}
        return new{S, A}(env)
    end
end

function reset!(env::FireResetWrapper; rng::AbstractRNG=Random.GLOBAL_RNG)
    reset!(env.env; rng=rng)
    s0 = state(env.env) |> copy
    while state(env.env) == s0
        step!(env.env, rand(action_space(env.env)); rng=rng)  # Try some action to start the game
        if in_absorbing_state(env.env) || truncated(env.env)
            reset!(env.env)
            s0 = state(env.env) |> copy
        end
    end
    nothing
end


"""
    VideoRecorderWrapper(env::AbstractMDP{S, A}, save_to::Union{String, Nothing}, n=1; format="mp4", fps=30, kwargs...) where {S, A}

Wrapper that records a video of the _wrapped_ environment every `n` episodes. If `save_to` is a string, the video is saved in `save_to` directory. If the directory already exists, it will be deleted and overwritten. The video format can be either `mp4` or `gif`. The video is recorded at `fps` frames per second. If `save_to` is a vector, then raw data of each recording will be pushed to the vector. The raw data is a vector of `Matrix{RGB{N0f8}}` frames. The frames can be converted to a video using `FileIO.save("video.mp4", frames, framerate=30)` or `FileIO.save("video.gif", cat(frames..., dims=3), fps=30)`. `kwargs` are passed to `visualize` when recording the video.

The difference between `VideoRecorderWrapper` and `VideoRecorderHook` is that the former records the video of the _wrapped_ environment, while the latter records the video of the environment being `interact`ed with. For example, if the interaction environment involves repeating an action k times, then the latter will record frames every k steps, while the former, if wrapped over the original environment, will record frames every step. As another use case, this wrapper, in conjunction with `EmpiricalPolicyEvaluationHook`, can be used to record videos of the test environment being played by the policy being evaluated.
"""
mutable struct VideoRecorderWrapper{S, A} <: AbstractWrapper{S, A}
    const env::AbstractMDP{S, A}
    const save_to::Union{String, Vector}
    const format::String
    const n::Int
    const fps::Int
    const viz_kwargs::Dict{Symbol, Any}

    episode_counter::Int
    steps_counter::Int
    episode_return::Float64
    frames::Vector{Matrix{RGB{Colors.N0f8}}}

    function VideoRecorderWrapper(env::AbstractEnv{S, A}, save_to::Union{String, Vector}, n=1; format="mp4", fps=30, kwargs...) where {S, A}
        if save_to isa String
            @assert format ∈ ["mp4", "gif"] "Only mp4 or gif are supported"
            rm(save_to, recursive=true, force=true)
            mkpath(save_to)
        end
        return new{S, A}(env, save_to, format, n, fps, kwargs, 0, 0, 0, [])
    end
end

function factory_reset!(env::VideoRecorderWrapper)
    factory_reset!(env.env)
    env.episode_counter = 0
    env.steps_counter = 0
    env.episode_return = 0
    empty!(env.frames)
    nothing
end

function reset!(env::VideoRecorderWrapper; rng::AbstractRNG=Random.GLOBAL_RNG)
    reset!(env.env; rng=rng)
    env.episode_counter += 1
    env.episode_return = 0
    empty!(env.frames)
    push!(env.frames, convert(Matrix{RGB{Colors.N0f8}}, visualize(env.env; env.viz_kwargs...)))
    nothing
end

function step!(env::VideoRecorderWrapper{S, A}, a::A; rng::AbstractRNG=Random.GLOBAL_RNG) where {S, A}
    step!(env.env, a; rng=rng)
    env.steps_counter += 1
    env.episode_return += reward(env.env)
    if env.episode_counter % env.n == 0
        viz = convert(Matrix{RGB{Colors.N0f8}}, visualize(env.env; env.viz_kwargs...))
        push!(env.frames, viz)

        if in_absorbing_state(env.env) || truncated(env.env)
            if env.save_to isa String
                fn = "$(env.save_to)/ep-$(env.episode_counter)-steps-$(env.steps_counter)-return-$(env.episode_return).$(env.format)"
                if env.format == "mp4"
                    save(fn, env.frames, framerate=env.fps)
                elseif env.format == "gif"
                    save(fn, cat(env.frames..., dims=3), fps=env.fps)
                end
            else
                push!(env.save_to, env.frames)
            end
        end
    end
    nothing
end


"""
    ActionRepeatWrapper(env::AbstractEnv, k::Int=4, agg_fn=sum)

Repeat action `k` times and aggregate reward using `agg_fn` (`sum` by default). If the wrapped environment transitions to an absorbing state or is truncated in the process, the wrapper episode ends with the accumulated reward up to that point. Moreover, in such cases, the wrapped environment does not reset automatically, so the user should call `reset!` manually.
The `agg_fn` can be any function that takes a vector of rewards and returns a scalar e.g., `sum`, `mean`, `median`, `maximum`, `minimum`, etc.
"""
mutable struct ActionRepeatWrapper{S, A} <: AbstractWrapper{S, A}
    const env::AbstractEnv{S, A}
    const k::Int
    const agg_fn
    const rewards::Vector{Float64}
    reward::Float64
    function ActionRepeatWrapper(env::AbstractEnv{S, A}, k::Int=4, agg_fn=+) where {S, A}
        return new{S, A}(env, k, agg_fn, Float64[], 0.0)
    end
end


function factory_reset!(env::ActionRepeatWrapper)
    factory_reset!(env.env)
    env.reward = 0.0
    nothing
end

function reset!(env::ActionRepeatWrapper; rng::AbstractRNG=Random.GLOBAL_RNG)
    reset!(env.env; rng=rng)
    empty!(env.rewards)
    env.reward = 0.0
    nothing
end

function step!(env::ActionRepeatWrapper{S, A}, a::A; rng::AbstractRNG=Random.GLOBAL_RNG) where {S, A}
    empty!(env.rewards)
    for i in 1:env.k
        step!(env.env, a; rng=rng)
        r = reward(env.env)
        push!(env.rewards, r)
        if in_absorbing_state(env.env) || truncated(env.env)
            break
        end
    end
    env.reward = env.agg_fn(env.rewards)
    nothing
end

@inline reward(env::ActionRepeatWrapper) = env.reward