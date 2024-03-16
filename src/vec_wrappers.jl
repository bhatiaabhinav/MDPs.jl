export VecNormalizeWrapper, VecActionRepeatWrapper, VecFrameSkipWrapper, VecReshapeObservationWrapper, VecUint8ToFloatWrapper, VecEvidenceObservationWrapper, VecFlattenObservationWrapper

Base.@kwdef mutable struct VecNormalizeWrapper{T, N, A} <: AbstractVecWrapper{Array{T, N}, A}
    const env::AbstractVecEnv{Array{T, N}, A}
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
    rets::Vector{Float64} = zeros(num_envs(env))
end

function VecNormalizeWrapper(env::AbstractVecEnv{Array{T, N}, A}; kwargs...) where {T, N, A}
    VecNormalizeWrapper{T, N, A}(; env=env, kwargs...)
end

function state_space(env::VecNormalizeWrapper)
    ss = deepcopy(state_space(env.env))
    ss.lows .= -env.clip_obs
    ss.highs .= env.clip_obs
    return ss
end


function state(env::VecNormalizeWrapper{T, N, A})::AbstractArray{T} where {T, N, A}
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

function reward(env::VecNormalizeWrapper)::Vector{Float64}
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
        return clamp.(reward(env.env) / (divisor + env.ϵ), -env.clip_reward, env.clip_reward)
    else
        return reward(env.env)
    end
end

function MDPs.reset!(env::VecNormalizeWrapper{T, N, A}, reset_all::Bool=true; rng::AbstractRNG=Random.GLOBAL_RNG) where {T, N, A}
    env_in_absorbing_state = in_absorbing_state(env.env)
    env_truncated = truncated(env.env)
    need_reset = env_in_absorbing_state .|| env_truncated
    reset!(env.env, reset_all; rng=rng)
    if env.update_stats
        env_state = state(env.env)
        for i in 1:num_envs(env)
            if need_reset[i] || reset_all
                env.rets[i] = 0
                push!(env.obs_rmv, selectdim(env_state, N+1, i))
            end
        end
    end
    nothing
end

function step!(env::VecNormalizeWrapper{T, N, A}, a; rng::AbstractRNG=Random.GLOBAL_RNG) where {T, N, A}
    step!(env.env, a; rng=rng)
    rs::Vector{Float64} = reward(env.env)
    env.rets = env.γ * env.rets + rs
    if env.update_stats
        for i in 1:num_envs(env)
            push!(env.rew_rmv, rs[i])
            push!(env.ret_rmv, env.rets[i])
            push!(env.obs_rmv, selectdim(state(env.env), N+1, i))
        end
    end
    nothing
end


struct VecReshapeObservationWrapper{T, N, M, A} <: AbstractVecWrapper{Array{T, N}, A}
    env::AbstractVecEnv{Array{T, M}, A}
    newshape::NTuple{N, Int}
    function VecReshapeObservationWrapper(env::AbstractVecEnv{Array{T, M}, A}, newshape::NTuple{N, Int}) where {T, N, M, A}
        new{T, N, M, A}(env, newshape)
    end
end

function state_space(env::VecReshapeObservationWrapper{T, N, M, A})::TensorSpace{T, N} where {T, N, M, A}
    sspace::TensorSpace{T, M} = state_space(env.env)
    lows, highs = sspace.lows, sspace.highs
    return TensorSpace{T, N}(reshape(lows, env.newshape), reshape(highs, env.newshape))
end

function state(env::VecReshapeObservationWrapper{T, N, M, A}) where {T, N, M, A}
    # curstate = state(env.env)
    # println((size(curstate), env.newshape, num_envs(env.env)))
    return reshape(state(env.env), env.newshape..., num_envs(env.env))
end


VecFlattenObservationWrapper(env) = VecReshapeObservationWrapper(env, (prod(size(state_space(env))[1:end-1]), ))



struct VecUint8ToFloatWrapper{T<:AbstractFloat, N, M, A} <: AbstractVecWrapper{Array{T, N}, A}
    env::AbstractVecEnv{Array{UInt8, N}, A}
    𝕊::TensorSpace{T, N}
    state::Array{T, M}
    function VecUint8ToFloatWrapper{T}(env::AbstractVecEnv{Array{UInt8, N}, A}) where {T<:AbstractFloat, N, A}
        sspace = state_space(env)
        lows = convert(Array{T, N}, sspace.lows) / 255
        highs = convert(Array{T, N}, sspace.highs) / 255
        𝕊 = TensorSpace{T, N}(lows, highs)
        # println("N ", N)
        s = convert(Array{T, N+1}, state(env)) / 255
        new{T, N, N+1, A}(env, 𝕊, s)
    end
end

@inline state_space(env::VecUint8ToFloatWrapper) = env.𝕊
@inline state(env::VecUint8ToFloatWrapper) = env.state

function factory_reset!(env::VecUint8ToFloatWrapper{T, N, M, A}) where {T, N, M, A}
    factory_reset!(env.env)
    env.state .= convert(Array{T, N+1}, state(env.env)) / 255
    nothing
end

function reset!(env::VecUint8ToFloatWrapper{T, N, M, A}, reset_all::Bool=true; rng::AbstractRNG=Random.GLOBAL_RNG) where {T, N, M, A}
    reset!(env.env, reset_all; rng=rng)
    env.state .= convert(Array{T, M}, state(env.env)) / 255
    nothing
end

function step!(env::VecUint8ToFloatWrapper{T, N, M, A}, a; rng::AbstractRNG=Random.GLOBAL_RNG) where {T, N, M, A}
    step!(env.env, a; rng=rng)
    env.state .= convert(Array{T, N+1}, state(env.env)) / 255
    nothing
end





struct VecActionRepeatWrapper{S, A} <: AbstractVecWrapper{S, A}
    env::AbstractVecEnv{S, A}
    k::Int
    agg_fn
    rewards::Vector{Vector{Float64}}
    reward::Vector{Float64}
    function VecActionRepeatWrapper(env::AbstractVecEnv{S, A}, k::Int=4, agg_fn=sum) where {S, A}
        new{S, A}(env, k, agg_fn, [Float64[] for _ in 1:num_envs(env)], Vector{Float64}(undef, num_envs(env)))
    end
end

const VecFrameSkipWrapper = VecActionRepeatWrapper

function factory_reset!(env::VecActionRepeatWrapper{S, A}) where {S, A}
    factory_reset!(env.env)
    foreach(r -> empty!(r), env.rewards)
    fill!(env.reward, 0)
    nothing
end

function reset!(env::VecActionRepeatWrapper{S, A}, reset_all::Bool=true; rng::AbstractRNG=Random.GLOBAL_RNG) where {S, A}
    if reset_all
        foreach(r -> empty!(r), env.rewards)
        fill!(env.reward, 0)
    else
        need_reset = in_absorbing_state(env.env) .|| truncated(env.env)
        for i in 1:num_envs(env)
            if need_reset[i]
                empty!(env.rewards[i])
                env.reward[i] = 0
            end
        end
    end
    reset!(env.env, reset_all; rng=rng)
    nothing
end

function step!(env::VecActionRepeatWrapper{S, A}, a; rng::AbstractRNG=Random.GLOBAL_RNG) where {S, A}
    foreach(r -> empty!(r), env.rewards)
    for t in 1:env.k
        step!(env.env, a; rng=rng)
        r = reward(env.env)
        for i in 1:num_envs(env)
            push!(env.rewards[i], r[i])
        end
        if any(in_absorbing_state(env.env)) || any(truncated(env.env))
            break
        end
    end
    for i in 1:num_envs(env)
        env.reward[i] = env.agg_fn(env.rewards[i])
    end
    nothing
end

@inline reward(env::VecActionRepeatWrapper) = env.reward






struct VecEvidenceObservationWrapper{T<:AbstractFloat, S, A} <: AbstractVecWrapper{Vector{T}, A}
    env::AbstractVecEnv{S, A}
    evidence::Matrix{T}
    𝕊::VectorSpace{T}
    function VecEvidenceObservationWrapper{T}(env::AbstractVecEnv{S, A}) where {T<:AbstractFloat, S, A}
        @assert S == Int || S <: Vector
        @assert A == Int || A <: Vector
        sspace, aspace = state_space(env), action_space(env)
        m, n = size(sspace, 1), size(aspace, 1)
        return new{T, S, A}(env, Matrix{T}(undef, 1+n+1+m, num_envs(env)), evidence_state_space(sspace, aspace, T))
    end
end

@inline state_space(env::VecEvidenceObservationWrapper) = env.𝕊
@inline state(env::VecEvidenceObservationWrapper) = env.evidence

function reset!(env::VecEvidenceObservationWrapper{T, S, A}, reset_all::Bool=true; rng::AbstractRNG=Random.GLOBAL_RNG) where {T, S, A}
    sspace, aspace = state_space(env.env), action_space(env.env)
    need_reset = in_absorbing_state(env.env) .|| truncated(env.env)
    reset!(env.env, reset_all; rng=rng)
    s = state(env.env)
    a = action(env.env)
    r = reward(env.env)
    for i in 1:num_envs(env)
        s_i::S = S == Int ? s[i] : s[:, i]
        a_i::A = A == Int ? a[i] : a[:, i]
        r_i = r[i]
        new_episode_flag_i = need_reset[i] || reset_all
        evidence_i = @view env.evidence[:, i]  # very important to take the view since it will be modified in place in the next line
        set_evidence!(evidence_i, new_episode_flag_i, a_i, r_i, s_i, sspace, aspace)
    end
    nothing
end

function step!(env::VecEvidenceObservationWrapper{T, S, A}, a; rng::AbstractRNG=Random.GLOBAL_RNG) where {T, S, A}
    step!(env.env, a; rng=rng)
    sspace, aspace = state_space(env.env), action_space(env.env)
    s = state(env.env)
    a = action(env.env)
    r = reward(env.env)
    for i in 1:num_envs(env)
        s_i::S = S == Int ? s[i] : s[:, i]
        a_i::A = A == Int ? a[i] : a[:, i]
        r_i = r[i]
        evidence_i = @view env.evidence[:, i]  # very important to take the view since it will be modified in place in the next line
        set_evidence!(evidence_i, false, a_i, r_i, s_i, sspace, aspace)
    end
    nothing
end