export VecNormalizeWrapper

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

function reset!(env::VecNormalizeWrapper{T, N, A}, reset_all::Bool; rng::AbstractRNG=Random.GLOBAL_RNG) where {T, N, A}
    need_reset = in_absorbing_state(env.env) .|| truncated(env.env)
    for i in 1:num_envs(env)
        if need_reset[i] || reset_all
            env.rets[i] = 0
            env.update_stats && push!(env.obs_rmv, selectdim(state(env.env), N+1, i))
        end
    end
    reset!(env.env, reset_all; rng=rng)
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