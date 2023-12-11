export AbstractVecEnv, VecEnv, get_envs, num_envs, AbstractVecWrapper

abstract type AbstractVecEnv{S, A} end

"""
    get_envs(v::AbstractVecEnv{S, A})

    Should return an iterator over the environments in the vectorized environment. May not be possible to implement for all vectorized environments.
"""
function get_envs(v::AbstractVecEnv{S, A}) where {S, A}
    v.envs
end

function num_envs(v::AbstractVecEnv{S, A}) where {S, A}
    length(get_envs(v))
end

@inline Base.length(v::AbstractVecEnv) = num_envs(v)

function state_space(v::AbstractVecEnv{S, A})::AbstractSpace{S} where {S, A}
    state_space(first(get_envs(v)))
end
function action_space(v::AbstractVecEnv{S, A})::AbstractSpace{A} where {S, A}
    action_space(first(get_envs(v)))
end
function action_meaning(v::AbstractVecEnv{S, A}, a::A)::String where {S, A}
    action_meaning(first(get_envs(v)), a)
end
function action_meanings(v::AbstractVecEnv{S, A})::Vector{String} where {S, A}
    action_meanings(first(get_envs(v)))
end

function state(v::AbstractVecEnv{Array{Tₛ, N}, A})::AbstractArray{Tₛ} where {Tₛ, N, A}
    if N == 1
        return mapfoldl(state, hcat, get_envs(v))  # hcat is faster than cat(x, y, dims=2)
    else
        return mapfoldl(state, (x, y) -> cat(x, y, dims=N+1), get_envs(v))
    end
end
function state(v::AbstractVecEnv{S, A})::Vector{S} where {S, A}
    mapfoldl(state, vcat, get_envs(v))
end

function action(v::AbstractVecEnv{S, Array{Tₐ, N}})::AbstractArray{Tₐ} where {S, Tₐ, N}
    if N == 1
        return mapfoldl(action, hcat, get_envs(v))  # hcat is faster than cat(x, y, dims=2)
    else
        return mapfoldl(action, (x, y) -> cat(x, y, dims=N+1), get_envs(v))
    end
end
function action(v::AbstractVecEnv{S, A})::Vector{A} where {S, A}
    mapfoldl(action, vcat, get_envs(v))
end

function reward(v::AbstractVecEnv{S, A})::Vector{Float64} where {S, A}
    mapfoldl(reward, vcat, get_envs(v))
end

function in_absorbing_state(v::AbstractVecEnv{S, A})::Vector{Bool} where {S, A}
    mapfoldl(in_absorbing_state, vcat, get_envs(v))
end

function truncated(v::AbstractVecEnv{S, A})::Vector{Bool} where {S, A}
    mapfoldl(truncated, vcat, get_envs(v))
end

function info(v::AbstractVecEnv{S, A})::Vector{Dict{Symbol, Any}} where {S, A}
    mapfoldl(info, vcat, get_envs(v))
end


function factory_reset!(v::AbstractVecEnv{S, A}) where {S, A}
    nothing
end

function reset!(v::AbstractVecEnv{S, A}, reset_all::Bool=true; rng::AbstractRNG=Random.GLOBAL_RNG) where {S, A}
    error("Not implemented")
end

function step!(v::AbstractVecEnv{S, A}, action; rng::AbstractRNG=Random.GLOBAL_RNG) where {S, A}
    error("Not implemented")
end








abstract type AbstractVecWrapper{S, A} <: AbstractVecEnv{S, A} end

function get_envs(v::AbstractVecWrapper)
    error("get_envs not implemented for $(typeof(v))")
end
num_envs(v::AbstractVecWrapper) = num_envs(unwrapped(v))

function unwrapped(v::AbstractVecWrapper, deep::Bool=false)::AbstractVecEnv
    if deep
        return unwrapped(v.env, deep)
    else
        return v.env
    end
end
function unwrapped(v::AbstractVecEnv, deep::Bool=false)::AbstractVecEnv
    return v
end

@inline state_space(v::AbstractVecWrapper) = state_space(unwrapped(v))
@inline action_space(v::AbstractVecWrapper) = action_space(unwrapped(v))
@inline action_meaning(v::AbstractVecWrapper{S, A}, a::A) where {S, A} = action_meaning(unwrapped(v), a)
@inline action_meanings(v::AbstractVecWrapper) = action_meanings(unwrapped(v))

@inline state(v::AbstractVecWrapper) = state(unwrapped(v))
@inline action(v::AbstractVecWrapper) = action(unwrapped(v))
@inline reward(v::AbstractVecWrapper) = reward(unwrapped(v))
@inline factory_reset!(v::AbstractVecWrapper) = factory_reset!(unwrapped(v))
@inline reset!(v::AbstractVecWrapper, reset_all::Bool=true; rng::AbstractRNG=Random.GLOBAL_RNG) = reset!(unwrapped(v), reset_all; rng=rng)
@inline step!(v::AbstractVecWrapper, a; rng::AbstractRNG=Random.GLOBAL_RNG) = step!(unwrapped(v), a; rng=rng)
@inline in_absorbing_state(v::AbstractVecWrapper) = in_absorbing_state(unwrapped(v))
@inline truncated(v::AbstractVecWrapper) = truncated(unwrapped(v))
@inline info(v::AbstractVecWrapper) = info(unwrapped(v))



















struct VecEnv{S, A, E <: AbstractEnv{S, A}} <: AbstractVecEnv{S, A}
    envs::Vector{E}
    multithreaded::Bool
    function VecEnv(envs::Vector{E}, multithreaded::Bool=false) where {S, A, E <: AbstractEnv{S, A}}
        new{S, A, E}(envs, multithreaded)
    end
end

function factory_reset!(v::VecEnv{S, A, E}) where {S, A, E <: AbstractEnv{S, A}}
    if v.multithreaded
        Threads.@threads for env in v.envs
            factory_reset!(env)
        end
    else
        foreach(factory_reset!, v.envs)
    end
end

function reset!(v::VecEnv{S, A, E}, reset_all::Bool=true; rng::AbstractRNG=Random.GLOBAL_RNG) where {S, A, E <: AbstractEnv{S, A}}
    if v.multithreaded
        Threads.@threads for env in v.envs
            if reset_all || (in_absorbing_state(env) || truncated(env))
                reset!(env, rng=rng)
            end
        end
    else
        for env in v.envs
            if reset_all || (in_absorbing_state(env) || truncated(env))
                reset!(env, rng=rng)
            end
        end
    end
end

function step!(v::VecEnv{S, A, E}, action::AbstractVector{A}; rng::AbstractRNG=Random.GLOBAL_RNG) where {S, A, E <: AbstractEnv{S, A}}
    if v.multithreaded
        Threads.@threads for i in 1:length(v.envs)
            step!(v.envs[i], action[i], rng=rng)
        end
    else
        for (env, a) in zip(v.envs, action)
            step!(env, a, rng=rng)
        end
    end
end

function step!(v::VecEnv{S, Array{Tₐ, N}, E}, action::AbstractArray{Tₐ, M}; rng::AbstractRNG=Random.GLOBAL_RNG) where {S, Tₐ, N, M, E <: AbstractEnv{S, Array{Tₐ, N}}}
    @assert M == N + 1
    if v.multithreaded
        Threads.@threads for i in 1:length(v.envs)
            a = selectdim(action, M, i) |> copy
            step!(v.envs[i], a, rng=rng)
        end
    else
        for (i, env) in enumerate(v.envs)
            a = selectdim(action, M, i) |> copy
            step!(env, a, rng=rng)
        end
    end
end