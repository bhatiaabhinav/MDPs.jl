export OneHotStateReprWrapper, FrameStackWrapper

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

in_absorbing_state(env::OneHotStateReprWrapper) = in_absorbing_state(env.env)

visualize(env::OneHotStateReprWrapper) = visualize(env.env)

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

@inline action_space(env::FrameStackWrapper) = action_space(env.env)
@inline state_space(env::FrameStackWrapper) = env.ss
@inline action_meaning(env::FrameStackWrapper, a::Int) = action_meaning(env.env, a)


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

in_absorbing_state(env::FrameStackWrapper) = in_absorbing_state(env.env)

visualize(env::FrameStackWrapper) = visualize(env.env)
