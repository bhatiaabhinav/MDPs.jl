using StatsBase

export AbstractMDP, state_space, action_space, action_meaning, action_meanings, start_state_support, start_state_probability, start_state_distribution, transition_support, transition_probability, transition_distribution, reward, is_absorbing, truncated, state, action, reward, reset!, factory_reset!, step!, in_absorbing_state, visualize
export AbstractWrapper, unwrapped

"""
    AbstractMDP{S, A}

Abstract type for Markov Decision Processes (MDPs). The type parameters `S` and `A` are the types of the state and action, respectively.

When defining a concrete MDP, the methods `state_space`, `action_space`, `action_meaning` and `action_meanings` must be implemented.

To fully specify an MDP with known transition and reward models, the methods `start_state_support`, `start_state_probability`, `start_state_distribution`, `transition_support`, `transition_probability`, `transition_distribution`, `reward` and `is_absorbing` must be implemented.

To have the MDP be used in a simulation (reinfocement learning) loop, there are two options:
If the transition and reward models have been implemented, then simply having fields `state`, `action`, and `reward` is sufficient.
Otherwise, the methods `state`, `action`, `reward`, `reset!`, `factory_reset!`, `step!`, `in_absorbing_state`, and `truncated` must be implemented.
"""
abstract type AbstractMDP{S, A} end

# necessary:

"""
    state_space(mdp::AbstractMDP{S, A})::AbstractSpace{S} where {S, A}

Returns the state space of the MDP.
"""
function state_space(mdp::AbstractMDP{S, A})::AbstractSpace{S} where {S, A}
    error("not implemented")
end

"""
    action_space(mdp::AbstractMDP{S, A})::AbstractSpace{A} where {S, A}

Returns the action space of the MDP.
"""
function action_space(mdp::AbstractMDP{S, A})::AbstractSpace{A} where {S, A}
    error("not implemented")
end

"""
    action_meaning(mdp::AbstractMDP{S, A}, a::A)::String where {S, A}

Returns a string describing the action `a`. By default, returns `"action \$a"`.
"""
function action_meaning(::AbstractMDP{S, A}, a::A)::String where {S, A}
    return "action $a"
end

"""
    action_meanings(mdp::AbstractMDP{S, Int})::Vector{String} where {S}

Returns a vector of strings describing the actions when the action space is discrete. By default, returns a vector of `action_meaning` of each action in the action space.
"""
function action_meanings(mdp::AbstractMDP{S, Int})::Vector{String} where {S}
    return map(a -> action_meaning(mdp, a), action_space(mdp))
end


# ------------------------- when the model is known ------------------------------

"""
    start_state_support(mdp::AbstractMDP{S, A}) where {S, A}

Returns a vector (or an iterable) of states that are possible starting states. By default, returns the state space.
"""
function start_state_support(mdp::AbstractMDP{S, A}) where {S, A} # return something finite iterable
    return state_space(mdp)
end

"""
    start_state_probability(mdp::AbstractMDP{S, A}, s::S)::Float64 where {S, A}

Returns the probability of starting in state `s`. Throws an error if not implemented.
"""
function start_state_probability(mdp::AbstractMDP{S, A}, s::S)::Float64 where {S, A}
    error("not implemented: start state probabilities are not known")
end

"""
    start_state_distribution(mdp::AbstractMDP{S, A}, support)::Vector{Float64} where {S,A}

Returns a vector of probabilities of starting in each state in `support`. By default, returns a vector of `start_state_probability(mdp, s)` for each `s` in `support`.
"""
function start_state_distribution(mdp::AbstractMDP{S, A}, support)::Vector{Float64} where {S,A} # probabilities of candidates_s₀
    return map(s₀ -> start_state_probability(mdp, s₀),  support)
end

"""
    transition_support(mdp::AbstractMDP{S, A}, s::S, a::A) where {S, A}

Returns a vector (or an iterable) of states that are possible next states when taking action `a` in state `s`. By default, returns the state space.
"""
function transition_support(mdp::AbstractMDP{S, A}, s::S, a::A) where {S, A}  # return something finite iterable
    return state_space(mdp)
end

"""
    transition_probability(mdp::AbstractMDP{S, A}, s::S, a::A, s′::S)::Float64 where {S, A}

Returns the probability of transitioning to state `s′` when taking action `a` in state `s`. Throws an error if not implemented.
"""
function transition_probability(env::AbstractMDP{S, A}, s::S, a::A, s′::S)::Float64 where {S, A}
    error("not implemented: transition probabilities are not known")
end

"""
    transition_distribution(mdp::AbstractMDP{S, A}, s::S, a::A, support)::Vector{Float64} where {S,A}

Returns a vector of probabilities of transitioning to each state in `support` when taking action `a` in state `s`. By default, returns a vector of `transition_probability(mdp, s, a, s′)` for each `s′` in `support`.
"""
function transition_distribution(mdp::AbstractMDP{S, A}, s::S, a::A, support)::Vector{Float64} where {S,A}  # probabilities of candidates_s′
    return map(s′ -> transition_probability(mdp, s, a, s′),  support)
end

"""
    reward(mdp::AbstractMDP{S, A}, s::S, a::A, s′::S)::Float64 where {S, A}

Returns the *mean* reward of transitioning to state `s′` when taking action `a` in state `s`. Throws an error if not implemented.
"""
function reward(mdp::AbstractMDP{S, A}, s::S, a::A, s′::S)::Float64 where {S, A}
    error("not implemented: reward function is not known")
end

"""
    is_absorbing(mdp::AbstractMDP{S, A}, s::S)::Bool where {S, A}

Returns `true` if state `s` is absorbing (i.e., a goal state). Throws an error if not implemented.
"""
function is_absorbing(mdp::AbstractMDP{S, A}, s::S)::Bool where {S, A}
    error("absorbing (goal) states unknown")
end

"""
    visualize(mdp::AbstractMDP{S, A}, s::S; kwargs...) where {S, A}

Visualize the state `s` of the MDP. Returns a Matrix{RGB24} or Matrix{ARGB32} if the state is an image, otherwise returns a string. Throws an error if not implemented.
"""
function visualize(mdp::AbstractMDP{S, A}, s::S; kwargs...) where {S, A}
    error("visualization not implemented")
end


# ---------------------------------------------------------------------------



# --------------------------------- mdp as an RL environment ---------------


"""
    state(env::AbstractMDP{S, A})::S where {S, A}

Returns the current state of the environment. By default, returns `env.state` assuming that `env` is a mutable struct with a field `state`.
"""
@inline function state(env::AbstractMDP{S, A})::S where {S, A}
    env.state
end

"""
    action(env::AbstractMDP{S, A})::A where {S, A}

Returns the latest action performed in the environment. By default, returns `env.action` assuming that `env` is a mutable struct with a field `action`.
"""
@inline function action(env::AbstractMDP{S, A})::A where {S, A}
    env.action
end

"""
    reward(env::AbstractMDP{S, A})::Float64 where {S, A}

Returns the reward of the latest transition. By default, returns `env.reward` assuming that `env` is a mutable struct with a field `reward`.
"""
@inline function reward(env::AbstractMDP{S, A})::Float64 where {S, A}
    env.reward
end

"""
    factory_reset!(env::AbstractMDP)

Reset any data structures and parameters (e.g., a hidden variables or logs) in the MDP struct, that would otherwise persist between episodes. This method is invoked by `interact` at the beginning of the experiment. By default, does nothing.
"""
function factory_reset!(env::AbstractMDP)
    nothing
end

"""
    reset!(env::AbstractMDP{S, A}; rng::AbstractRNG=Random.GLOBAL_RNG) where {S, A}

Resets the environment to a random initial state. By default, assuming that environment dynamics have been implemented and the `env` is a mutable struct with fields `state` and `action`, this function samples a random initial state from the start state distribution and sets `env.state` to that state and `env.action` to `0` (or a vector of zeros if `A` is not `Int`). A random number generator `rng` is passed to the function to allow for reproducible results. This function returns `nothing`.
"""
function reset!(env::AbstractMDP{S, A}; rng::AbstractRNG=Random.GLOBAL_RNG)::Nothing where {S, A}
    support = collect(start_state_support(env))
    env.state = sample(rng, support, ProbabilityWeights(start_state_distribution(env, support)))
    if A == Int
        env.action = 0
    else
        fill!(env.action, 0)
    end
    nothing
end


"""
    step!(env::AbstractMDP{S, A}, a::A; rng::AbstractRNG=Random.GLOBAL_RNG) where {S, A}

Takes action `a` in the environment and updates the environment state and generates a reward. By default, assuming that environment dynamics have been implemented and the `env` is a mutable struct with fields `state`, `action`, and `reward`, this function samples a random next state from the transition distribution and sets `env.state` to that state and `env.action` to `a`. If the environment is in an absorbing state, the environment state and reward are not updated and a warning is thrown. A random number generator `rng` is passed to the function to allow for reproducible results. This function returns `nothing`.
"""
function step!(env::AbstractMDP{S, A}, a::A; rng::AbstractRNG=Random.GLOBAL_RNG)::Nothing where {S, A}
    @assert a ∈ action_space(env)
    env.action = a
    if in_absorbing_state(env)
        @warn "The environment is in an absorbing state. This `step!` will not do anything. Please call `reset!`."
        env.reward = 0
    else
        s = state(env)
        support = collect(transition_support(env, s, a))
        s′ = sample(rng, support, ProbabilityWeights(transition_distribution(env, s, a, support)))
        r = reward(env, s, a, s′)
        env.state = s′
        env.action = a
        env.reward = r
    end
    nothing
end

"""
    in_absorbing_state(env::AbstractMDP)::Bool

Returns `true` if the environment is in an absorbing state. By default, returns `is_absorbing(env, state(env))` assuming that `env` is a mutable struct with a field `state` and `is_absorbing` is implemented.
"""
function in_absorbing_state(env::AbstractMDP)::Bool
    return is_absorbing(env, state(env))
end

"""
    truncated(env::AbstractMDP)::Bool

Returns `true` to signal that the episode has ended prematurely, which may be due to the environment reaching a maximum number of steps. Note that this is different from the environment being in an absorbing (goal) state. By default, returns `false`.
"""
function truncated(env::AbstractMDP)::Bool
    return false
end

"""
    visualize(env::AbstractMDP{S, A}; kwargs...) where {S, A}

Visualize the current state of the environment. By default, returns `visualize(env, state(env), args...; kwargs...)` assuming that `env` is a mutable struct with a field `state` and `visualize(env, s)` is implemented.
"""
function visualize(env::AbstractMDP{S, A}; kwargs...) where {S, A}
    return visualize(env, state(env); kwargs...)
end

# -------------------------------------------------------------------













"""
    AbstractWrapper{S, A} <: AbstractMDP{S, A}

An abstract type for wrapping an MDP. This is useful for adding additional functionality to an MDP without having to reimplement the MDP interface. By default, all methods are forwarded to the wrapped MDP. To implement a wrapper, you should implement the `unwrapped` function, which returns the wrapped MDP. By default, this function returns `env.env` assuming that the wrapper is a struct with a field `env` that is the wrapped MDP.
"""
abstract type AbstractWrapper{S, A} <: AbstractMDP{S, A} end

"""
    unwrapped(env::AbstractWrapper)::AbstractMDP

Returns the wrapped MDP. By default, returns `env.env` assuming that the wrapper is a struct with a field `env` that is the wrapped MDP.
"""
unwrapped(env::AbstractWrapper)::AbstractMDP = env.env

@inline state_space(env::AbstractWrapper{S, A}) where {S, A} = state_space(unwrapped(env))
@inline action_space(env::AbstractWrapper{S, A}) where {S, A} = action_space(unwrapped(env))
@inline action_meaning(env::AbstractWrapper{S, A}, a::A) where {S, A} = action_meaning(unwrapped(env), a)
@inline action_meanings(env::AbstractWrapper{S, A}) where {S, A} = action_meanings(unwrapped(env))
@inline start_state_support(env::AbstractWrapper{S, A}) where {S, A} = start_state_support(unwrapped(env))
@inline start_state_probability(env::AbstractWrapper{S, A}, s::S) where {S, A} = start_state_probability(unwrapped(env), s)
@inline start_state_distribution(env::AbstractWrapper{S, A}, support) where {S, A} = start_state_distribution(unwrapped(env), support)
@inline transition_support(env::AbstractWrapper{S, A}, s::S, a::A) where {S, A} = transition_support(unwrapped(env), s, a)
@inline transition_probability(env::AbstractWrapper{S, A}, s::S, a::A, s′::S) where {S, A} = transition_probability(unwrapped(env), s, a, s′)
@inline transition_distribution(env::AbstractWrapper{S, A}, s::S, a::A, support) where {S, A} = transition_distribution(unwrapped(env), s, a, support)
@inline reward(env::AbstractWrapper{S, A}, s::S, a::A, s′::S) where {S, A} = reward(unwrapped(env), s, a, s′)
@inline is_absorbing(env::AbstractWrapper{S, A}, s::S) where {S, A} = is_absorbing(unwrapped(env), s)
@inline visualize(env::AbstractWrapper{S, A}, s::S; kwargs...) where {S, A} = visualize(unwrapped(env), s; kwargs...)

@inline state(env::AbstractWrapper{S, A}) where {S, A} = state(unwrapped(env))
@inline action(env::AbstractWrapper{S, A}) where {S, A} = action(unwrapped(env))
@inline reward(env::AbstractWrapper{S, A}) where {S, A} = reward(unwrapped(env))
@inline factory_reset!(env::AbstractWrapper{S, A}) where {S, A} = factory_reset!(unwrapped(env))
@inline reset!(env::AbstractWrapper{S, A}; rng::AbstractRNG=Random.GLOBAL_RNG) where {S, A} = reset!(unwrapped(env); rng=rng)
@inline step!(env::AbstractWrapper{S, A}, a::A; rng::AbstractRNG=Random.GLOBAL_RNG) where {S, A} = step!(unwrapped(env), a; rng=rng)
@inline in_absorbing_state(env::AbstractWrapper{S, A}) where {S, A} = in_absorbing_state(unwrapped(env))
@inline truncated(env::AbstractWrapper{S, A}) where {S, A} = truncated(unwrapped(env))
@inline visualize(env::AbstractWrapper{S, A}; kwargs...) where {S, A} = visualize(unwrapped(env); kwargs...)