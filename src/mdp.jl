using StatsBase

export AbstractMDP, state_space, action_space, action_meaning, action_meanings,  horizon, discount_factor, start_state_support, start_state_probability, start_state_distribution, transition_support, transition_probability, transition_distribution, reward, is_absorbing, state, action, reward, reset!, factory_reset!, step!, in_absorbing_state, visualize

abstract type AbstractMDP{S, A} end  # S = typeof(state(env)), A = typeof(action(env))

# necessary:

function state_space(mdp::AbstractMDP{S, A})::AbstractSpace{S} where {S, A}
    error("not implemented")
end

function action_space(mdp::AbstractMDP{S, A})::AbstractSpace{A} where {S, A}
    error("not implemented")
end

function action_meaning(::AbstractMDP{S, A}, a::A)::String where {S, A}
    return "action $a"
end

function action_meanings(mdp::AbstractMDP{S, Int})::Vector{String} where {S}
    return map(a -> action_meaning(mdp, a), action_space(mdp))
end

function horizon(mdp::AbstractMDP)::Int
    error("not implemented")
end

discount_factor(mdp::AbstractMDP)::Float64 = 0.99


# ------------------------- when the model is known ------------------------------
function start_state_support(mdp::AbstractMDP{S, A}) where {S, A} # return something finite iterable
    return state_space(mdp)
end

function start_state_probability(mdp::AbstractMDP{S, A}, s::S)::Float64 where {S, A}
    error("starting states probabilities unknown")
end

function start_state_distribution(mdp::AbstractMDP{S, A}, support)::Vector{Float64} where {S,A} # probabilities of candidates_s₀
    return map(s₀ -> start_state_probability(mdp, s₀),  support)
end

function transition_support(mdp::AbstractMDP{S, A}, s::S, a::A) where {S, A}  # return something finite iterable
    return state_space(mdp)
end

function transition_probability(env::AbstractMDP{S, A}, s::S, a::A, s′::S)::Float64 where {S, A}
    error("transition probabilities unknown")
end

function transition_distribution(mdp::AbstractMDP{S, A}, s::S, a::A, support)::Vector{Float64} where {S,A}  # probabilities of candidates_s′
    return map(s′ -> transition_probability(mdp, s, a, s′),  support)
end

function reward(mdp::AbstractMDP{S, A}, s::S, a::A, s′::S)::Float64 where {S, A}
    error("reward function unknown")
end

function is_absorbing(mdp::AbstractMDP{S, A}, s::S)::Bool where {S, A}
    error("absorbing (goal) states unknown")
end

function visualize(mdp::AbstractMDP{S, A}, s::S) where {S, A}
    error("visualization not implemented")
end


# ---------------------------------------------------------------------------



# --------------------------------- mdp as an RL environment ---------------

# expected to have feilds `state`, `action`, `reward`. If model unknown, need to override reset!, step!, is_terminated

"""Reset any data structures and parameters (e.g., a hidden state or logs) that would otherwise persist between episodes."""
function factory_reset!(env::AbstractMDP)
    nothing
end


@inline function state(env::AbstractMDP{S, A})::S where {S, A}
    env.state
end

@inline function action(env::AbstractMDP{S, A})::A where {S, A}
    env.action
end

@inline function reward(env::AbstractMDP{S, A})::Float64 where {S, A}
    env.reward
end

"""Sample new episode"""
function reset!(env::AbstractMDP{S, A}; rng::AbstractRNG=Random.GLOBAL_RNG)::Nothing where {S, A}
    support = collect(start_state_support(env))
    env.state = sample(rng, support, ProbabilityWeights(start_state_distribution(env, support)))
    nothing
end

function step!(env::AbstractMDP{S, A}, a::A; rng::AbstractRNG=Random.GLOBAL_RNG)::Nothing where {S, A}
    s = state(env)
    support = collect(transition_support(env, s, a))
    s′ = sample(rng, support, ProbabilityWeights(transition_distribution(env, s, a, support)))
    r = reward(env, s, a, s′)
    env.state = s′
    env.action = a
    env.reward = r
    nothing
end

function in_absorbing_state(env::AbstractMDP)::Bool
    return is_absorbing(env, state(env))
end

function visualize(env::AbstractMDP)
    return visualize(env, state(env))
end

# -------------------------------------------------------------------