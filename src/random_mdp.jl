using Random
using Distributions

export RandomDiscreteMDP

"""
    RandomDiscreteMDP(nstates, nactions; α = 1.0, β = 1.0, uniform_dist_rewards=false)

Create a random discrete MDP with `nstates` states and `nactions` actions. The transition probabilities are sampled from a Dirichlet distribution with parameter `α` and the mean rewards are sampled from a normal distribution with mean 1 and standard deviation `β`. If `uniform_dist_rewards` is true, then the rewards are sampled from a uniform distribution on the interval [1 - β, 1 + β]. When the MDP is 'stepped', the reward is generated stochastically from the normal distribution with mean `R(a|s)` and standard deviation 1.0.
"""
mutable struct RandomDiscreteMDP <: AbstractMDP{Int, Int}
    const nstates::Int
    const nactions::Int
    const d₀::Vector{Float64}
    const T::Array{Float64, 3} # Pr(s'| a, s)
    const R::Matrix{Float64}  # R(a|s)

    state::Int
    action::Int
    reward::Float64
    function RandomDiscreteMDP(rng, nstates, nactions; α = 1.0, β = 1.0, uniform_dist_rewards=false)
        d₀ = zeros(nstates)
        d₀[1] = 1
        T = zeros(nstates, nactions, nstates)
        for s in 1:nstates
            T[:, :, s] .= rand(rng, Dirichlet(nstates, α), nactions)
        end
        @assert all(sum(T; dims=1) .≈ 1.0)
        if uniform_dist_rewards
            R = rand(rng, Uniform(1.0 - β, 1.0 + β), nactions, nstates)
        else
            R = 1.0 .+ β * randn(rng, nactions, nstates)
        end
        return new(nstates, nactions, d₀, T, R, 1, 1, 0)
    end
end

RandomDiscreteMDP(nstates, nactions; kwargs...) = RandomDiscreteMDP(Random.GLOBAL_RNG, nstates, nactions; kwargs...)

state_space(mdp::RandomDiscreteMDP) = IntegerSpace(mdp.nstates)
action_space(mdp::RandomDiscreteMDP) = IntegerSpace(mdp.nactions)

start_state_probability(mdp::RandomDiscreteMDP, s::Int)::Float64 = mdp.d₀[s]

transition_probability(mdp::RandomDiscreteMDP, s::Int, a::Int, s′::Int)::Float64 = mdp.T[s′, a, s]

reward(mdp::RandomDiscreteMDP, s::Int, a::Int, s′::Int)::Float64 = mdp.R[a, s]

is_absorbing(mdp::RandomDiscreteMDP, s::Int)::Bool = false


function reset!(env::RandomDiscreteMDP; rng::AbstractRNG=Random.GLOBAL_RNG)::Nothing
    support = collect(start_state_support(env))
    env.state = sample(rng, support, ProbabilityWeights(start_state_distribution(env, support)))
    env.action = 0
    env.reward = 0
    nothing
end

function step!(env::RandomDiscreteMDP, a::Int; rng::AbstractRNG=Random.GLOBAL_RNG)::Nothing
    @assert a ∈ action_space(env)
    env.action = a
    if in_absorbing_state(env)
        @warn "The environment is in an absorbing state. This `step!` will not do anything. Please call `reset!`."
        env.reward = 0
    else
        s = state(env)
        support = collect(transition_support(env, s, a))
        s′ = sample(rng, support, ProbabilityWeights(transition_distribution(env, s, a, support)))
        env.state = s′
        env.action = a
        r̄ = reward(env, s, a, s′)
        env.reward = r̄ + randn(rng)
    end
    nothing
end