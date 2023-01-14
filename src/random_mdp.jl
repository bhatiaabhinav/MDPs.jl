using Random
using Distributions

export RandomDiscreteMDP

"""State transitions follow a flat Dirichlet distribution. Rewards are determinisitic R(a|s) and are independently sampled from Normal(1,1) for each instance of the MDP. Each episode starts with state=1 deterministically."""
mutable struct RandomDiscreteMDP <: AbstractMDP{Int, Int}
    const nstates::Int
    const nactions::Int
    const d₀::Vector{Float64}
    const T::Array{Float64, 3} # Pr(s'| a, s)
    const R::Matrix{Float64}  # R(a|s)

    state::Int
    action::Int
    reward::Float64
    function RandomDiscreteMDP(rng, nstates, nactions)
        d₀ = zeros(nstates)
        d₀[1] = 1
        T = zeros(nstates, nactions, nstates)
        for s in 1:nstates
            T[:, :, s] .= rand(rng, Dirichlet(nstates, 1.0), nactions)
        end
        @assert all(sum(T; dims=1) .≈ 1.0)
        R = 1.0 .+ randn(rng, nactions, nstates)
        return new(nstates, nactions, d₀, T, R, 1, 1, 0)
    end
end

state_space(mdp::RandomDiscreteMDP) = IntegerSpace(mdp.nstates)
action_space(mdp::RandomDiscreteMDP) = IntegerSpace(mdp.nactions)

start_state_probability(mdp::RandomDiscreteMDP, s::Int)::Float64 = mdp.d₀[s]

transition_probability(mdp::RandomDiscreteMDP, s::Int, a::Int, s′::Int)::Float64 = mdp.T[s′, a, s]

reward(mdp::RandomDiscreteMDP, s::Int, a::Int, s′::Int)::Float64 = mdp.R[a, s]

is_absorbing(mdp::RandomDiscreteMDP, s::Int)::Bool = false