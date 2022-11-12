using Random

export RandomDiscreteMDP

mutable struct RandomDiscreteMDP <: AbstractMDP{Int, Int}
    const nstates::Int
    const nactions::Int
    const d₀::Vector{Float64}
    const T::Array{Float64, 3} # Pr(s'| a, s)
    const R::Matrix{Float64}  # R(a|s)

    state::Int
    action::Int
    reward::Float64
    function RandomDiscreteMDP(nstates, nactions)
        d₀ = rand(nstates)
        d₀ /= sum(d₀)
        T = rand(nstates, nactions, nstates)
        T ./= sum(T, dims=1)
        R = rand(nactions, nstates)
        return new(nstates, nactions, d₀, T, R, 1, 1, 0)
    end
end

state_space(mdp::RandomDiscreteMDP) = IntegerSpace(mdp.nstates)
action_space(mdp::RandomDiscreteMDP) = IntegerSpace(mdp.nactions)

start_state_probability(mdp::RandomDiscreteMDP, s::Int)::Float64 = mdp.d₀[s]

transition_probability(mdp::RandomDiscreteMDP, s::Int, a::Int, s′::Int)::Float64 = mdp.T[s′, a, s]

reward(mdp::RandomDiscreteMDP, s::Int, a::Int, s′::Int)::Float64 = mdp.R[a, s]

is_absorbing(mdp::RandomDiscreteMDP, s::Int)::Bool = false