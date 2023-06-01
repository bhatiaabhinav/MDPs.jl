using Random

export RandomPolicy

"""
    RandomPolicy(statespace, actionspace)

Random policy that samples actions uniformly at random from the action space.
"""
struct RandomPolicy{S, A} <: AbstractPolicy{S, A}
    actionspace::AbstractSpace{A}
    function RandomPolicy(statespace::AbstractSpace{S}, actionspace::AbstractSpace{A}) where {S, A}
        new{S, A}(actionspace)
    end
end

function (p::RandomPolicy{S, A})(rng::AbstractRNG, ::S)::A where {S, A}
    return rand(rng, p.actionspace)
end

function (p::RandomPolicy{S, A})(::S, ::A)::Float64 where {S, A}
    1 / length(p.actionspace)
end


"""
    RandomPolicy(mdp::AbstractMDP)

Random policy that samples actions uniformly at random from the action space of the MDP.
"""
RandomPolicy(mdp::AbstractMDP) = RandomPolicy(state_space(mdp), action_space(mdp))