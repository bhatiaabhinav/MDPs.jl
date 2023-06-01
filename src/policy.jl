export AbstractPolicy

"""
    AbstractPolicy{S, A}

Abstract type for MDP policies. A policy is a function that takes a state `s` (of type `S`) and returns an action `a` (of type `A`). The policy can be stochastic, in which case the function takes a random number generator `rng` as an additional argument. When inputs to the policy are both `s` and `a`, the policy returns the probability of taking action `a` in state `s`.
"""
abstract type AbstractPolicy{S, A} end

"""
    (::AbstractPolicy{S, A})([rng=GLOBAL_RNG], ::S)::A

Sample an action from the policy for state `s` using random number generator `rng`.
"""
function (::AbstractPolicy{S, A})(rng::AbstractRNG, ::S)::A where {S, A}
    error("not implented")
end

function (p::AbstractPolicy{S, A})(s::S)::A where {S, A}
    p(Random.GLOBAL_RNG, s)
end


"""
    (::AbstractPolicy{S, A})(::S, ::A)::Float64

Return the probability of taking action `a` in state `s`.
"""
function (::AbstractPolicy{S, A})(::S, ::A) where {S, A}
    error("not implented")
end


