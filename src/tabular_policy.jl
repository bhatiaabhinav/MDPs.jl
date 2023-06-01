export TabularPolicy, GreedyPolicy, EpsilonGreedyPolicy

"""
    TabularPolicy(preferences)

Tabular policy that takes actions according to the preferences stored in the matrix `preferences`. The policy is stochastic and the probability of taking action `a` in state `s` is given by `preferences[a, s] / sum(preferences[:, s])`.
"""
mutable struct TabularPolicy <: AbstractPolicy{Int, Int}
    preferenes::Matrix{Float64}  # π(a|s)
end

function (p::TabularPolicy)(rng::AbstractRNG, s::Int)::Int
    return sample(rng, 1:size(p.preferenes, 1), ProbabilityWeights(p.preferenes[:, s]))
end

function (p::TabularPolicy)(s::Int, a::Int)::Float64
    return p.preferenes[a, s] / sum(p.preferenes[:, s])
end


"""
    GreedyPolicy(preferences)

Tabular policy that takes actions according to the preferences stored in the matrix `preferences`. The policy is deterministic and the action with the highest preference is taken.
"""
mutable struct GreedyPolicy <: AbstractPolicy{Int, Int}
    preferenes::Matrix{Float64}  # π(a|s)
end

function (p::GreedyPolicy)(rng::AbstractRNG, s::Int)::Int
    return argmax(p.preferenes[:, s])
end

function (p::GreedyPolicy)(s::Int, a::Int)::Float64
    return Float64(a == p(s))
end


"""
    EpsilonGreedyPolicy(preferences, ϵ)

Tabular policy that takes actions according to the preferences stored in the matrix `preferences`. The policy is stochastic and the probability of taking the action with the highest preference is `1 - ϵ + ϵ / n`, where `n` is the number of actions. The probability of taking any other action is `ϵ / n`.
"""
mutable struct EpsilonGreedyPolicy <: AbstractPolicy{Int, Int}
    preferenes::Matrix{Float64}  # π(a|s)
    ϵ::Float64
end

function (p::EpsilonGreedyPolicy)(rng::AbstractRNG, s::Int)::Int
    return rand(rng) < p.ϵ ? rand(rng, 1:size(p.preferenes, 1)) : argmax(p.preferenes[:, s])
end

function (p::EpsilonGreedyPolicy)(s::Int, a::Int)::Float64
    n = size(p.preferenes, 1)
    return a == argmax(p.preferenes[:, s]) ? (1 - p.ϵ + p.ϵ / n) : p.ϵ / n
end




# struct SoftGreedyPolicy <: AbstractPolicy{Int, Int}
#     preferenes::Matrix{Float64}  # π(a|s)
#     temperature::Float64
# end

# function (p::SoftGreedyPolicy)(rng::AbstractRNG, s::Int)::Int
#     return rand(rng) < p.ϵ ? rand(rng, 1:size(p.preferenes, 1)) : argmax(p.preferenes[:, s])
# end

# function (p::SoftGreedyPolicy)(s::Int, a::Int)::Float64
#     n = size(p.preferenes, 1)
#     return a == argmax(p.preferenes[:, s]) ? (1 - p.ϵ + p.ϵ / n) : p.ϵ / n
# end