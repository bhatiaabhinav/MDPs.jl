export TabularPolicy, GreedyPolicy, EpsilonGreedyPolicy, BoltzmannPolicy, TabularPolicyContinuous

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


"""
    BoltzmannPolicy(preferences, τ)

Tabular policy that takes actions according to the preferences stored in the matrix `preferences`. The policy is stochastic and the probability of taking action `a` in state `s` is given by `exp(preferences[a, s] / τ) / sum(exp.(preferences[:, s] / τ))`.
"""
mutable struct BoltzmannPolicy <: AbstractPolicy{Int, Int}
    preferenes::Matrix{Float64}
    temperature::Float64
end
BoltzmannPolicy(preferences::Matrix{Float64}; τ::Real=1.0) = BoltzmannPolicy(preferences, τ)

function (p::BoltzmannPolicy)(rng::AbstractRNG, s::Int)::Int
    return sample(rng, 1:size(p.preferenes, 1), ProbabilityWeights(exp.(p.preferenes[:, s] / p.temperature)))
end

function (p::BoltzmannPolicy)(s::Int, a::Int)::Float64
    return exp(p.preferenes[a, s] / p.temperature) / sum(exp.(p.preferenes[:, s] / p.temperature))
end


"""
    TabularPolicyContinuous(tabular_policy::AbstractPolicy{Int, Int}, ts::EnumerableTensorSpace{T, N})

Tabular policy that acts on an environment having an enumerable state space `ts` and an integer action space. The probability of taking action `a` in state `s` is given by probability of that action according to the tabular policy `tabular_policy` in the integer state = index of `s` in `ts`.
"""
struct TabularPolicyContinuous{T, N} <: AbstractPolicy{Vector{T}, Int}
    tabular_policy::AbstractPolicy{Int, Int}
    ts::EnumerableTensorSpace{T, N}
    function TabularPolicyContinuous(tabular_policy::AbstractPolicy{Int, Int}, ts::EnumerableTensorSpace{T, N}) where {T, N}
        return new{T, N}(tabular_policy, ts)
    end
end

function (p::TabularPolicyContinuous{T})(rng::AbstractRNG, s::Vector{T})::Int where {T}
    return p.tabular_policy(rng, indexin(s, p.ts)[])
end

function (p::TabularPolicyContinuous{T})(s::Vector{T}, a::Int)::Float64 where {T}
    return p.tabular_policy(indexin(s, p.ts)[] , a)
end