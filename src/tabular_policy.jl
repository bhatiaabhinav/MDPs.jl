export TabularPolicy, GreedyPolicy, EpsilonGreedyPolicy


mutable struct TabularPolicy <: AbstractPolicy{Int, Int}
    preferenes::Matrix{Float64}  # π(a|s)
end

function (p::TabularPolicy)(rng::AbstractRNG, s::Int)::Int
    return sample(rng, 1:size(p.preferenes, 1), ProbabilityWeights(p.preferenes[:, s]))
end

function (p::TabularPolicy)(s::Int, a::Int)::Float64
    return p.preferenes[a, s] / sum(p.preferenes[:, s])
end



mutable struct GreedyPolicy <: AbstractPolicy{Int, Int}
    preferenes::Matrix{Float64}  # π(a|s)
end

function (p::GreedyPolicy)(rng::AbstractRNG, s::Int)::Int
    return argmax(p.preferenes[:, s])
end

function (p::GreedyPolicy)(s::Int, a::Int)::Float64
    return Float64(a == p(s))
end



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