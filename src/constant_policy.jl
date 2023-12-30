export ConstantPolicy


struct ConstantPolicy{S, A} <: AbstractPolicy{S, A}
    action::A
end

function ConstantPolicy(env::AbstractEnv{S, A}, a::A) where {S, A}
    return ConstantPolicy{S, A}(a)
end

function (p::ConstantPolicy{S, A})(rng::AbstractRNG, s::S) where {S, A}
    return p.action
end

function (p::ConstantPolicy{S, A})(s::S, a::A) where {S, A}
    return p.action == a
end