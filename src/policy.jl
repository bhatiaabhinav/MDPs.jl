export AbstractPolicy


abstract type AbstractPolicy{S, A} end

function (::AbstractPolicy{S, A})(rng::AbstractRNG, ::S)::A where {S, A}
    error("not implented")
end

function (p::AbstractPolicy{S, A})(s::S)::A where {S, A}
    p(Random.GLOBAL_RNG, s)
end

function (::AbstractPolicy{S, A})(::S, ::A) where {S, A}
    error("not implented")
end


