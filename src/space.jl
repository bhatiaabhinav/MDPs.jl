using Random

export AbstractSpace, IntegerSpace, TensorSpace, VectorSpace, MatrixSpace, discretize

"""
    AbstractSpace{E}

Abstract type for a space of elements of type `E`. Methods `eltype`, `ndims`, `in`, `size`, `length`, `iterate` and `rand` should be defined for any concrete subtype of `AbstractSpace{E}`.
"""
abstract type AbstractSpace{E} end
@inline Base.eltype(::Type{<:AbstractSpace{E}}) where E = E
@inline Base.eltype(::AbstractSpace{E}) where E = E
@inline Base.ndims(::Type{<:AbstractSpace{E}}) where E = ndims(E)
@inline Base.ndims(::AbstractSpace{E}) where E = ndims(E)
@inline Base.in(::Any, ::AbstractSpace{E}) where E = false

"""
    IntegerSpace(n)

Space of integers from 1 to `n`.
"""
struct IntegerSpace <: AbstractSpace{Int}
    n::Int
    function IntegerSpace(n::Int)
        @assert n > 0
        new(n) 
    end
end

@inline Base.size(ds::IntegerSpace, args...) = size(1:ds.n, args...)
@inline Base.length(ds::IntegerSpace) = length(1:ds.n)
@inline Base.in(el::Int, ds::IntegerSpace) = in(el, 1:ds.n)
@inline Base.Iterators.iterate(ds::IntegerSpace, args...) = iterate(1:ds.n, args...)

Random.rand(rng::AbstractRNG, d::Random.SamplerTrivial{IntegerSpace}) = rand(rng, 1:d[].n);


"""
    struct TensorSpace{T <: Real, N} <: AbstractSpace{Array{T, N}}

Space of tensors of type `T` and ndims (rank) `N`. Note that `size(cs::TensorSpace) = (size(cs.lows)..., Inf)` and `length(cs::TensorSpace) = Inf`.

    TensorSpace{T, N}(low::Real, high::Real, size::NTuple{N, Int})
    TensorSpace{T, N}(lows::Array{T, N}, highs::Array{T, N})

Construct a `TensorSpace` with bounds `low` and `high` or `lows` and `highs` and size `size`. The bounds must be ordered such that `low[i] <= high[i]` for all `i`. The size must be a tuple of `N` integers.
"""
struct TensorSpace{T <: Real, N} <: AbstractSpace{Array{T, N}}
    lows::Array{T, N}
    highs::Array{T, N}
    function TensorSpace{T, N}(low::Real, high::Real, size::NTuple{N, Int}) where {T<:Real, N}
        @assert low <= high
        new{T, N}(fill(T(low), size), fill(T(high), size))
    end
    function TensorSpace{T, N}(lows::Array{T, N}, highs::Array{T, N}) where {T<:Real, N}
        @assert all(lows .<= highs)
        @assert size(lows) == size(highs)
        new{T, N}(lows, highs)
    end
end

@inline Base.size(cs::TensorSpace) = (size(cs.lows)..., Inf)
@inline Base.size(cs::TensorSpace, dim) = size(cs)[dim]
@inline Base.length(cs::TensorSpace) = Inf
@inline Base.in(m::Array{T, N}, cs::TensorSpace{T, N}) where {T,N} = all(cs.lows .<= m .<= cs.highs)
function Random.rand(rng::AbstractRNG, c::Random.SamplerTrivial{TensorSpace{T, N}}) where {T<:AbstractFloat, N}
    cs::TensorSpace{T, N} = c[]
    p1::Array{T, N} = rand(rng, T, size(cs.lows))
    scale::Array{T, N} = (min.(cs.highs, floatmax(T)) .- max.(cs.lows, -floatmax(T)))
    shift::Array{T, N} = max.(cs.lows, -floatmax(T))
    return p1 .* scale .+ shift 
end
function Random.rand(rng::AbstractRNG, c::Random.SamplerTrivial{TensorSpace{T, N}}) where {T<:Integer, N}
    cs::TensorSpace{T, N} = c[]
    p1::Array{T, N} = rand(rng, T, size(cs.lows))
    scale::Array{T, N} = (min.(cs.highs, typemax(T)) .- max.(cs.lows, -typemax(T)))
    shift::Array{T, N} = max.(cs.lows, -typemax(T))
    return p1 .* scale .+ shift 
end

"""
    VectorSpace{T}

    Alias for `TensorSpace{T, 1}`.
"""
const VectorSpace{T} = TensorSpace{T, 1}

"""
    MatrixSpace{T}

    Alias for `TensorSpace{T, 2}`.
"""
const MatrixSpace{T} = TensorSpace{T, 2}


"""
    discretize(x::Array{T, N}, ts::TensorSpace{T, N}, num_buckets::Vector{Int})::Int

Discretize `x` into one of `prod(num_buckets)` buckets. `ts` is a `TensorSpace` and `num_buckets` is a vector of length `ndims(x)` specifying the number of buckets in each dimension.
"""
function discretize(x::Array{T, N}, ts::TensorSpace{T, N}, num_buckets::Vector{Int})::Int where {T<:AbstractFloat, N}
    return discretize(x, ts.lows, ts.highs, num_buckets)
end

"""
    discretize(x::Array{T, N}, lows::Array{T, N}, highs::Array{T, N}, num_buckets::Vector{Int})::Int

Discretize `x` into one of `prod(num_buckets)` buckets. `lows` and `highs` are the lower and upper bounds of the space. `num_buckets` is a vector of length `ndims(x)` specifying the number of buckets in each dimension.
"""
function discretize(x::Array{T, N}, lows::Array{T, N}, highs::Array{T, N}, num_buckets::Vector{Int})::Int where {T<:AbstractFloat, N}
    x = clamp.((x - lows) ./ (highs - lows), T(1e-9), T(1))
    x = Int.(ceil.(num_buckets .* x))
    m = reverse(cumprod(reverse(num_buckets)))
    x = sum((x.-1)[1:end-1] .* m[2:end]) + x[end]
    @assert x <= prod(num_buckets)
    return x
end