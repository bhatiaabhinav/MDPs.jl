using Random

export AbstractSpace, IntegerSpace, TensorSpace, VectorSpace, MatrixSpace, EnumerableTensorSpace, EnumerableVectorSpace, EnumerableMatrixSpace, discretize

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
@inline Base.getindex(ds::IntegerSpace, i::Int) = i
@inline Base.firstindex(ds::IntegerSpace) = 1
@inline Base.lastindex(ds::IntegerSpace) = ds.n
@inline Base.keys(ds::IntegerSpace) = 1:ds.n
@inline Base.indexin(elems, ds::IntegerSpace) = indexin(elems, 1:ds.n)

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
    EnumerableTensorSpace{T, N}(elements::AbstractVector{Array{T, N}})

Construct a `EnumerableTensorSpace` with finite number of elements `elements`. All elements must be of the same size and be unique. For iteration, the elements are returned in the order they are provided in the constructor. The ith element can be accessed with space[i].
"""
struct EnumerableTensorSpace{T <: Real, N, M} <: AbstractSpace{Array{T, N}}
    elements::Array{T, M}  # All elements stored in a single array. M = N + 1
    lows::Array{T, N}  # Lower bounds of the space
    highs::Array{T, N}  # Upper bounds of the space
    elements_to_index::Dict{Array{T, N}, Int}  # Map from element to index in `elements`

    function EnumerableTensorSpace{T, N}(elements::AbstractVector{Array{T, N}}) where {T<:Real, N}
        @assert length(elements) > 0
        @assert allequal(size.(elements))
        lows = min.(elements...)
        highs = max.(elements...)
        elements_to_index = Dict{Array{T, N}, Int}()
        for (i, element) in enumerate(elements)
            @assert !haskey(elements_to_index, element)  "All elements must be unique"
            elements_to_index[element] = i
        end
        M = N + 1
        new{T, N, M}(cat(elements...; dims=M), lows, highs, elements_to_index)
    end
end


@inline Base.size(cs::EnumerableTensorSpace) = size(cs.elements)
@inline Base.size(cs::EnumerableTensorSpace, dim) = size(cs.elements, dim)
@inline Base.length(cs::EnumerableTensorSpace) = size(cs.elements)[end]
@inline Base.in(m::Array{T, N}, cs::EnumerableTensorSpace{T, N}) where {T<:AbstractFloat, N} = haskey(cs.elements_to_index, m)
@inline Base.iterate(cs::EnumerableTensorSpace{T, N, M}, iter_state=1) where {T<:AbstractFloat, N, M} = iter_state > length(cs) ? nothing : (copy(selectdim(cs.elements, M, iter_state)), iter_state + 1)
Random.rand(rng::AbstractRNG, c::EnumerableTensorSpace{T, N, M}) where {T<:AbstractFloat, N, M} = c[rand(rng, 1:length(c))]
@inline Base.getindex(cs::EnumerableTensorSpace{T, N, M}, i::Int) where {T<:AbstractFloat, N, M} = copy(selectdim(cs.elements, M, i))
@inline Base.firstindex(cs::EnumerableTensorSpace{T, N, M}) where {T<:AbstractFloat, N, M} = 1
@inline Base.lastindex(cs::EnumerableTensorSpace{T, N, M}) where {T<:AbstractFloat, N, M} = length(cs)
@inline Base.keys(cs::EnumerableTensorSpace{T, N, M}) where {T<:AbstractFloat, N, M} = LinearIndices(1:length(cs))
function Base.indexin(elems::AbstractArray{Array{T, N}}, cs::EnumerableTensorSpace{T, N, M})::Array{Union{Nothing, Int}} where {T<:AbstractFloat, N, M}
    map(elem -> haskey(cs.elements_to_index, elem) ? cs.elements_to_index[elem] : nothing, elems)
end
@inline Base.indexin(elem::Array{T, N}, cs::EnumerableTensorSpace{T, N, M}) where {T<:AbstractFloat, N, M} = indexin(fill(elem), cs)

const EnumerableVectorSpace{T} = EnumerableTensorSpace{T, 1}
const EnumerableMatrixSpace{T} = EnumerableTensorSpace{T, 2}


"""
    discretize(x::Array{T, N}, lows::Array{T, N}, highs::Array{T, N}, num_bins::Array{Int, N}, precomputed_cumprod::Union{Array{Int, N}, Nothing}=nothing)::Int

Discretize a multi-dimensional input `x` into a single integer representing a bucket index.

# Arguments
- `x::Array{T, N}`: The input vector or N-dimensional array to be discretized. Each element should be a floating-point number.
- `lows::Array{T, N}`: The lower bounds for each dimension of the input vector/array.
- `highs::Array{T, N}`: The upper bounds for each dimension of the input vector/array.
- `num_bins::Array{Int, N}`: The number of bins into which each dimension should be discretized.
- `precomputed_cumprod::Union{Array{Int, N}, Nothing}`: Optional argument. The precomputed cumulative product of the bucket sizes. If `nothing`, the function computes the cumulative product.
- `assume_inbounds::Bool`: Optional argument. If `true`, the function does not check that the input vector lies within the specified range. Default: `false

# Returns
- `::Int`: The index of the bucket into which `x` falls when the space is discretized according to the specified bounds and number of buckets.

# Example
```julia
x = [0.6, 0.7]
lows = [0.0, 0.0]
highs = [1.0, 1.0]
buckets = [2, 2]
println(discretize(x, lows, highs, buckets))  # output: 4
```
"""
function discretize(x::Array{T, N}, lows::Array{T, N}, highs::Array{T, N}, num_bins::Array{Int, N}; precomputed_cumprod::Union{Vector{Int}, Nothing}=nothing, assume_inbounds::Bool=false)::Int where {T<:AbstractFloat, N}
    if !assume_inbounds  # Check that the input vector lies within the specified range
        for i in eachindex(x)
            @assert (lows[i] <= x[i] <= highs[i]) "x[$i] = $(x[i]) is not in the range $(lows[i]) to $(highs[i])]"
        end
    end
    xf64 = convert(Array{Float64, N}, x)  # Convert the input vector to Float64
    lowsf64 = convert(Array{Float64, N}, lows)  # Convert the lower bounds to Float64
    highsf64 = convert(Array{Float64, N}, highs)  # Convert the upper bounds to Float64
    bucket_widths = @. (highsf64 - lowsf64) / num_bins  # Compute the width of each bucket along each dimension
    posf64 = @. (xf64 - lowsf64) / bucket_widths  # Compute the relative position of the input vector along each dimension
    posf64 = clamp.(posf64, 0.0, num_bins .- 1e-8) # Clamp the relative position to the range [0, num_bins - 1e-8] to avoid floating-point errors and to handle the edge case where the input vector is equal to the upper bound
    discrete_vec = floor.(Int, posf64)  # Discretize the input vector
    cumprod_values = isnothing(precomputed_cumprod) ? cumprod([1; num_bins[1:end-1]]) : precomputed_cumprod  # Use precomputed cumulative product if available, else compute it
    index = sum(reshape(discrete_vec, :) .* cumprod_values) + 1  # Map the discretized vector to a single integer. Add 1 to avoid 0-indexing
    return index
end


export create_reverse_mapping

function create_reverse_mapping(lows::Array{T, N}, highs::Array{T, N}, num_bins::Array{Int, N}; precomputed_cumprod::Union{Vector{Int}, Nothing}=nothing)::Dict{Int, Array{T, N}} where {T<:AbstractFloat, N}
    # Compute the bucket widths
    bucket_widths = (highs .- lows) ./ num_bins
    println("bucket_widths = $bucket_widths")

    # Initialize an empty dictionary
    reverse_map = Dict{Int, Array{T, N}}()

    # Generate all combinations of bucket indices
    indices = Iterators.product((1:bin for bin in num_bins)...) |> collect
    println("indices = $indices")

    precomputed_cumprod = isnothing(precomputed_cumprod) ? cumprod([1; num_bins[1:end-1]]) : precomputed_cumprod

    for index_set in indices
        # Create a representative point for this set of indices (choose the center of the bucket)
        x = [lows[i] + (index - 0.5) * bucket_widths[i] for (i, index) in enumerate(index_set)]
        x = reshape(x, size(lows))
        # Discretize this point to get the action index
        index = discretize(x, lows, highs, num_bins, precomputed_cumprod=precomputed_cumprod, assume_inbounds=true)
        println("$index_set. $x -> $index")
        # Store the mapping from index to representative point
        reverse_map[index] = x
    end

    return reverse_map
end
