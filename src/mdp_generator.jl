export MDPGenerator

"""Reproducible. Iterating over a given MDPGenerator object will always produce the same sequence of MDPs (as long as the generator_fn uses the rng passed as an argument)"""
struct MDPGenerator{M <: AbstractMDP}
    generator_fn                # accepts (i, rng)
    rng₀::AbstractRNG           # will be cloned at first iteration, and will not be mutated
    num_mdps::Int               # num mdps to generate
    sspace::AbstractSpace
    aspace::AbstractSpace
    function MDPGenerator(generator_fn, rng₀::AbstractRNG, num_mdps::Int)
        mdp1 = generator_fn(1, Xoshiro(0))
        sspace = state_space(mdp1)
        aspace = action_space(mdp1)
        M = typeof(mdp1)
        return new{M}(generator_fn, copy(rng₀), num_mdps, sspace, aspace)
    end
end

MDPGenerator(generator_fn, num_mdps::Int) = MDPGenerator(generator_fn, Xoshiro(), num_mdps)
MDPGenerator(generator_fn) = MDPGenerator(generator_fn, Xoshiro(), typemax(Int))
MDPGenerator(generator_fn, rng₀::AbstractRNG) = MDPGenerator(generator_fn, rng₀, typemax(Int))

@inline Base.length(mdpg::MDPGenerator) = mdpg.num_mdps
@inline Base.eltype(::Type{MDPGenerator{M}}) where {M <: AbstractMDP} = M

function Base.iterate(mdpg::MDPGenerator{M})::Tuple{M, Tuple{Int, AbstractRNG}} where {M<:AbstractMDP}
    i, rng = 1, copy(mdpg.rng₀)
    return (mdpg.generator_fn(i, rng), (i + 1, rng))
end

function Base.iterate(mdpg::MDPGenerator{M}, iter_state::Tuple{Int, AbstractRNG})::Union{Nothing, Tuple{M, Tuple{Int, AbstractRNG}}} where {M<:AbstractMDP}
    i, rng = iter_state
    i > mdpg.num_mdps && return nothing
    return (mdpg.generator_fn(i, rng), (i + 1, rng))
end



@inline state_space(mdpg::MDPGenerator) = mdpg.sspace
@inline action_space(mdpg::MDPGenerator) = mdpg.aspace