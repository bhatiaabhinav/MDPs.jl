using Statistics

export RunningMeanVariance

Base.@kwdef mutable struct RunningMeanVariance{T<:AbstractFloat, N}
    const shape::NTuple{N, Int}
    const μ::Array{Float64, N} = zeros(Float64, shape...)
    const M₂::Array{Float64, N} = zeros(Float64, shape...)
    n::Int = 0
    lock::ReentrantLock = ReentrantLock()
end
RunningMeanVariance{T}() where T<:AbstractFloat = RunningMeanVariance{T, 0}(shape=()) 

function Base.empty!(rmv::RunningMeanVariance)
    lock(rmv.lock) do
        rmv.n = 0
        fill!(rmv.μ, 0)
        fill!(rmv.M₂, 0)
    end
    nothing
end

function Base.push!(rmv::RunningMeanVariance{T, N}, x::Union{AbstractArray{T, N}, T}) where {T, N}
    lock(rmv.lock) do
        rmv.n += 1
        Δ = x .- rmv.μ
        rmv.μ .+= Δ / rmv.n
        rmv.M₂ .+= Δ .* (x .- rmv.μ)
    end
    nothing
end

function Statistics.mean(rmv::RunningMeanVariance{T, N})::Union{T, Array{T, N}} where {T, N}
    μ::Array{Float64, N} = lock(rmv.lock) do
        return rmv.n < 1 ? fill(NaN, rmv.shape...) : rmv.μ
    end
    return N == 0 ? T(μ[]) : convert(Array{T, N}, μ)
end

function Statistics.var(rmv::RunningMeanVariance{T, N}; corrected::Bool=true)::Union{T, Array{T, N}} where {T, N}
    σ²::Array{Float64, N} = lock(rmv.lock) do
        return rmv.n < 2 ? fill(NaN, rmv.shape...) : rmv.M₂ / (rmv.n - Int(corrected))
    end
    return N == 0 ? T(σ²[]) : convert(Array{T, N}, σ²)
end

function Statistics.std(rmv::RunningMeanVariance{T, N}; corrected::Bool=true)::Union{T, Array{T, N}} where {T, N}
    σ::Union{Float64, Array{Float64, N}} = lock(rmv.lock) do
        return rmv.n < 2 ? (N == 0 ? NaN : fill(NaN, rmv.shape...)) : sqrt.(rmv.M₂ / (rmv.n - Int(corrected)))
    end
    return N == 0 ? T(σ) : convert(Array{T, N}, σ)  # sqrt already converts 0-dim Array to scalar. No need to do σ[]
end