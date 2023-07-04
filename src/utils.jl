using Statistics
using Plots
using StatsPlots
using CSV
using DataFrames

export RunningMeanVariance, dfread, plot_run!, plot_run, compare_runs, runs_in_dir, compare_dir_runs, plot_rungroup!, plot_rungroup, compare_rungroups, compare_dirs_rungroups

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











dfread(csv_filename, colname) = CSV.read(csv_filename, DataFrame)[:, colname]

"""
    plot_run!(pl::Plots.Plot, csv_filename::String; x::Symbol=:episodes, y::Symbol=:R̄, xlabel=x, ylabel=y, label = replace(basename(csv_filename), ".csv" => ""), plot_kwargs...)

Plot! a run from a csv file. The csv file should have columns `x` and `y` (default `:episodes` and `:R̄` respectively). The label is automatically generated from the filename. Addtional keyword arguments are passed to `plot!`.
"""
function plot_run!(pl::Plots.Plot, csv_filename::String; x::Symbol=:episodes, y::Symbol=:R̄, xlabel=x, ylabel=y, label = replace(basename(csv_filename), ".csv" => ""), plot_kwargs...)
    df = CSV.read(csv_filename, DataFrame)
    plot!(pl, df[:, x], df[:, y]; xlabel=xlabel, ylabel=ylabel, label=label, plot_kwargs...)
    pl
end
plot_run!(csv_filename::String; kwargs...) = plot_run!(Plots.current(), csv_filename; kwargs...)

"""
    plot_run(csv_filename::String; x::Symbol=:episodes, y::Symbol=:R̄, xlabel=x, ylabel=y, label = replace(basename(csv_filename), ".csv" => ""), plot_kwargs...)

Plot a run from a csv file. The csv file should have columns `x` and `y` (default `:episodes` and `:R̄` respectively). The label is automatically generated from the filename. Addtional keyword arguments are passed to `plot!`.
"""
plot_run(args...; kwargs...) = plot_run!(plot(), args...; kwargs...)

"""
    compare_runs(csv_filenames::Vararg{String}; x::Symbol=:episodes, y::Symbol=:R̄, labels=:default, plot_kwargs...)

Plot multiple runs from csv files by invoking `plot_run` on each file. The csv files should have columns `x` and `y` (default `:episodes` and `:R̄` respectively). Labels are automatically generated from the filenames, unless `labels` is specified. Addtional keyword arguments are passed to `plot!`.
"""
function compare_runs(csv_filenames::Vararg{String}; x::Symbol=:episodes, y::Symbol=:R̄, labels=:default, plot_kwargs...)
    pl = plot()
    for (i, csv_filename) in enumerate(csv_filenames)
        if labels == :default
            plot_run!(pl, csv_filename; x=x, y=y)
        else
            plot_run!(pl, csv_filename; x=x, y=y, label=labels[i])
        end
    end
    plot!(;plot_kwargs...)
    pl
end

"""
    runs_in_dir(dir::String=".", pattern=r".*.csv")

Returns a list of filenames in a directory that match a pattern. The default pattern is to match all csv files. The filenames are returned as absolute paths.
"""
function runs_in_dir(dir::String=".", pattern=r".*.csv")
    filter(x -> occursin(pattern, x), readdir(dir; join=true))
end

"""
    compare_dir_runs(dirname="."; pattern=r".*.csv", kwargs...)

Compare all runs in a directory using `compare_runs`. The default pattern is to match all csv files. Addtional keyword arguments are passed to `compare_runs`.
"""
function compare_dir_runs(dirname="."; pattern=r".*.csv", kwargs...)
    if isdir(dirname) 
        csv_filenames = runs_in_dir(dirname, pattern)
        return compare_runs(csv_filenames...; kwargs...)
    end
end

"""
    plot_rungroup!([pl::Plots.Plot], csv_filenames::Vector{String}; x::Symbol=:episodes, y::Symbol=:R̄, errorstyle=:ribbon, xlabel=x, ylabel=y, label=join([replace(basename(filename), ".csv" => "") for (i, filename) in enumerate(csv_filenames)], ","), plot_kwargs...)

Plot! a group of runs from csv files as a ribbon using `errorline!` from `StatsPlots` package. The csv files should have columns `x` and `y` (default `:episodes` and `:R̄` respectively). The label is automatically generated from the filenames. Addtional keyword arguments are passed to `plot!`.
"""
function plot_rungroup!(pl, csv_filenames::Vector{String}; x::Symbol=:episodes, y::Symbol=:R̄, errorstyle=:ribbon, xlabel=x, ylabel=y, label=join([replace(basename(filename), ".csv" => "") for (i, filename) in enumerate(csv_filenames)], ","), plot_kwargs...)
    length(csv_filenames) == 0 && return pl
    xgroup = dfread(csv_filenames[1], x)
    ygroup = hcat(dfread.(csv_filenames, y)...)
    errorline!(pl, xgroup, ygroup, xlabel=xlabel, ylabel=ylabel, label=label, errorstyle=errorstyle, plot_kwargs...)
end
plot_rungroup!(csv_filenames::Vector{String}; kwargs...) = plot_rungroup!(Plots.current(), csv_filenames; kwargs...)

"""
    plot_rungroup(csv_filenames::Vector{String}; x::Symbol=:episodes, y::Symbol=:R̄, errorstyle=:ribbon, xlabel=x, ylabel=y, label=join([replace(basename(filename), ".csv" => "") for (i, filename) in enumerate(csv_filenames)], ","), plot_kwargs...)

Plot a group of runs from csv files as a ribbon using `errorline!` from `StatsPlots` package. The csv files should have columns `x` and `y` (default `:episodes` and `:R̄` respectively). The label is automatically generated from the filenames. Addtional keyword arguments are passed to `plot!`.
"""
plot_rungroup(csv_filenames::Vector{String}; kwargs...) = plot_rungroup!(plot(), csv_filenames; kwargs...)

"""
    compare_rungroups(csv_filenames_lists::Vararg{Vector{String}}; x::Symbol=:episodes, y::Symbol=:R̄, errorstyle=:ribbon, labels=:default, plot_kwargs...)

Plot multiple groups of runs from csv files by invoking `plot_rungroup` on each list of csv files. The csv files should have columns `x` and `y` (default `:episodes` and `:R̄` respectively). Labels are automatically generated from the filenames, unless `labels` is specified. Addtional keyword arguments are passed to `plot!`.
"""
function compare_rungroups(csv_filenames_lists::Vararg{Vector{String}}; x::Symbol=:episodes, y::Symbol=:R̄, errorstyle=:ribbon, labels=:default, plot_kwargs...)
    pl = plot()
    for (i, csv_filenames) in enumerate(csv_filenames_lists)
        println(csv_filenames)
        if labels == :default
            plot_rungroup!(pl, csv_filenames; x=x, y=y, errorstyle=errorstyle) 
        else
            plot_rungroup!(pl, csv_filenames; x=x, y=y, label=labels[i], errorstyle=errorstyle)
        end
    end
    plot!(;plot_kwargs...)
    pl
end

"""
    compare_dirs_rungroups(dirs::Vararg{String}; pattern=r".*.csv", kwargs...)

Compare all runs in a list of directories using `compare_rungroups`. The list of files that match the given pattern in a directory form a single rungroup. The default pattern is to match all csv files. Addtional keyword arguments are passed to `compare_rungroups`.
"""
function compare_dirs_rungroups(dirs::Vararg{String}; pattern=r".*.csv", kwargs...)
    csvgroups = [runs_in_dir(dir, pattern) for dir in dirs]
    compare_rungroups(csvgroups...; labels=dirs, kwargs...)
end
