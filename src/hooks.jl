using StatsBase
using ProgressMeter
using DataFrames
using CSV

export EmptyHook, EmpiricalPolicyEvaluationHook, ProgressMeterHook, LoggingHook, DataRecorderHook, VideoRecorderHook, PlotHook, PlotEverythingHook


"""
    EmptyHook()

Empty hook that does nothing.
"""
struct EmptyHook <: AbstractHook end


"""
    EmpiricalPolicyEvaluationHook(π, γ, horizon, n, sample_size; env=nothing)

Hook that evaluates the policy `π` every `n` episodes in the experiment using `sample_size` trajectory samples. The mean returns are stored in the `returns` list of the hook. The evaluation is done on the environment `env`. If no environment is provided, the environment used in the experiment is used instead. The trajectories are truncated at `horizon` steps and the discount factor for recording returns is `γ`. Internally, the `interact` function is used to generate the trajectories and the mean return is computed as: `mean(interact(env, π, γ, horizon, sample_size)[1])`.
"""
struct EmpiricalPolicyEvaluationHook <: AbstractHook
    π::AbstractPolicy
    γ::Real
    horizon::Real
    n::Int
    sample_size::Int
    env::Union{Nothing, AbstractMDP}
    returns::Vector{Float64}
    EmpiricalPolicyEvaluationHook(π::AbstractPolicy, γ::Real, horizon::Real, n::Int, sample_size::Int; env::Union{Nothing, AbstractMDP}=nothing) = new(π, γ, horizon, n, sample_size, env, Float64[])
end

function preexperiment(peh::EmpiricalPolicyEvaluationHook; env, kwargs...)
    _env = isnothing(peh.env) ? env : peh.env
    push!(peh.returns, mean(interact(_env, peh.π, peh.γ, peh.horizon, peh.sample_size)[1]))
    nothing
end

function postepisode(peh::EmpiricalPolicyEvaluationHook; env, returns, kwargs...)
    if length(returns) % peh.n == 0
        _env = isnothing(peh.env) ? env : peh.env
        push!(peh.returns, mean(interact(_env, peh.π, peh.γ, peh.horizon, peh.sample_size)[1]))
    end
    nothing
end




"""
    ProgressMeterHook(; kwargs...)

Hook that uses a ProgressMeter to display progress as the experiment is running. Progress is defined as the number of episodes completed divided by `max_trials` or the number of steps taken divided by `max_steps`, whichever is larger. The hook can be initialized with any keyword arguments accepted by `ProgressMeter.Progress`.
"""
struct ProgressMeterHook <: AbstractHook
    progress::Progress
    ProgressMeterHook(; kwargs...) = new(Progress(0; kwargs...))
end

function postepisode(pmh::ProgressMeterHook; max_trials, lengths, max_steps, steps, kwargs...)
    episodes = length(lengths)
    if (episodes / max_trials) > (steps / max_steps)
        pmh.progress.n = max_trials
        update!(pmh.progress, episodes)
    else
        pmh.progress.n = max_steps
        update!(pmh.progress, steps)
    end
    nothing
end

function postexperiment(pmh::ProgressMeterHook; kwargs...)
    finish!(pmh.progress)
    nothing
end


"""
    LoggingHook(stats_getter = nothing; n::Int = 1, smooth_over::Int = 100, loggers::Vector{Base.AbstractLogger} = [Base.current_logger()])

Hook that logs metrics and statistics using the specified `loggers`, which is a list of `AbstractLogger`s. By default, only `Base.current_logger()` is used, which is usually a `ConsoleLogger`. The metrics/statistics are logged every `n` episodes and include the number of `steps` taken, the number of `episodes` completed, the return `R` of the last episode, the average return `R̄` over the last `smooth_over` episodes, and the length `L` of the last episode. If a `stats_getter` function is provided, the entries in the statistics dictionary returned by the function is also logged. The `stats_getter` function is called with no arguments and must return a dictionary with `Symbol` keys and any values.
"""
mutable struct LoggingHook <: AbstractHook
    stats_getter
    n::Int
    smooth_over::Int
    loggers::Vector{Base.AbstractLogger}
    function LoggingHook(stats_getter=nothing; n::Int=1, smooth_over::Int=100, loggers::Vector{Base.AbstractLogger}=Base.AbstractLogger[Base.current_logger()])
        new(stats_getter, n, smooth_over, loggers)
    end
end

function postepisode(slh::LoggingHook; steps, returns, lengths, kwargs...)
    episodes = length(returns)
    if episodes % slh.n == 0
        min_recs = slh.smooth_over
        R̄ =  length(returns) < min_recs ? mean(returns) : mean(returns[end-min_recs+1:end])
        R = returns[end]
        L = lengths[end]
        stats::Dict{Symbol, Any} = isnothing(slh.stats_getter) ? Dict{Symbol, Any}() : slh.stats_getter()
        for logger in slh.loggers
            Base.with_logger(logger) do
                @info "stats" episodes=episodes steps=steps R=R R̄=R̄ L=L stats...
            end
        end
    end
    nothing
end

"""
    DataRecorderHook(stats_getter = nothing, csv = nothing; smooth_over::Int = 100, overwrite::Bool = false)

Hook that records metrics and statistics in a `DataFrame`. The metrics/statistics include the number of `steps` taken, the number of `episodes` completed, the return `R` of the last episode, the average return `R̄` over the last `smooth_over` episodes, and the length `L` of the last episode. If a `stats_getter` function is provided, the entries in the statistics dictionary returned by the function is also recorded. The `stats_getter` function is called with no arguments and must return a dictionary with `Symbol` keys and any values. If a `csv` file path is provided, the `DataFrame` is written to the file every `n` episodes. If the file already exists, the file name is incremented by appending a number to the file name (e.g. `data.csv` becomes `data.1.csv`), unless `overwrite` is set to `true`.
"""
mutable struct DataRecorderHook <: AbstractHook
    stats_getter
    smooth_over::Int
    csv::Union{Nothing, String}
    data::DataFrame
    function DataRecorderHook(stats_getter=nothing, csv=nothing; smooth_over=100,  overwrite=false)
        data = DataFrame(steps=Int[], episodes=Int[], R=Float64[],  R̄=Float64[], L=Int[])
        if !isnothing(csv)
            if isfile(csv) && !overwrite
                # increment the file name
                i = 1
                while isfile(csv * ".$i"); i += 1; end
                csv *= ".$i"
            end
            mkpath(dirname(csv))
            CSV.write(csv, data)
        end
        new(stats_getter, smooth_over, csv, data)
    end
end

function postepisode(drh::DataRecorderHook; steps, returns, lengths, kwargs...)
    episodes = length(returns)
    min_recs = drh.smooth_over
    R̄ =  length(returns) < min_recs ? mean(returns) : mean(returns[end-min_recs+1:end])
    R = returns[end]
    L = lengths[end]
    stats::Dict{Symbol, Any} = isnothing(drh.stats_getter) ? Dict{Symbol, Any}() : drh.stats_getter()
    num_cols = ncol(drh.data)
    push!(drh.data, (steps=steps, episodes=episodes, R̄=R̄, R=R, L=L, stats...), promote=true, cols=:union)
    new_num_cols = ncol(drh.data)
    if !isnothing(drh.csv)
        if new_num_cols > num_cols
            # new columns were added, so write CSV from scratch
            mkpath(dirname(drh.csv))
            CSV.write(drh.csv, drh.data)
        else
            # only the last row was updated, so append to CSV
            if !isfile(drh.csv)
                @warn "CSV file does not exist, did you delete it? Creating new file."
                mkpath(dirname(drh.csv))
                CSV.write(drh.csv, drh.data)
            else
                CSV.write(drh.csv, drh.data[end:end, :], append=true)
            end
        end
    end
    nothing
end


function postexperiment(drh::DataRecorderHook; kwargs...)
    if !isnothing(drh.csv)
        mkpath(dirname(drh.csv))
        CSV.write(drh.csv, drh.data)
    end
    nothing
end










using FileIO
using Colors
using Luxor

"""
    VideoRecorderHook(dirname, n; format="mp4", fps=30)

Hook that records a video of the environment every `n` episodes. The video is saved in the specified `dirname` directory. If the directory already exists, it will be deleted and overwritten. The video format can be either `mp4` or `gif`. The video is recorded at `fps` frames per second. Currently, `fps` is only supported for `gif` format.
"""
struct VideoRecorderHook <: AbstractHook
    dirname::String
    format::String
    n::Int
    fps::Int
    frames::Vector{Matrix{RGB{Colors.N0f8}}}
    function VideoRecorderHook(dirname, n; format="mp4", fps=30)
        @assert format ∈ ["mp4", "gif"] "Only mp4 or gif are supported"
        rm(dirname, recursive=true, force=true)
        mkpath(dirname)
        new(dirname, format, n, fps, [])
    end
end


function MDPs.preepisode(vr::VideoRecorderHook; env, kwargs...)
    empty!(vr.frames)
    push!(vr.frames, convert(Matrix{RGB{Colors.N0f8}}, visualize(env; kwargs...)))
    nothing
end

function MDPs.poststep(vr::VideoRecorderHook; env, returns, kwargs...)
    if length(returns) % vr.n == 0
        viz = convert(Matrix{RGB{Colors.N0f8}}, visualize(env; kwargs...))
        push!(vr.frames, viz)
    end
    nothing
end

function MDPs.postepisode(vr::VideoRecorderHook; steps, returns, kwargs...)
    if length(returns) % vr.n == 0
        fn = "$(vr.dirname)/ep-$(length(returns))-steps-$(steps)-return-$(returns[end]).$(vr.format)"
        if vr.format == "mp4"
            save(fn, vr.frames)
        elseif vr.format == "gif"
            save(fn, cat(vr.frames..., dims=3), fps=vr.fps)
        end
    end
    nothing
end

"""
    PlotHook(csvs, y, save_as; x=:episodes, dt=1.0, plot_kwargs...)

Hook that plots the `y` column of the CSV file(s) specified by `csvs` against the `x` column. The plot is saved as `save_as`. The `dt` parameter specifies the time interval between plot updates. The plot is generated using `compare_runs`. Any additional keyword arguments are ultimately passed to `compare_runs`.
"""
mutable struct PlotHook <: AbstractHook
    csvs::Union{Vector{String}, String}
    save_as::String
    x::Symbol
    y::Symbol
    plot_kwargs::Dict{Symbol, Any}
    dt::Float64
    tlast::Float64
    function PlotHook(csvs::Union{String, Vector{String}}, y::Symbol, save_as::String; x=:episodes, dt=1.0, plot_kwargs...)
        new(csvs, save_as, x, y, plot_kwargs, dt, -Inf)
    end
end

function make_and_save_plot(ph::PlotHook)
    csvs = ph.csvs isa String ? [ph.csvs] : ph.csvs
    compare_runs(csvs...; x=ph.x, y=ph.y, ph.plot_kwargs...)
    mkpath(dirname(ph.save_as))
    savefig(ph.save_as)
    nothing
end

function poststep(ph::PlotHook; kwargs...)
    if time() - ph.tlast > ph.dt
        make_and_save_plot(ph)
        ph.tlast = time()
    end
    nothing
end

function postexperiment(ph::PlotHook; kwargs...)
    make_and_save_plot(ph)
    nothing
end



"""
    PlotEverythignHook(csvs, save_dir, save_format="png"; x=:episodes, dt=1.0, plot_kwargs...)

Hook that plots all columns of the CSV file(s) specified by `csvs` against the `x` column. Each column plot is saved to a different file, having the same name as the column name. The plots are saved to `save_dir` directory in `save_format`. The `dt` parameter specifies the time interval between plot updates. Each plot is generated using `compare_runs`. Any additional keyword arguments are ultimately passed to `compare_runs`.
"""
mutable struct PlotEverythingHook <: AbstractHook
    csvs::Union{Vector{String}, String}
    save_dir::String
    save_format::String
    x::Symbol
    plot_kwargs::Dict{Symbol, Any}
    dt::Float64
    tlast::Float64
    function PlotEverythingHook(csvs::Union{String, Vector{String}}, save_dir::String, save_format="png"; x=:episodes, dt=1.0, plot_kwargs...)
        new(csvs, save_dir, save_format, x, plot_kwargs, dt, -Inf)
    end
end

function make_and_save_plot(ph::PlotEverythingHook)
    csvs = ph.csvs isa String ? [ph.csvs] : ph.csvs
    colnames = union([(isfile(csv) ? propertynames(CSV.read(csv, DataFrame)) : []) for csv in csvs]...)
    for y in colnames
        y == ph.x && continue
        compare_runs(csvs...; x=ph.x, y=y, ph.plot_kwargs...)
        mkpath(dirname(ph.save_dir))
        savefig(joinpath(ph.save_dir, "$(y).$(ph.save_format)"))
    end
    nothing
end

function poststep(ph::PlotEverythingHook; kwargs...)
    if time() - ph.tlast > ph.dt
        make_and_save_plot(ph)
        ph.tlast = time()
    end
    nothing
end

function postexperiment(ph::PlotEverythingHook; kwargs...)
    make_and_save_plot(ph)
    nothing
end