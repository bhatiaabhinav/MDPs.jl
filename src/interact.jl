export interact, AbstractHook, preexperiment, preepisode, prestep, poststep, postepisode, postexperiment

"""
    AbstractHook

Abstract type for hooks that can be used for callbacks in the `interact` function.
"""
abstract type AbstractHook end

"""
    preexperiment(hook::Union{AbstractPolicy, AbstractHook};  env::AbstractMDP{S, A}, policy::AbstractPolicy{S, A}, max_steps::Real, max_trials::Real, horizon::Real, γ::Real, rng::AbstractRNG, kwargs...)::Nothing where {S, A}

Function called before the experiment starts.

# Arguments
- `env::AbstractMDP{S, A}`: the environment used in the experiment.
- `policy::AbstractPolicy{S, A}`: the policy used in the experiment.
- `max_steps::Real`: the maximum number of steps in the experiment.
- `max_trials::Real`: the maximum number of trials in the experiment.
- `horizon::Real`: the horizon used in the experiment.
- `γ::Real`: the discount factor used in the experiment.
- `rng::AbstractRNG`: the random number generator used in the experiment.
- `kwargs...`: additional keyword arguments passed to the `interact` function.

# Returns
- `Nothing`
"""
function preexperiment(::Union{AbstractPolicy, AbstractHook}; env::AbstractMDP{S, A}, policy::AbstractPolicy{S, A}, max_steps::Real, max_trials::Real, horizon::Real, γ::Real, rng::AbstractRNG, kwargs...)::Nothing where {S, A}
    nothing
end
"""
    preepisode(hook::Union{AbstractPolicy, AbstractHook}; env::AbstractMDP{S, A}, policy::AbstractPolicy{S, A}, steps::Int, lengths::Vector{Int}, returns::Vector{Float64}, max_steps::Real, max_trials::Real, horizon::Real, γ::Real, rng::AbstractRNG, kwargs...)::Nothing where {S, A}

Function called just after the environment is reset.

# Arguments
- `env::AbstractMDP{S, A}`: the environment used in the experiment.
- `policy::AbstractPolicy{S, A}`: the policy used in the experiment.
- `steps::Int`: the number of steps in the current experiment.
- `lengths::Vector{Int}`: the list of episode lengths so far. The last element is the number of steps in the current episode so far.
- `returns::Vector{Float64}`: the list of episode returns so far. The last element is the return of the current episode so far.
- `max_steps::Real`: the maximum number of steps in the experiment.
- `max_trials::Real`: the maximum number of trials (episodes) in the experiment.
- `horizon::Real`: the horizon used in the experiment.
- `γ::Real`: the discount factor used in the experiment.
- `rng::AbstractRNG`: the random number generator used in the experiment.
- `kwargs...`: additional keyword arguments passed to the `interact` function.

# Returns
- `Nothing`
"""
function preepisode(::Union{AbstractPolicy, AbstractHook}; env::AbstractMDP{S, A}, policy::AbstractPolicy{S, A}, steps::Int, lengths::Vector{Int}, returns::Vector{Float64}, max_steps::Real, max_trials::Real, horizon::Real, γ::Real, rng::AbstractRNG, kwargs...)::Nothing where {S, A}
    nothing
end
"""
    prestep(hook::Union{AbstractPolicy, AbstractHook}; env::AbstractMDP{S, A}, policy::AbstractPolicy{S, A}, steps::Int, lengths::Vector{Int}, returns::Vector{Float64}, max_steps::Real, max_trials::Real, horizon::Real, γ::Real, rng::AbstractRNG, kwargs...)::Nothing where {S, A}

Function called just before the policy is queried for an action.

# Arguments
- `env::AbstractMDP{S, A}`: the environment used in the experiment.
- `policy::AbstractPolicy{S, A}`: the policy used in the experiment.
- `steps::Int`: the number of steps in the current experiment.
- `lengths::Vector{Int}`: the list of episode lengths so far. The last element is the number of steps in the current episode so far.
- `returns::Vector{Float64}`: the list of episode returns so far. The last element is the return of the current episode so far.
- `max_steps::Real`: the maximum number of steps in the experiment.
- `max_trials::Real`: the maximum number of trials (episodes) in the experiment.
- `horizon::Real`: the horizon used in the experiment.
- `γ::Real`: the discount factor used in the experiment.
- `rng::AbstractRNG`: the random number generator used in the experiment.
- `kwargs...`: additional keyword arguments passed to the `interact` function.

# Returns
- `Nothing`
"""
function prestep(::Union{AbstractPolicy, AbstractHook}; env::AbstractMDP{S, A}, policy::AbstractPolicy{S, A}, steps::Int, lengths::Vector{Int}, returns::Vector{Float64}, max_steps::Real, max_trials::Real, horizon::Real, γ::Real, rng::AbstractRNG, kwargs...)::Nothing where {S, A}
    nothing
end
"""
    poststep(hook::Union{AbstractPolicy, AbstractHook}; env::AbstractMDP{S, A}, policy::AbstractPolicy{S, A}, steps::Int, lengths::Vector{Int}, returns::Vector{Float64}, max_steps::Real, max_trials::Real, horizon::Real, γ::Real, rng::AbstractRNG, kwargs...)::Nothing where {S, A}

Function called just after the environment is stepped.

# Arguments
- `env::AbstractMDP{S, A}`: the environment used in the experiment.
- `policy::AbstractPolicy{S, A}`: the policy used in the experiment.
- `steps::Int`: the number of steps in the current experiment.
- `lengths::Vector{Int}`: the list of episode lengths so far. The last element is the number of steps in the current episode so far.
- `returns::Vector{Float64}`: the list of episode returns so far. The last element is the return of the current episode so far.
- `max_steps::Real`: the maximum number of steps in the experiment.
- `max_trials::Real`: the maximum number of trials (episodes) in the experiment.
- `horizon::Real`: the horizon used in the experiment.
- `γ::Real`: the discount factor used in the experiment.
- `rng::AbstractRNG`: the random number generator used in the experiment.
- `kwargs...`: additional keyword arguments passed to the `interact` function.

# Returns
- `Nothing`
"""
function poststep(::Union{AbstractPolicy, AbstractHook}; env::AbstractMDP{S, A}, policy::AbstractPolicy{S, A}, steps::Int, lengths::Vector{Int}, returns::Vector{Float64}, max_steps::Real, max_trials::Real, horizon::Real, γ::Real, rng::AbstractRNG, kwargs...)::Nothing where {S, A}
    nothing
end
"""
    postepisode(hook::Union{AbstractPolicy, AbstractHook}; env::AbstractMDP{S, A}, policy::AbstractPolicy{S, A}, steps::Int, lengths::Vector{Int}, returns::Vector{Float64}, max_steps::Real, max_trials::Real, horizon::Real, γ::Real, rng::AbstractRNG, kwargs...)::Nothing where {S, A}

Function called after each episode ends, which is when `in_absorbing_state` is `true` or when `truncated` is `true` or if the number of steps in the episode equals `horizon`.
Note that if the experiment is terminated prematurely due to `max_steps`, this function is not called if the last episode is not ended.

# Arguments
- `env::AbstractMDP{S, A}`: the environment used in the experiment.
- `policy::AbstractPolicy{S, A}`: the policy used in the experiment.
- `steps::Int`: the number of steps in the current experiment.
- `lengths::Vector{Int}`: the list of episode lengths so far. The last element is the number of steps in the current episode so far.
- `returns::Vector{Float64}`: the list of episode returns so far. The last element is the return of the current episode so far.
- `max_steps::Real`: the maximum number of steps in the experiment.
- `max_trials::Real`: the maximum number of trials (episodes) in the experiment.
- `horizon::Real`: the horizon used in the experiment.
- `γ::Real`: the discount factor used in the experiment.
- `rng::AbstractRNG`: the random number generator used in the experiment.
- `kwargs...`: additional keyword arguments passed to the `interact` function.

# Returns
- `Nothing`
"""
function postepisode(::Union{AbstractPolicy, AbstractHook}; env::AbstractMDP{S, A}, policy::AbstractPolicy{S, A}, steps::Int, lengths::Vector{Int}, returns::Vector{Float64}, max_steps::Real, max_trials::Real, horizon::Real, γ::Real, rng::AbstractRNG, kwargs...)::Nothing where {S, A}
    nothing
end
"""
    postexperiment(hook::Union{AbstractPolicy, AbstractHook}; env::AbstractMDP{S, A}, policy::AbstractPolicy{S, A}, steps::Int, lengths::Vector{Int}, returns::Vector{Float64}, max_steps::Real, max_trials::Real, horizon::Real, γ::Real, rng::AbstractRNG, kwargs...)::Nothing where {S, A}

Function called after the experiment ends.

# Arguments
- `env::AbstractMDP{S, A}`: the environment used in the experiment.
- `policy::AbstractPolicy{S, A}`: the policy used in the experiment.
- `steps::Int`: the number of steps in the current experiment.
- `lengths::Vector{Int}`: the list of episode lengths.
- `returns::Vector{Float64}`: the list of episode returns.
- `max_steps::Real`: the maximum number of steps in the experiment.
- `max_trials::Real`: the maximum number of trials (episodes) in the experiment.
- `horizon::Real`: the horizon used in the experiment.
- `γ::Real`: the discount factor used in the experiment.
- `rng::AbstractRNG`: the random number generator used in the experiment.
- `kwargs...`: additional keyword arguments passed to the `interact` function.

# Returns
- `Nothing`
"""
function postexperiment(::Union{AbstractPolicy, AbstractHook}; env::AbstractMDP{S, A}, policy::AbstractPolicy{S, A}, steps::Int, lengths::Vector{Int}, returns::Vector{Float64}, max_steps::Real, max_trials::Real, horizon::Real, γ::Real, rng::AbstractRNG, kwargs...)::Nothing where {S, A}
    nothing
end










"""
    interact(env::AbstractMDP, policy::AbstractPolicy, γ::Real, horizon::Real, max_trials::Real, hooks...; max_steps::Real=Inf, rng::AbstractRNG=Random.GLOBAL_RNG, kwargs...)::Tuple{Vector{Float64}, Vector{Int}}

Run an experiment on the environment `env` using the policy `policy` for `max_trials` episodes, or until overall `max_steps` steps have been taken in the experiment. `γ` is the discount factor used when computing returns. Each episode/trial is truncated at `horizon` steps if it does not terminate automatically due to `in_absorbing_state` or `truncated` functions of the environment. An random number generator `rng` is used to generate the random numbers in the experiment. The experiment is run with `hook` objects on which the following functions are called at different points in the experiment:

1. `preexperiment(hook; ...)`: called before the experiment starts
2. `preepisode(hook; ...)`: called just after resetting the environment
3. `prestep(hook; ...)` : called just before the policy is queried for an action
4. `poststep(hook; ...)`: called just after the environment is stepped
5. `postepisode(hook; ...)`: called just after an episode ends. Note that if the experiment is ended prematurely due to the total number of steps exceeding `max_steps`, this hook is not called after the last episode if it was not properly terminated.
6. `postexperiment(hook; ...)`: called after the experiment ends

The order in which the hooks are called for any of these functions is same as the order in which they are passed to the `interact` function. Note that the `policy` is also considered a hook and has the highest priority.

The hooks are called with the following keyword arguments:

- `env`: the environment used in the experiment
- `policy`: the policy used in the experiment
- `steps`: the number of steps taken in the experiment
- `lengths`: the lengths of the episodes in the experiment. The last element of this vector is the number of steps taken so far in the current episode.
- `returns`: the returns of the episodes in the experiment. The last element of this vector is the return so far in the current episode.
- `max_steps`: the maximum number of steps allowed in the experiment
- `max_trials`: the maximum number of episodes allowed in the experiment
- `horizon`: the horizon of the problem.
- `γ`: the discount factor used in the experiment to compute returns
- `rng`: the random number generator used in the experiment
- `kwargs...`: any additional keyword arguments passed to the `interact` function

The `interact` function returns a tuple containing the returns and lengths of the episodes in the experiment.

# arguments
- `env::AbstractMDP`: the environment to run the experiment on
- `policy::AbstractPolicy`: the policy to use in the experiment
- `γ::Real`: the discount factor to use when computing returns
- `horizon::Real`: the horizon of the problem.
- `max_trials::Real`: the maximum number of episodes to run
- `hooks...`: the hooks to call during the experiment
- `max_steps::Real=Inf`: the maximum number of steps to run in the experiment across all episodes
- `rng::AbstractRNG=Random.GLOBAL_RNG`: the random number generator to use in the experiment
- `kwargs...`: any additional keyword arguments to pass to the hooks

# returns
- `returns::Vector{Float64}`: the returns of the episodes in the experiment
- `lengths::Vector{Int}`: the lengths of the episodes in the experiment

# example
```julia
using MDPs
mdp = RandomDiscreteMDP(10, 2)  # 10 states, 2 actions
policy = RandomPolicy(mdp)
γ = 1.0
horizon = 100
max_trials = 1000
returns, lengths = interact(mdp, policy, γ, horizon, max_trials)
println("Average return: ", sum(returns) / length(returns))
```
----
"""
function interact(env::AbstractMDP{S, A}, policy::AbstractPolicy{S, A}, γ::Real, horizon::Real, max_trials::Real, hooks...; max_steps::Real=Inf, rng::AbstractRNG=Random.GLOBAL_RNG, kwargs...)::Tuple{Vector{Float64}, Vector{Int}} where {S, A}
    steps::Int = 0
    lengths::Vector{Int} = Int[]
    returns::Vector{Float64} = Float64[]

    _preexperiment(hook) = preexperiment(hook; env, policy, max_steps, max_trials, horizon, γ, rng, kwargs...)
    _preepisode(hook) = preepisode(hook; env, policy, steps, lengths, returns, max_steps, max_trials, horizon, γ, rng, kwargs...)
    _prestep(hook) = prestep(hook; env, policy, steps, lengths, returns, max_steps, max_trials, horizon, γ, rng, kwargs...)
    _poststep(hook) = poststep(hook; env, policy, steps, lengths, returns, max_steps, max_trials, horizon, γ, rng, kwargs...)
    _postepisode(hook) = postepisode(hook; env, policy, steps, lengths, returns, max_steps, max_trials, horizon, γ, rng, kwargs...)
    _postexperiment(hook) = postexperiment(hook; env, policy, steps, lengths, returns, max_steps, max_trials, horizon, γ, rng, kwargs...)

    factory_reset!(env)

    hooks = vcat(policy, collect(hooks))
    foreach(_preexperiment, hooks)
    while (steps < max_steps) && (length(returns) < max_trials)
        reset!(env; rng=rng)
        push!(lengths, 0)
        push!(returns, 0)
        foreach(_preepisode, hooks)
        while !in_absorbing_state(env) && !truncated(env) && (lengths[end] < horizon) && (steps < max_steps)
            foreach(_prestep, hooks)
            s = Tuple(state(env))
            # println("here")
            a = policy(rng, state(env))
            step!(env, a; rng=rng)
            steps += 1
            r = reward(env)
            s′ = Tuple(state(env))
            @debug "experience" s a r s′
            returns[end] += γ^(lengths[end]) * reward(env)
            lengths[end] += 1
            foreach(_poststep, hooks)
        end
        if in_absorbing_state(env) || (lengths[end] >= horizon) || truncated(env)
            foreach(_postepisode, hooks)
        end
    end
    foreach(_postexperiment, hooks)

    return returns, lengths
end