export interact, AbstractHook, preexperiment, preepisode, prestep, poststep, postepisode, postexperiment


abstract type AbstractHook end

preexperiment(::Union{AbstractPolicy, AbstractHook}; kwargs...) = nothing
preepisode(::Union{AbstractPolicy, AbstractHook}; kwargs...) = nothing
prestep(::Union{AbstractPolicy, AbstractHook}; kwargs...) = nothing
poststep(::Union{AbstractPolicy, AbstractHook}; kwargs...) = nothing
postepisode(::Union{AbstractPolicy, AbstractHook}; kwargs...) = nothing
postexperiment(::Union{AbstractPolicy, AbstractHook}; kwargs...) = nothing



function interact(env::AbstractMDP{S, A}, policy::AbstractPolicy{S, A}, γ::Real, horizon::Int, max_trials::Real, hooks...; max_steps::Real=Inf, rng::AbstractRNG=Random.GLOBAL_RNG, kwargs...)::Tuple{Vector{Float64}, Vector{Int}} where {S, A}
    steps::Int = 0
    lengths::Vector{Int} = Int[]
    returns::Vector{Float64} = Float64[]

    _preexperiment(hook) = preexperiment(hook; env, policy, max_steps, max_trials, rng, kwargs...)
    _preepisode(hook) = preepisode(hook; env, policy, steps, lengths, returns, max_steps, max_trials, rng, kwargs...)
    _prestep(hook) = prestep(hook; env, policy, steps, lengths, returns, max_steps, max_trials, rng, kwargs...)
    _poststep(hook) = poststep(hook; env, policy, steps, lengths, returns, max_steps, max_trials, rng, kwargs...)
    _postepisode(hook) = postepisode(hook; env, policy, steps, lengths, returns, max_steps, max_trials, rng, kwargs...)
    _postexperiment(hook) = postexperiment(hook; env, policy, steps, lengths, returns, max_steps, max_trials, rng, kwargs...)

    hooks = vcat(policy, collect(hooks))
    foreach(_preexperiment, hooks)
    while (steps < max_steps) && (length(returns) < max_trials)
        reset!(env; rng=rng)
        push!(lengths, 0)
        push!(returns, 0)
        foreach(_preepisode, hooks)
        while !in_absorbing_state(env) && (lengths[end] < horizon) && (steps < max_steps)
            foreach(_prestep, hooks)
            s = Tuple(state(env))
            a = policy(state(env))
            step!(env, a; rng=rng)
            steps += 1
            r = reward(env)
            s′ = Tuple(state(env))
            # println(reward(env))
            @debug "experience" s a r s′
            returns[end] += γ^(lengths[end]) * reward(env)
            lengths[end] += 1
            foreach(_poststep, hooks)
        end
        if in_absorbing_state(env) || (lengths[end] >= horizon)
            foreach(_postepisode, hooks)
        end
    end
    foreach(_postexperiment, hooks)

    return returns, lengths
end