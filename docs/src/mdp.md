# Defining an MDP


## Spaces

```@docs
AbstractSpace
```

```@docs
IntegerSpace
```

```@docs
TensorSpace
```

```@docs
VectorSpace
```

```@docs
MatrixSpace
```

## AbstractMDP

```@docs
AbstractMDP
```

### Compulsory methods


```@docs
state_space
```

```@docs
action_space
```

```@docs
action_meaning
```

```@docs
action_meanings
```

### Methods to specify the dynamics

These methods need not be implemented if the MDP is to be used purely as a simulator i.e., RL environment.


```@docs
start_state_support
```

```@docs
start_state_probability
```

```@docs
start_state_distribution
```

```@docs
transition_support
```

```@docs
transition_probability
```

```@docs
transition_distribution
```

```@docs
reward(::AbstractMDP{S, A}, ::S, ::A, ::S) where {S, A}
```

```@docs

```

```@docs
is_absorbing
```

```@docs
visualize(::AbstractMDP{S, A}, ::S; kwargs...) where {S, A}
```


### Using the MDP as a simulator

If the dynamics are already specified, then the following methods are not required to be implemented as long as the MDP struct is mutable and has fields `state`, `action`, and `reward`.

```@docs
state
```

```@docs
action
```

```@docs
reward(::AbstractMDP{S, A}) where {S, A}
```

```@docs
factory_reset!
```

```@docs
reset!
```

```@docs
step!
```

```@docs
in_absorbing_state
```

```@docs
truncated
```

```@docs
visualize(::AbstractMDP{S, A}; kwargs...) where {S, A}
```


## Random Discrete MDPs

```@docs
RandomDiscreteMDP
```
