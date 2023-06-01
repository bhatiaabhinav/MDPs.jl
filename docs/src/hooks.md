# Hooks

## Defining a hook

To implement a hook, inherit from the `AbstractHook` type and implement the following functions. Note that `AbstractPolicy` can also behave as a hook.

Generally, RL learning algorithms are implemented using hooks. For example, a [Q-Learning](https://github.com/bhatiaabhinav/TabularRL.jl/blob/main/src/q_learning.jl) hook would update the Q-values in the `poststep` call.

```@docs
AbstractHook
```

```@docs
preexperiment
```

```@docs
preepisode
```

```@docs
prestep
```

```@docs
poststep
```

```@docs
postepisode
```

```@docs
postexperiment
```

## Inbuilt hooks

```@docs
EmptyHook
```

```@docs
EmpiricalPolicyEvaluationHook
```

```@docs
ProgressMeterHook
```
